"""Harvest trade-proposal decisions from real teacher-driven play.

Why not a wider synthetic generator
-----------------------------------
D2.7 records four separate occasions where a synthetic sample produced a
confident wrong conclusion, the last with 402 states and tight intervals. The
problem was never sample size; it was that generated boards are not the boards
real play produces. Widening the generator to ~500 proposal boards would buy
statistical power over the wrong distribution.

So this collects the same decisions from actual games: the teacher drives,
and every state where at least two `exch_trade` actions are legal is recorded
along with the full candidate set and what the teacher actually did. That is
simultaneously the ranking data (which candidate, when it proposes) and the
gate data (whether it proposes at all), drawn from the distribution the policy
will be judged on.

Split discipline: by **game seed**, not by decision. Decisions within one game
share a board and are heavily correlated, so a decision-level split would leak
and inflate held-out accuracy.

Output
------
One gzipped shard per game under `probes/trade_shards/`, written as the game
ends. Resume is "does the shard exist", which is the D0.8 rule: the original
single-file version buffered everything until the end, and at the 1,000-game
scale Part C needs that is a 2-3 hour run with nothing recoverable if it dies.
The legacy single file `probes/trade_harvest.jsonl` (seeds 910000-910119) is
still read by the analysis scripts alongside the shards.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ASU_FROZEN_TEACHER import ASUValueV1  # noqa: E402
from ASU_FROZEN_TEACHER.evaluate import _ScriptedAdapter, _new_seeded_game  # noqa: E402
from monopoly_game_engine.actions import OFFSETS  # noqa: E402
from monopoly_game_engine.agents_fixed import FP_AGENT_CLASSES  # noqa: E402
from monopoly_game_engine.constants import COLOR_GROUPS, PROPERTIES, PROPERTY_IDS  # noqa: E402

from competition_agent.proc import managed_pool  # noqa: E402
from competition_agent.spec_model import (  # noqa: E402
    SHORT_TURNS, multi_turn_landings, rent_for,
)

OUT = Path(__file__).resolve().parent / "probes" / "trade_harvest.jsonl"
SHARDS = Path(__file__).resolve().parent / "probes" / "trade_shards"
N = len(PROPERTY_IDS)


def decode(action, others):
    loc = action - OFFSETS["exch_trade"]
    p = loc // (N * (N - 1))
    rem = loc % (N * (N - 1))
    off = rem // (N - 1)
    rr = rem % (N - 1)
    req = rr if rr < off else rr + 1
    if p >= len(others):
        return None
    return others[p], PROPERTY_IDS[off], PROPERTY_IDS[req]


def deed_facts(env, pid, sq):
    """Raw, model-free quantities. No valuation is baked in here, so the
    analysis can test hypotheses this file does not presuppose."""
    prop = env.properties[sq]
    color = PROPERTIES[sq]["color"]
    group = COLOR_GROUPS[color]
    owner = prop.owner

    saved = prop.owner
    prop.owner = pid
    try:
        rent_if_ours = 0.0
        for opp in env.players:
            if opp.player_id == pid or opp.bankrupt:
                continue
            for land, p in multi_turn_landings(opp.position, SHORT_TURNS):
                if land == sq:
                    rent_if_ours += p * rent_for(env, sq)
    finally:
        prop.owner = saved

    return {
        "sq": sq,
        "price": PROPERTIES[sq]["price"],
        "base_rent": PROPERTIES[sq]["rent"][0],
        "color": color,
        "group_size": len(group),
        "ours_in_group": sum(1 for s in group
                             if env.properties[s].owner == pid),
        "theirs_in_group": sum(1 for s in group
                               if env.properties[s].owner not in (None, pid)),
        "owner": -1 if owner is None else owner,
        "mortgaged": bool(prop.mortgaged),
        "houses": prop.houses,
        "rent_if_ours": rent_if_ours,
        "is_monopoly": bool(prop.is_monopoly),
    }


def seats_snapshot(env):
    """Per-seat cash / holdings at the moment of the decision.

    Absent from the original harvest, which is why Part B could say nothing
    about the counterparty. A proposal's fate depends on who is being asked,
    and `tgt` alone does not carry that. Cheap to record, impossible to
    reconstruct later.
    """
    return {
        str(p.player_id): {
            "cash": float(p.cash),
            "deeds": len(p.properties),
            "net": float(p.net_worth()),
            "bankrupt": bool(p.bankrupt),
            "pos": int(p.position),
        }
        for p in env.players
    }


def _play_shard(job):
    """One game -> one gzipped shard. Returns counts, never the records."""
    seed, max_steps = job
    out = SHARDS / f"g{seed}.jsonl.gz"
    if out.exists():
        return seed, 0, 0, True
    rows = _play((seed, max_steps))
    tmp = out.with_suffix(".gz.tmp")
    with gzip.open(tmp, "wt") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    tmp.rename(out)                       # atomic: a killed run leaves no
    return (seed, len(rows),              # half-written shard that resume
            sum(1 for r in rows if r["proposed"]), False)   # would trust


def _play(seed_maxsteps):
    seed, max_steps = seed_maxsteps
    game = _new_seeded_game(seed)
    env = game.env
    teacher = ASUValueV1(0)
    opp = {i: _ScriptedAdapter(FP_AGENT_CLASSES[i - 1](i), i) for i in (1, 2, 3)}
    others = [1, 2, 3]
    out, steps = [], 0

    while steps < max_steps and not env.done:
        actor = env.whose_turn()
        if actor == 0:
            legal = [int(a) for a in env.get_allowed_actions(0)]
            cands = [a for a in legal
                     if OFFSETS["exch_trade"] <= a < OFFSETS["auction"]]
            chosen = int(teacher.choose_action(env))
            if len(cands) >= 2:
                facts = {}

                def f(sq):
                    if sq not in facts:
                        facts[sq] = deed_facts(env, 0, sq)
                    return facts[sq]

                rows = []
                for a in cands:
                    d = decode(a, others)
                    if d is None:
                        continue
                    tgt, off, req = d
                    rows.append({"a": a, "tgt": tgt,
                                 "off": f(off), "req": f(req)})
                if rows:
                    out.append({
                        # Full 300-dim observation, added for Candidate D: the
                        # previous harvest carried only hand-picked deed
                        # features, which is the cap D2.5 and the Phase 3 head
                        # both ran into. One vector per STATE, not per
                        # candidate - candidates share the state.
                        "obs": [round(float(x), 4)
                                for x in env._get_state(0).tolist()],
                        "seed": seed, "step": steps,
                        "phase": env.phase,
                        "round": int(getattr(env, "round", -1)),
                        "our_cash": env.players[0].cash,
                        "our_deeds": len(env.players[0].properties),
                        "seats": seats_snapshot(env),
                        "chosen": chosen,
                        "proposed": chosen in cands,
                        "n_cands": len(rows),
                        "cands": rows,
                    })
            action = chosen
        else:
            action = int(opp[actor].choose_action(env))
        game.step(action)
        steps += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--seed-base", type=int, default=910000)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    SHARDS.mkdir(parents=True, exist_ok=True)
    jobs = [(args.seed_base + k, args.max_steps) for k in range(args.seeds)]
    print(f"{len(jobs)} game(s) requested; "
          f"{sum(1 for j in jobs if (SHARDS / f'g{j[0]}.jsonl.gz').exists())} "
          f"shard(s) already present")

    t0 = time.time()
    states = props = reused = done = 0
    with managed_pool(args.workers) as pool:
        for seed, n, p, cached in pool.imap_unordered(_play_shard, jobs):
            states += n
            props += p
            reused += int(cached)
            done += 1
            if done % 20 == 0 or done == len(jobs):
                el = (time.time() - t0) / 60
                rate = done / max(el, 1e-9)
                print(f"  {done}/{len(jobs)} games  {states} states  "
                      f"{props} proposals  {rate:.1f} g/min  "
                      f"ETA {(len(jobs)-done)/max(rate,1e-9):.0f} min",
                      flush=True)

    files = sorted(SHARDS.glob("g*.jsonl.gz"))
    mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"\nshards {len(files)}  ({mb:.0f} MB gzipped)  reused {reused}")
    print(f"decision states with >=2 exchange candidates : {states}")
    print(f"  of which the teacher proposed             : {props} "
          f"({100*props/max(states,1):.1f}%)")
    print(f"wrote {SHARDS}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
