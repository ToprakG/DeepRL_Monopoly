"""Measure one agent configuration against one field. Paired by seed.

Why one configuration per process
---------------------------------
`TRADE_WEIGHTS` and `TRADE_RANKER` are read when `spec_policy` imports, so two
configurations cannot coexist in one interpreter. Each is therefore run as its
own invocation over the same seed set, and `pair_runs.py` pairs the outputs.
The pairing is what makes 2,000 games enough: board luck cancels when both arms
play the same seed from the same seat, and the harness's own noise floor was
measured at 0 flipped games in 2,000 (D7.4).

Fields
------
    asu      2 agent seats vs 2 ASUValueV1, parity 50%   - the record's 27.5%
    strong   1 vs fixed-b/d/e   (~1252 ELO), parity 25%  - the record's 34.2%
    weak     1 vs fixed-a/b/c   (~1103 ELO), parity 25%  - the record's 55.8%

The 2v2 shape for ASU is kept because that is how 27.5% was measured; changing
it would make the new number incomparable to the one on the record.

Proposal accounting
-------------------
Every run also counts trade proposals made and accepted. A ranking improvement
can only pay off through proposals that are accepted, so if the field declines
almost everything, the ranking cannot matter however good it is — and that
would explain a top-1 gain failing to reach the win rate rather than leaving it
unexplained. Collected here so it costs no extra games.
"""

from __future__ import annotations

import argparse
import json
import math
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

from monopoly_game_engine.actions import OFFSETS, ActionType  # noqa: E402

from competition_agent.policies import build_policy  # noqa: E402
from competition_agent.proc import ensure_hash_seed, managed_pool  # noqa: E402

FIELDS = {
    "asu": ("teacher", "teacher"),
    "strong": ("fixed-b", "fixed-d", "fixed-e"),
    "weak": ("fixed-a", "fixed-b", "fixed-c"),
}
ACCEPT = int(ActionType.ACCEPT_TRADE)
DECLINE = int(ActionType.DECLINE_TRADE)


def seats_for(field, seed):
    """Which seats the agent occupies, rotated so seat effects cancel."""
    if field == "asu":
        return (0, 2) if seed % 2 == 0 else (1, 3)
    return (seed % 4,)


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - m), min(1.0, c + m)


def _game(job):
    seed, field, max_steps = job
    from ASU_FROZEN_TEACHER.evaluate import _new_seeded_game
    mine = seats_for(field, seed)
    game = _new_seeded_game(seed)
    env = game.env

    agents, fill = {}, list(FIELDS[field])
    for s in range(4):
        agents[s] = (build_policy("final", s, seed * 4 + s) if s in mine
                     else build_policy(fill.pop(0), s, seed * 4 + s))

    steps, proposed, accepted, declined = 0, 0, 0, 0
    pending = False
    t0 = time.perf_counter()
    while not env.done and steps < max_steps:
        actor = env.whose_turn()
        a = int(agents[actor].choose_action(env))
        if actor in mine and OFFSETS["exch_trade"] <= a < OFFSETS["auction"]:
            proposed += 1
            pending = True
        elif pending and actor not in mine and a in (ACCEPT, DECLINE):
            accepted += a == ACCEPT
            declined += a == DECLINE
            pending = False
        game.step(a)
        steps += 1

    active = [p.player_id for p in env.players if not p.bankrupt]
    decisive = len(active) == 1
    won = env.winner() in mine
    return {
        "arm": "run", "seed": seed, "field": field, "seats": list(mine),
        "steps": steps, "decisive": decisive,
        "leader_win": won,
        "decisive_win": decisive and active[0] in mine,
        "bankrupt": all(env.players[s].bankrupt for s in mine),
        "any_bankrupt": any(env.players[s].bankrupt for s in mine),
        "proposed": proposed, "accepted": accepted, "declined": declined,
        "seconds": time.perf_counter() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", choices=sorted(FIELDS), required=True)
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--seed-base", type=int, default=960000)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    ensure_hash_seed()

    out = (Path(__file__).resolve().parent / "probes"
           / f"field_{args.field}_{args.tag}.json")
    partial = out.with_suffix(".partial.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    done = {}
    if partial.exists() and not args.no_resume:
        for line in partial.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["seed"]] = r
        print(f"resuming: {len(done)} game(s) recorded")

    jobs = [(args.seed_base + k, args.field, args.max_steps)
            for k in range(args.games)]
    todo = [j for j in jobs if j[0] not in done]
    rows = [r for r in done.values()
            if r["seed"] - args.seed_base < args.games]
    print(f"field {args.field} {FIELDS[args.field]}   "
          f"TRADE_RANKER={os.environ.get('TRADE_RANKER', '-')}   "
          f"TRADE_WEIGHTS={os.environ.get('TRADE_WEIGHTS', '-')}")
    print(f"{len(todo)} to play, {len(rows)} reused")

    t0 = time.time()
    if todo:
        with partial.open("a") as sink, managed_pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_game, todo), 1):
                rows.append(r)
                sink.write(json.dumps(r) + "\n")
                sink.flush()
                if i % 100 == 0 or i == len(todo):
                    el = (time.time() - t0) / 60
                    rate = i / max(el, 1e-9)
                    print(f"  {i}/{len(todo)}  {rate:.0f} g/min  "
                          f"ETA {(len(todo)-i)/max(rate,1e-9):.1f} min",
                          flush=True)
    out.write_text(json.dumps(rows, indent=1))

    n = len(rows)
    k = sum(r["leader_win"] for r in rows)
    p, lo, hi = wilson(k, n)
    prop = sum(r["proposed"] for r in rows)
    acc = sum(r["accepted"] for r in rows)
    dec = sum(r["declined"] for r in rows)
    parity = 50.0 if args.field == "asu" else 25.0
    print(f"\n{args.field}: {k}/{n}  {100*p:.2f}%  [{100*lo:.2f}, {100*hi:.2f}]"
          f"   parity {parity:.0f}%")
    print(f"  bankrupt (all our seats) {100*sum(r['bankrupt'] for r in rows)/n:.1f}%"
          f"   decisive {100*sum(r['decisive'] for r in rows)/n:.1f}%")
    print(f"  proposals made {prop} ({prop/n:.2f}/game)   "
          f"accepted {acc} ({100*acc/max(prop,1):.1f}%)   declined {dec}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
