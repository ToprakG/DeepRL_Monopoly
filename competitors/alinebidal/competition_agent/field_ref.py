"""Seat a reference policy against a field, on the same convention as field_ab.

Purpose
-------
`37.95%` against the strong field means nothing on its own. If the frozen
teacher also scores ~38% there, the agent is already at teacher level on that
field and the headroom Candidate D failed to find does not exist; if the
teacher scores 60%, the headroom is real and we are not reaching it. The
project rule is that a bench result is not interpreted without the opponents'
strength, and this is the same rule applied to the ceiling rather than the
floor.

Kept separate from `field_ab.py` rather than adding a `--policy` flag to it,
because that file is in the import path of a running measurement and
`mp.Pool` respawns dead workers by re-importing from disk.

Seat convention is copied exactly from `field_ab`: `seat = seed % 4` for a
1-vs-3 field, so the reference policy is measured over the same seats, the
same seeds and the same rotation as the agent it is being compared to.

Proposal accounting is kept for the same reason it is in `field_ab`: against
the strong field the agent's proposals are accepted 0.06% of the time, and
whether the teacher fares any better decides whether that is a defect in our
proposals or a property of the field.
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
    "strong": ("fixed-b", "fixed-d", "fixed-e"),
    "weak": ("fixed-a", "fixed-b", "fixed-c"),
}
ACCEPT = int(ActionType.ACCEPT_TRADE)
DECLINE = int(ActionType.DECLINE_TRADE)


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - m), min(1.0, c + m)


def _game(job):
    seed, field, policy, max_steps = job
    from ASU_FROZEN_TEACHER.evaluate import _new_seeded_game
    seat = seed % 4                       # field_ab's convention, unchanged
    game = _new_seeded_game(seed)
    env = game.env
    agents, fill = {}, list(FIELDS[field])
    for s in range(4):
        agents[s] = (build_policy(policy, s, seed * 4 + s) if s == seat
                     else build_policy(fill.pop(0), s, seed * 4 + s))

    steps, proposed, accepted, declined = 0, 0, 0, 0
    pending = False
    t0 = time.perf_counter()
    while not env.done and steps < max_steps:
        actor = env.whose_turn()
        a = int(agents[actor].choose_action(env))
        if actor == seat and OFFSETS["exch_trade"] <= a < OFFSETS["auction"]:
            proposed += 1
            pending = True
        elif pending and actor != seat and a in (ACCEPT, DECLINE):
            accepted += a == ACCEPT
            declined += a == DECLINE
            pending = False
        game.step(a)
        steps += 1

    active = [p.player_id for p in env.players if not p.bankrupt]
    decisive = len(active) == 1
    return {
        "arm": "run", "seed": seed, "field": field, "policy": policy,
        "seats": [seat], "steps": steps, "decisive": decisive,
        "leader_win": env.winner() == seat,
        "decisive_win": decisive and active[0] == seat,
        "bankrupt": bool(env.players[seat].bankrupt),
        "proposed": proposed, "accepted": accepted, "declined": declined,
        "seconds": time.perf_counter() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", choices=sorted(FIELDS), required=True)
    ap.add_argument("--policy", default="teacher")
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--seed-base", type=int, default=960000)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=2)
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
        print(f"resuming: {len(done)} recorded")

    jobs = [(args.seed_base + k, args.field, args.policy, args.max_steps)
            for k in range(args.games)]
    todo = [j for j in jobs if j[0] not in done]
    rows = [r for r in done.values()
            if r["seed"] - args.seed_base < args.games]
    print(f"{args.policy} vs {FIELDS[args.field]}   "
          f"{len(todo)} to play, {len(rows)} reused")

    t0 = time.time()
    if todo:
        with partial.open("a") as sink, managed_pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_game, todo), 1):
                rows.append(r)
                sink.write(json.dumps(r) + "\n")
                sink.flush()
                if i % 50 == 0 or i == len(todo):
                    el = (time.time() - t0) / 60
                    rate = i / max(el, 1e-9)
                    print(f"  {i}/{len(todo)}  {rate:.0f} g/min  "
                          f"ETA {(len(todo)-i)/max(rate,1e-9):.0f} min",
                          flush=True)
    out.write_text(json.dumps(rows, indent=1))

    n = len(rows)
    k = sum(r["leader_win"] for r in rows)
    p, lo, hi = wilson(k, n)
    prop = sum(r["proposed"] for r in rows)
    acc = sum(r["accepted"] for r in rows)
    print(f"\n{args.policy} vs {args.field}: {k}/{n}  {100*p:.2f}%  "
          f"[{100*lo:.2f}, {100*hi:.2f}]   parity 25%")
    print(f"  bankrupt {100*sum(r['bankrupt'] for r in rows)/n:.1f}%   "
          f"decisive {100*sum(r['decisive'] for r in rows)/n:.1f}%")
    print(f"  proposals {prop} ({prop/n:.2f}/game)  "
          f"accepted {acc} ({100*acc/max(prop,1):.1f}%)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
