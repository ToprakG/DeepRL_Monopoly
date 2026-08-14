"""Phase 6 Part A — collect states labelled with the outcome that follows them.

Why the label changes
---------------------
Every previous attempt trained against the teacher's *action choices* and
gained agreement without gaining win rate. Three agreement-to-strength
projections were made and all three were wrong. This trains against the thing
actually being optimised: whether the seat went on to win.

What is recorded
----------------
Teacher-vs-teacher games (`ASUValueV1` on all four seats), every decision
point, from the perspective of the acting seat:

    obs      300-dim observation for that seat        (float16)
    seat     which seat acted
    step     decision index within the game
    game     game id (the seed) — the unit the train/held-out split uses
    win      1 if that seat won the game, else 0      <- the target
    rank     finishing rank by net worth, 1 = best    <- auxiliary, Part B
    net      final net worth of that seat             <- auxiliary, Part B

`rank` and `net` are cheaper signals than the binary win and are kept for the
shaped-label variant in Part B.

Storage
-------
One compressed `.npz` per game, written as soon as that game ends. Resume is
"does the file exist" — nothing is held in memory across games, which is the
D0.8 rule after three long runs were lost to `pool.map` buffering. Observations
are float16: the engine's state vector is normalised into [0,1] ranges, so the
precision loss is far below the noise in a binary outcome label, and it halves
a corpus that would otherwise run to hundreds of megabytes.

Sanity check
------------
Across four seats and one winner per game, the mean of `win` must land near
0.25. The script asserts this at the end; a different value means the label
join is wrong, not that the agent is unusual.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ASU_FROZEN_TEACHER import ASUValueV1  # noqa: E402
from ASU_FROZEN_TEACHER.evaluate import _new_seeded_game  # noqa: E402

from competition_agent.proc import managed_pool  # noqa: E402

OUTDIR = Path(__file__).resolve().parent / "probes" / "outcomes"


def _play(job):
    seed, max_steps = job
    out = OUTDIR / f"g{seed}.npz"
    if out.exists():
        return seed, 0, True                       # already collected

    game = _new_seeded_game(seed)
    env = game.env
    agents = {s: ASUValueV1(s) for s in range(4)}

    obs_l, seat_l, step_l = [], [], []
    steps = 0
    while not env.done and steps < max_steps:
        actor = env.whose_turn()
        legal = env.get_allowed_actions(actor)
        # Only decision points: a forced move carries no signal about choice,
        # though it still carries signal about the position. Keep them — the
        # value function scores STATES, not choices, so forced states are as
        # informative as any other.
        obs_l.append(np.asarray(env._get_state(actor), dtype=np.float16))
        seat_l.append(actor)
        step_l.append(steps)
        try:
            game.step(int(agents[actor].choose_action(env)))
        except Exception:                                  # noqa: BLE001
            break
        steps += 1

    # ---- outcome, resolved once the game is over ----------------------
    nets = np.array([p.net_worth() for p in env.players], dtype=np.float32)
    alive = [p.player_id for p in env.players if not p.bankrupt]
    winner = alive[0] if len(alive) == 1 else int(np.argmax(nets))
    # rank 1 = best net worth
    order = np.argsort(-nets)
    rank = np.empty(4, dtype=np.int8)
    for r, pid in enumerate(order):
        rank[pid] = r + 1

    seat_arr = np.array(seat_l, dtype=np.int8)
    np.savez_compressed(
        out,
        obs=np.stack(obs_l) if obs_l else np.zeros((0, 300), np.float16),
        seat=seat_arr,
        step=np.array(step_l, dtype=np.int32),
        win=(seat_arr == winner).astype(np.int8),
        rank=rank[seat_arr],
        net=nets[seat_arr],
        n_steps=np.int32(steps),
    )
    return seed, len(obs_l), False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--seed-base", type=int, default=700000)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    jobs = [(args.seed_base + k, args.max_steps) for k in range(args.games)]

    import time
    t0 = time.time()
    rows = reused = 0
    with managed_pool(args.workers) as pool:
        for i, (seed, n, was_cached) in enumerate(
            pool.imap_unordered(_play, jobs), 1
        ):
            rows += n
            reused += int(was_cached)
            if i % 25 == 0 or i == len(jobs):
                el = (time.time() - t0) / 60
                rate = i / max(el, 1e-9)
                print(f"  {i}/{len(jobs)} games  {rows} new rows  "
                      f"{rate:.1f} g/min  ETA {(len(jobs)-i)/max(rate,1e-9):.0f} min",
                      flush=True)

    # ---- sanity: one winner in four seats -> mean(win) ~= 0.25 ---------
    files = sorted(OUTDIR.glob("g*.npz"))
    tot = wins = 0
    for f in files[:200]:                      # a sample is enough to catch a join bug
        d = np.load(f)
        tot += len(d["win"])
        wins += int(d["win"].sum())
    frac = wins / max(tot, 1)
    print(f"\nfiles {len(files)}   sampled rows {tot}   mean(win) = {frac:.4f}")
    if not 0.15 <= frac <= 0.35:
        print("FAIL: mean(win) is far from 0.25 — the label join is wrong. "
              "Do not train on this corpus.")
        return 1
    print("sanity OK (mean win near 0.25 as four seats and one winner require)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
