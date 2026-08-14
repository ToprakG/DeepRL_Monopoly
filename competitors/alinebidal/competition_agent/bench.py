"""Phase 0 baseline harness.

Plays N games between arbitrary policy classes seated in any arrangement and
reports per-seat win rate (with a 95% Wilson interval), mean net worth, and
mean game length.

Usage
-----
    python3 competition_agent/bench.py --games 50 \
        --policies teacher,teacher,teacher,teacher

    python3 competition_agent/bench.py --games 200 --workers 8 \
        --policies rollout,fixed-a,fixed-b,fixed-c

Determinism
-----------
Game ``k`` uses seed ``--seed + k`` and is created through the repository's
own ``_new_seeded_game`` so results line up with the existing ASU baseline
artifacts. Stochastic policies are seeded from ``(game_seed, seat)``. The seed
of every game is recorded in the JSON output.

Terminal states
---------------
The engine ends a game either by elimination (one solvent player left) or by
hitting its round cap; ``--max-steps`` adds a third, harness-level cap. Only
elimination games have a true winner. Following the convention of the
repository's own evaluator, capped games are reported separately as
"provisional leader" results rather than being silently counted as wins, and
both rates are printed.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

# Pin BLAS/OMP threads before torch is imported anywhere down the chain.
# We parallelise across whole games, so per-process intra-op threading buys
# nothing and costs a lot: torch defaults to 4 intra-op + 10 inter-op threads
# per worker, which put ~150 threads behind 10 workers and drove the load
# average to ~200 on a 10-core box. One thread per worker, N workers.
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ASU_FROZEN_TEACHER.evaluate import _new_seeded_game  # noqa: E402
from ASU_FROZEN_TEACHER.spec import FROZEN_SPEC_HASH  # noqa: E402
from competition_agent.policies import build_policy  # noqa: E402
from competition_agent.proc import ensure_hash_seed, managed_pool  # noqa: E402

NUM_SEATS = 4


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def wilson_interval(successes: int, trials: int, z: float = 1.96):
    """95% Wilson score interval for a binomial proportion."""
    if trials == 0:
        return (0.0, 0.0, 0.0)
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    margin = (
        z
        * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
        / denom
    )
    return (p, max(0.0, centre - margin), min(1.0, centre + margin))


# --------------------------------------------------------------------------
# one game
# --------------------------------------------------------------------------
def play_game(
    seed: int, policy_names: Sequence[str], max_steps: int
) -> Dict[str, Any]:
    """Play a single game and return its record. Pure; safe to run in a pool."""
    game = _new_seeded_game(seed)
    env = game.env

    agents = {
        seat: build_policy(name, seat, rng_seed=seed * NUM_SEATS + seat)
        for seat, name in enumerate(policy_names)
    }

    steps = 0
    started = time.perf_counter()
    while not env.done and steps < max_steps:
        actor = env.whose_turn()
        action = agents[actor].choose_action(env)
        game.step(action)
        steps += 1
    elapsed = time.perf_counter() - started

    active = [p.player_id for p in env.players if not p.bankrupt]
    step_capped = steps >= max_steps and not env.done
    decisive = len(active) == 1
    leader = env.winner()

    return {
        "seed": seed,
        "policies": list(policy_names),
        "steps": steps,
        "rounds": int(getattr(env, "round", -1)),
        "seconds": elapsed,
        "decisive": decisive,
        "step_capped": step_capped,
        "winner": active[0] if decisive else None,
        "leader": leader,
        "net_worth": [float(p.net_worth()) for p in env.players],
        "bankrupt": [bool(p.bankrupt) for p in env.players],
    }


def _worker(args):
    # Env vars alone do not bind torch's own thread pools in a forked child.
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass
    return play_game(*args)


# --------------------------------------------------------------------------
# aggregation / reporting
# --------------------------------------------------------------------------
def summarise(records: List[Dict[str, Any]], policy_names: Sequence[str]):
    n = len(records)
    decisive = [r for r in records if r["decisive"]]
    seats = []
    for seat in range(NUM_SEATS):
        lead_wins = sum(1 for r in records if r["leader"] == seat)
        dec_wins = sum(1 for r in decisive if r["winner"] == seat)
        lp, llo, lhi = wilson_interval(lead_wins, n)
        dp, dlo, dhi = wilson_interval(dec_wins, len(decisive))
        seats.append(
            {
                "seat": seat,
                "policy": policy_names[seat],
                "leader_wins": lead_wins,
                "leader_rate": lp,
                "leader_ci": [llo, lhi],
                "decisive_wins": dec_wins,
                "decisive_rate": dp,
                "decisive_ci": [dlo, dhi],
                "mean_net_worth": (
                    sum(r["net_worth"][seat] for r in records) / n if n else 0.0
                ),
                "bankrupt_rate": (
                    sum(1 for r in records if r["bankrupt"][seat]) / n
                    if n
                    else 0.0
                ),
            }
        )
    return {
        "games": n,
        "decisive_games": len(decisive),
        "capped_games": n - len(decisive),
        "step_capped_games": sum(1 for r in records if r["step_capped"]),
        "mean_steps": sum(r["steps"] for r in records) / n if n else 0.0,
        "mean_rounds": sum(r["rounds"] for r in records) / n if n else 0.0,
        "mean_seconds": sum(r["seconds"] for r in records) / n if n else 0.0,
        "seats": seats,
    }


def print_report(summary: Dict[str, Any], policy_names: Sequence[str]) -> None:
    print()
    print(f"games={summary['games']}  "
          f"decisive={summary['decisive_games']}  "
          f"capped={summary['capped_games']}  "
          f"step_capped={summary['step_capped_games']}")
    print(f"mean length: {summary['mean_steps']:.1f} steps / "
          f"{summary['mean_rounds']:.1f} rounds   "
          f"({summary['mean_seconds']:.2f}s per game)")
    print()
    header = (
        f"{'seat':>4}  {'policy':<12} {'leader%':>8} {'95% CI':>16} "
        f"{'decisive%':>10} {'mean NW':>10} {'bankrupt%':>10}"
    )
    print(header)
    print("-" * len(header))
    for s in summary["seats"]:
        ci = f"[{s['leader_ci'][0]*100:5.1f},{s['leader_ci'][1]*100:5.1f}]"
        print(
            f"{s['seat']:>4}  {s['policy']:<12} "
            f"{s['leader_rate']*100:7.1f}% {ci:>16} "
            f"{s['decisive_rate']*100:9.1f}% "
            f"{s['mean_net_worth']:10.0f} "
            f"{s['bankrupt_rate']*100:9.1f}%"
        )
    print()


def main(argv=None) -> int:
    # The strong field contains `fixed-d`, whose colour-target set
    # iterates in hash order. Pin it before any game is played.
    ensure_hash_seed()
    ap = argparse.ArgumentParser(description="Monopoly policy benchmark")
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0,
                    help="base seed; game k uses seed+k")
    ap.add_argument("--max-steps", type=int, default=3000,
                    help="harness-level step cap per game")
    ap.add_argument("--policies", type=str,
                    default="teacher,teacher,teacher,teacher",
                    help="comma-separated, one per seat (4 seats)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel worker processes")
    ap.add_argument("--output", type=str, default=None,
                    help="write full JSON record here; a .partial.jsonl "
                         "sidecar streams games as they finish")
    ap.add_argument("--rotate", action="store_true",
                    help="play every seed under BOTH seat arrangements "
                         "(the given order and its rotation by one), so seat "
                         "effects cancel in a single command. Seat effects "
                         "are real here: identical policies have differed by "
                         "7 points across arrangements.")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore an existing .partial.jsonl and replay all "
                         "seeds from scratch")
    args = ap.parse_args(argv)

    policy_names = [p.strip() for p in args.policies.split(",")]
    if len(policy_names) != NUM_SEATS:
        ap.error(f"--policies needs exactly {NUM_SEATS} entries, "
                 f"got {len(policy_names)}")

    seeds = [args.seed + k for k in range(args.games)]
    arrangements = [policy_names]
    if args.rotate:
        arrangements.append(policy_names[1:] + policy_names[:1])

    # Resume: a long run must survive interruption. Completed games are
    # streamed to a JSONL sidecar as they finish, and seeds already present
    # there are skipped. See DECISIONS.md D0.8 — a 2h09 reference run was
    # lost because results only materialised at the end.
    resume_path = (
        Path(args.output).with_suffix(".partial.jsonl") if args.output
        else None
    )
    records: List[Dict[str, Any]] = []
    done_seeds: set = set()
    if resume_path and resume_path.exists() and not args.no_resume:
        for line in resume_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("policies") in arrangements:
                records.append(rec)
                done_seeds.add((rec["seed"], tuple(rec["policies"])))
        if done_seeds:
            print(f"resuming: {len(done_seeds)} game(s) already complete "
                  f"in {resume_path.name}")

    jobs = []
    for arr in arrangements:
        for s in seeds:
            if (s, tuple(arr)) not in done_seeds:
                jobs.append((s, arr, args.max_steps))
    todo = jobs

    print(f"frozen_spec_hash: {FROZEN_SPEC_HASH[:16]}...")
    print(f"seats: {policy_names}   seeds: {seeds[0]}..{seeds[-1]}   "
          f"workers: {args.workers}   to play: {len(todo)}"
          + (f"   ({len(arrangements)} seat arrangements)" if args.rotate else ""))

    wall = time.perf_counter()
    sink = resume_path.open("a") if resume_path else None
    try:
        if args.workers > 1 and jobs:
            with managed_pool(args.workers) as pool:
                stream = pool.imap_unordered(_worker, jobs)
                for i, rec in enumerate(stream, 1):
                    records.append(rec)
                    if sink:
                        sink.write(json.dumps(rec) + "\n")
                        sink.flush()
                    if i % 10 == 0 or i == len(jobs):
                        print(f"  {i}/{len(jobs)} games "
                              f"({time.perf_counter() - wall:.0f}s)", flush=True)
        else:
            for i, job in enumerate(jobs, 1):
                rec = _worker(job)
                records.append(rec)
                if sink:
                    sink.write(json.dumps(rec) + "\n")
                    sink.flush()
    finally:
        if sink:
            sink.close()
    wall = time.perf_counter() - wall
    records.sort(key=lambda r: r["seed"])

    summary = summarise(records, policy_names)
    print_report(summary, policy_names)
    print(f"wall clock: {wall:.1f}s")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {
                "frozen_spec_hash": FROZEN_SPEC_HASH,
                "args": vars(args),
                "summary": summary,
                "games": records,
            },
            indent=2,
        ) + "\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
