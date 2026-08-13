#!/usr/bin/env python3
"""Seat-balanced hybrid BC clone (search) vs FPAgent A/B/C, parallelized."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from monopoly_bench.adapters import ASUAdapter, FixedAdapter, SearchAdapter
from monopoly_bench.arena import balanced_single_seats, play_game, summarize
from monopoly_bench.config import SearchConfig
from monopoly_bench.model import MonopolyZeroNet
from monopoly_game_engine.agents_fixed import FPAgentA, FPAgentB, FPAgentC

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / "artifacts_scratch/oracle_hybrid_bc/hybrid_clone_0000.pt"
FIXED = (FPAgentA, FPAgentB, FPAgentC)


def _game_job(job: dict) -> tuple[int, object]:
    model = MonopolyZeroNet.load_inference(job["checkpoint"], device="cpu")
    search = replace(SearchConfig(), simulations=int(job["sims"]))
    champion = SearchAdapter(model, search, self_play=False)
    if job.get("vs") == "asu":
        opponents = [ASUAdapter(), ASUAdapter(), ASUAdapter()]
    else:
        opponents = [FixedAdapter(cls) for cls in FIXED]
    champion_seat = int(job["champion_seat"])
    policies = {champion_seat: champion}
    opp_i = 0
    for seat in range(4):
        if seat == champion_seat:
            continue
        policies[seat] = opponents[opp_i]
        opp_i += 1
    result = play_game(
        game_id=int(job["game_id"]),
        seed=int(job["seed"]),
        policies=policies,
        max_rounds=int(job["max_rounds"]),
        record_seats=set(),
    )
    return int(job["game_id"]), result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--seed", type=int, default=5_100_000)
    parser.add_argument("--sims", type=int, default=32)
    parser.add_argument("--vs", choices=("fixed", "asu"), default="fixed")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=200)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts_scratch/hybrid_clone_vs_fixed_8.json",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Write JSON to --output only; don't dump the full report to stdout.",
    )
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {args.checkpoint}")

    seats = balanced_single_seats(args.games)
    jobs = [
        {
            "game_id": index,
            "seed": args.seed + index,
            "champion_seat": next(iter(target)),
            "checkpoint": str(args.checkpoint.resolve()),
            "sims": args.sims,
            "vs": args.vs,
            "max_rounds": args.max_rounds,
        }
        for index, target in enumerate(seats)
    ]

    started = time.perf_counter()
    results: list = [None] * len(jobs)
    workers = min(args.workers, args.games)
    print(
        f"hybrid_clone vs {args.vs} | games={args.games} sims={args.sims} "
        f"workers={workers} ckpt={args.checkpoint}",
        flush=True,
    )
    if workers == 1:
        for job in jobs:
            gid, result = _game_job(job)
            results[gid] = result
            print(
                f"h2h progress {gid + 1}/{args.games} winner={result.winner} "
                f"seat={job['champion_seat']} completed={result.completed}",
                flush=True,
            )
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            done = 0
            for gid, result in pool.imap_unordered(_game_job, jobs, chunksize=1):
                results[gid] = result
                done += 1
                seat = jobs[gid]["champion_seat"]
                print(
                    f"h2h progress {done}/{args.games} game={gid} winner={result.winner} "
                    f"clone_seat={seat} completed={result.completed} crashes={result.crashes}",
                    flush=True,
                )

    summary = summarize(results, seats)
    clone_wins = sum(
        1
        for result, target in zip(results, seats)
        if result.completed and result.winner in target
    )
    latencies = [dt for result in results for dt in result.search_latencies]
    report = {
        "checkpoint": str(args.checkpoint),
        "vs": args.vs,
        "lineup": (
            ["hybrid_clone", "asu-value-v1", "asu-value-v1", "asu-value-v1"]
            if args.vs == "asu"
            else ["hybrid_clone", "fixed-a", "fixed-b", "fixed-c"]
        ),
        "games": args.games,
        "seed": args.seed,
        "sims": args.sims,
        "workers": workers,
        "wall_seconds": time.perf_counter() - started,
        "summary": summary.as_dict(),
        "clone_wins": clone_wins,
        "n_searches": len(latencies),
        "latency_mean_s": float(np.mean(latencies)) if latencies else None,
        "latency_p50_s": float(np.percentile(latencies, 50)) if latencies else None,
        "latency_p95_s": float(np.percentile(latencies, 95)) if latencies else None,
        "per_game": [
            {
                "game_id": r.game_id,
                "seed": r.seed,
                "clone_seat": next(iter(seats[r.game_id])),
                "winner": r.winner,
                "completed": r.completed,
                "decisions": r.decisions,
                "crashes": r.crashes,
                "illegal_actions": r.illegal_actions,
                "error": r.error,
            }
            for r in results
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(
        f"clone vs {args.vs} sims={args.sims} WR={summary.win_rate:.3f} "
        f"({clone_wins}/{args.games}) wilson_lo={summary.wilson_lower:.3f} "
        f"p95={report['latency_p95_s']} wall={report['wall_seconds']:.1f}s "
        f"-> {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
