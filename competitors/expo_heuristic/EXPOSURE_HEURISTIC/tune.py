"""
Cross-entropy tuning of EXPO's weight vector.

EXPO is a *static* heuristic: nothing inside it changes while a game is
played, and nothing carries between games. This module is where the
improvement actually happens -- offline, by searching the nine weights in
``agent.TUNABLE`` for the vector with the best win rate against a fixed
opponent panel.

The cross-entropy method is the right fit here because the objective is a
noisy, non-differentiable win rate: sample a population of weight vectors
from a Gaussian, evaluate each on a *common* set of seeds (so the
comparison is paired and luck largely cancels), keep the top fraction, and
refit the Gaussian to those elites.

Guard against the obvious trap: a vector that wins on the seeds it was
selected on has not necessarily improved. ``--holdout`` re-scores the
final vector on seeds never used during the search.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import EXPOSURE_HEURISTIC  # noqa: F401
from EXPOSURE_HEURISTIC.collect import collect, summarise

#: name -> (initial value, sigma, low, high)
SEARCH_SPACE = {
    "RENT_HORIZON":           (7.00, 2.50, 1.0, 20.0),
    "DENIAL_WEIGHT":          (0.45, 0.25, 0.0, 1.5),
    "TRADE_RIVAL_WEIGHT":     (0.75, 0.30, 0.0, 2.0),
    "ENDGAME_ROUNDS":         (14.0, 8.00, 0.0, 60.0),
    "RESERVE_RENT_FRACTION":  (0.55, 0.25, 0.0, 2.0),
    "RESERVE_FLOOR":          (110.0, 60.0, 0.0, 500.0),
    "AUCTION_VALUE_FRACTION": (0.50, 0.20, 0.05, 1.5),
    "AUCTION_LIST_CAP":       (1.15, 0.40, 0.3, 3.0),
    "MONOPOLY_AUCTION_CAP":   (2.00, 0.70, 0.5, 4.0),
    # The two solvency terms. SOLVENCY_HORIZON is the hypothesised real
    # lever on the bankruptcy death-spiral, so it gets a wide range.
    "SOLVENCY_HORIZON":       (3.00, 3.00, 0.0, 20.0),
    "RUIN_SAFETY":            (1.00, 0.80, 0.0, 4.0),
}
NAMES = list(SEARCH_SPACE)
LOW = np.array([SEARCH_SPACE[n][2] for n in NAMES])
HIGH = np.array([SEARCH_SPACE[n][3] for n in NAMES])


def as_weights(vector) -> dict:
    weights = {}
    for name, value in zip(NAMES, vector):
        weights[name] = int(round(value)) if name == "ENDGAME_ROUNDS" else float(value)
    return weights


def score(vector, seeds, opponents, workers, max_rounds) -> float:
    games, _ = collect(
        seeds, opponents=opponents, weights=as_weights(vector),
        workers=workers, max_rounds=max_rounds,
    )
    return summarise(games)["win_rate"]


def tune(generations, population, elite_frac, blocks, opponents,
         workers, max_rounds, seed=0):
    rng = np.random.default_rng(seed)
    mean = np.array([SEARCH_SPACE[n][0] for n in NAMES], dtype=float)
    sigma = np.array([SEARCH_SPACE[n][1] for n in NAMES], dtype=float)
    n_elite = max(2, int(population * elite_frac))
    history = []

    for generation in range(generations):
        # Fresh seeds each generation prevent locking onto one lucky block,
        # but every candidate in a generation shares them (paired comparison).
        seeds = list(range(generation * blocks, (generation + 1) * blocks))
        candidates = np.clip(
            rng.normal(mean, sigma, size=(population, len(NAMES))), LOW, HIGH
        )
        candidates[0] = mean  # always carry the incumbent
        scores = np.array([
            score(c, seeds, opponents, workers, max_rounds) for c in candidates
        ])
        elite = candidates[np.argsort(-scores)[:n_elite]]
        mean = elite.mean(axis=0)
        sigma = elite.std(axis=0) + 0.05 * (HIGH - LOW) / 10.0
        history.append({
            "generation": generation,
            "best": float(scores.max()),
            "mean": float(scores.mean()),
            "incumbent": float(scores[0]),
            "weights": as_weights(mean),
        })
        print(json.dumps(history[-1]), flush=True)

    return as_weights(mean), history


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CEM tuner for EXPO weights")
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--elite-frac", type=float, default=0.3)
    parser.add_argument("--blocks", type=int, default=8,
                        help="seat-balanced blocks per candidate (x4 games)")
    parser.add_argument("--opponents", nargs=3,
                        default=["fixed-b", "fixed-d", "fixed-e"])
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--max-rounds", type=int, default=200)
    parser.add_argument("--holdout", type=int, default=40,
                        help="blocks of unseen seeds for the final check")
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/expo_tuned.json"))
    args = parser.parse_args(argv)

    best, history = tune(
        args.generations, args.population, args.elite_frac, args.blocks,
        args.opponents, args.workers, args.max_rounds,
    )

    # Score baseline and tuned vector on seeds the search never touched.
    holdout = list(range(100_000, 100_000 + args.holdout))
    baseline_vector = [SEARCH_SPACE[n][0] for n in NAMES]
    baseline = score(baseline_vector, holdout, args.opponents,
                     args.workers, args.max_rounds)
    tuned = score([best[n] for n in NAMES], holdout, args.opponents,
                  args.workers, args.max_rounds)

    result = {
        "weights": best,
        "holdout_blocks": args.holdout,
        "holdout_games": args.holdout * 4,
        "baseline_win_rate": round(baseline, 4),
        "tuned_win_rate": round(tuned, 4),
        "improved": bool(tuned > baseline),
        "opponents": args.opponents,
        "history": history,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
