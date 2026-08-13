"""Downsample ACCEPT_TRADE-heavy hybrid labels for behavioral cloning."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from monopoly_game_engine.actions import ActionType, action_to_description

ARRAY_KEYS = (
    "states",
    "legal_masks",
    "actors",
    "selected_actions",
    "values",
    "outcomes",
    "policy_actions",
    "policy_weights",
)

ACCEPT = int(ActionType.ACCEPT_TRADE)
DECLINE = int(ActionType.DECLINE_TRADE)
BUY = int(ActionType.BUY_PROPERTY)


def bucket_for_action(action: int) -> str:
    action = int(action)
    if action == ACCEPT:
        return "accept"
    if action == DECLINE:
        return "decline"
    if action == BUY:
        return "buy"
    text = action_to_description(action)
    if text.startswith("auction"):
        return "auction"
    if text.startswith("improve_"):
        return "build"
    return "other"


def bucket_labels(selected: np.ndarray) -> np.ndarray:
    return np.asarray([bucket_for_action(int(a)) for a in selected], dtype=object)


def accept_keep_rate(*, accept_frac: float, target_accept_frac: float) -> float:
    """Fraction of ACCEPT rows to keep so final ACCEPT share ≈ target."""

    if accept_frac <= 0:
        return 1.0
    if accept_frac <= target_accept_frac:
        return 1.0
    other = 1.0 - accept_frac
    # target = f*A / (other + f*A)  =>  f = target*other / (A*(1-target))
    return float(
        np.clip(
            target_accept_frac * other / (accept_frac * (1.0 - target_accept_frac)),
            0.0,
            1.0,
        )
    )


def resample_indices(
    selected: np.ndarray,
    *,
    target_accept_frac: float = 0.20,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = np.asarray(selected)
    accept_mask = selected == ACCEPT
    n_accept = int(accept_mask.sum())
    n = len(selected)
    accept_frac = n_accept / n if n else 0.0
    keep_rate = accept_keep_rate(
        accept_frac=accept_frac, target_accept_frac=target_accept_frac
    )
    accept_idx = np.flatnonzero(accept_mask)
    other_idx = np.flatnonzero(~accept_mask)
    n_keep_accept = int(round(n_accept * keep_rate))
    if n_keep_accept < n_accept:
        keep_accept = rng.choice(accept_idx, size=n_keep_accept, replace=False)
    else:
        keep_accept = accept_idx
    kept = np.concatenate([other_idx, keep_accept])
    rng.shuffle(kept)
    return kept.astype(np.int64)


def sampling_probabilities(
    selected: np.ndarray,
    *,
    target_mass: dict[str, float] | None = None,
) -> np.ndarray:
    """Per-row sampling probs so batches track target family mass."""

    if target_mass is None:
        target_mass = {
            "accept": 0.15,
            "decline": 0.05,
            "buy": 0.30,
            "build": 0.15,
            "auction": 0.15,
            "other": 0.20,
        }
    buckets = bucket_labels(selected)
    counts = Counter(str(b) for b in buckets)
    # Renormalize targets over buckets that exist.
    present = {k: v for k, v in target_mass.items() if counts.get(k, 0) > 0}
    if not present:
        return np.full(len(selected), 1.0 / max(len(selected), 1), dtype=np.float64)
    total_t = sum(present.values())
    present = {k: v / total_t for k, v in present.items()}
    weights = np.empty(len(selected), dtype=np.float64)
    for index, bucket in enumerate(buckets):
        key = str(bucket)
        weights[index] = present.get(key, 0.0) / counts[key]
    weights /= weights.sum()
    return weights


def resample_arrays(
    examples: dict[str, np.ndarray],
    *,
    target_accept_frac: float = 0.20,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    selected = examples["selected_actions"]
    before = Counter(bucket_for_action(int(a)) for a in selected)
    indices = resample_indices(
        selected, target_accept_frac=target_accept_frac, seed=seed
    )
    out = {key: value[indices] for key, value in examples.items()}
    after = Counter(bucket_for_action(int(a)) for a in out["selected_actions"])
    n_before = len(selected)
    n_after = len(indices)
    report = {
        "n_before": n_before,
        "n_after": n_after,
        "target_accept_frac": target_accept_frac,
        "seed": seed,
        "accept_keep_rate": accept_keep_rate(
            accept_frac=before.get("accept", 0) / n_before if n_before else 0.0,
            target_accept_frac=target_accept_frac,
        ),
        "buckets_before": {k: before[k] / n_before for k in sorted(before)},
        "buckets_after": {k: after[k] / n_after for k in sorted(after)},
        "counts_before": dict(before),
        "counts_after": dict(after),
    }
    return out, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts_scratch/oracle_hybrid_merged/labels_hybrid_merged.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts_scratch/oracle_hybrid_merged/labels_hybrid_resampled.npz"
        ),
    )
    parser.add_argument("--target-accept-frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "artifacts_scratch/oracle_hybrid_merged/resample_report.json"
        ),
    )
    args = parser.parse_args(argv)
    with np.load(args.input, allow_pickle=False) as payload:
        missing = [k for k in ARRAY_KEYS if k not in payload.files]
        if missing:
            raise SystemExit(f"{args.input} missing {missing}")
        examples = {k: np.asarray(payload[k]).copy() for k in ARRAY_KEYS}
    out, report = resample_arrays(
        examples,
        target_accept_frac=args.target_accept_frac,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **out)
    report["input"] = str(args.input)
    report["output"] = str(args.output)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {args.output} ({report['n_before']} -> {report['n_after']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
