"""Merge hybrid oracle label shards and emit a sanity report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from monopoly_game_engine.actions import OFFSETS, ActionType, action_to_description
from monopoly_game_engine.constants import RULESET_VERSION

ARRAY_KEYS = (
    "states",
    "legal_masks",
    "actors",
    "selected_actions",
    "values",
    "outcomes",
)
ACTION_DIM = 2_958
STATE_DIM = 300
NUM_PLAYERS = 4


def _discover_npz(shard_dir: Path) -> list[Path]:
    finals = sorted(p for p in shard_dir.glob("labels_seed*.npz") if p.parent == shard_dir)
    covered = {p.stem for p in finals}
    parts: list[Path] = []
    for ckpt in sorted(shard_dir.glob("labels_seed*.ckpt")):
        stem = ckpt.name.removesuffix(".ckpt")
        if stem in covered:
            continue
        parts.extend(sorted(ckpt.glob("part_*.npz")))
    return finals + parts


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        missing = [key for key in ARRAY_KEYS if key not in payload.files]
        if missing:
            raise ValueError(f"{path} missing fields {missing}")
        return {key: np.asarray(payload[key]).copy() for key in ARRAY_KEYS}


def _family(action: int) -> str:
    text = action_to_description(int(action))
    if "(" in text:
        return text.split("(", 1)[0]
    if text.startswith("auction_"):
        return "auction"
    return text


def _soft_targets_from_jsonl(
    jsonl_path: Path,
    *,
    n_labels: int,
    selected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build sparse visit targets; fall back to one-hot when jsonl is missing/short."""

    max_width = 1
    rows: list[list[tuple[int, float]]] = []
    if jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            dist = row.get("visit_distribution") or {}
            pairs = sorted(
                ((int(action), float(weight)) for action, weight in dist.items() if float(weight) > 0),
                key=lambda item: (-item[1], item[0]),
            )
            rows.append(pairs)
            max_width = max(max_width, len(pairs) or 1)
    # Pad / truncate to selected length.
    if len(rows) < n_labels:
        rows.extend([[] for _ in range(n_labels - len(rows))])
    elif len(rows) > n_labels:
        rows = rows[:n_labels]

    max_width = min(max(max_width, 1), 64)
    actions = np.full((n_labels, max_width), -1, dtype=np.int64)
    weights = np.zeros((n_labels, max_width), dtype=np.float32)
    soft_rows = 0
    hard_mismatch = 0
    for index, pairs in enumerate(rows):
        if not pairs:
            action = int(selected[index])
            actions[index, 0] = action
            weights[index, 0] = 1.0
            continue
        soft_rows += 1
        total = sum(weight for _, weight in pairs) or 1.0
        top = pairs[:max_width]
        for slot, (action, weight) in enumerate(top):
            actions[index, slot] = action
            weights[index, slot] = weight / total
        mode = top[0][0]
        if mode != int(selected[index]):
            hard_mismatch += 1
    meta = {
        "soft_rows": soft_rows,
        "one_hot_rows": n_labels - soft_rows,
        "mode_vs_selected_mismatch": hard_mismatch,
        "max_width": max_width,
        "jsonl": str(jsonl_path) if jsonl_path.exists() else None,
    }
    return actions, weights, meta


def merge_shards(shard_dir: Path) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    paths = _discover_npz(shard_dir)
    if not paths:
        raise FileNotFoundError(f"No label npz shards under {shard_dir}")
    groups = []
    inventory = []
    for path in paths:
        data = _load_npz(path)
        n = int(data["states"].shape[0])
        inventory.append({"path": str(path.relative_to(shard_dir)), "n_labels": n})
        # Attach soft targets from sibling jsonl when present.
        if path.name.startswith("part_"):
            jsonl = path.with_suffix(".jsonl")
        else:
            jsonl = path.with_suffix(".jsonl")
        actions, weights, soft_meta = _soft_targets_from_jsonl(
            jsonl, n_labels=n, selected=data["selected_actions"]
        )
        data["policy_actions"] = actions
        data["policy_weights"] = weights
        inventory[-1]["soft"] = soft_meta
        groups.append(data)

    # Pad sparse policy width to a common max.
    width = max(int(group["policy_actions"].shape[1]) for group in groups)
    for group in groups:
        cur = int(group["policy_actions"].shape[1])
        if cur == width:
            continue
        n = int(group["policy_actions"].shape[0])
        padded_a = np.full((n, width), -1, dtype=np.int64)
        padded_w = np.zeros((n, width), dtype=np.float32)
        padded_a[:, :cur] = group["policy_actions"]
        padded_w[:, :cur] = group["policy_weights"]
        group["policy_actions"] = padded_a
        group["policy_weights"] = padded_w

    merged = {
        key: np.concatenate([group[key] for group in groups], axis=0)
        for key in (*ARRAY_KEYS, "policy_actions", "policy_weights")
    }
    merged["actors"] = merged["actors"].astype(np.int64, copy=False)
    merged["selected_actions"] = merged["selected_actions"].astype(np.int64, copy=False)
    return merged, inventory


def sanity_check(examples: dict[str, np.ndarray], inventory: list[dict[str, Any]]) -> dict[str, Any]:
    n = int(examples["states"].shape[0])
    expected = {
        "states": (n, STATE_DIM),
        "legal_masks": (n, ACTION_DIM),
        "actors": (n,),
        "selected_actions": (n,),
        "values": (n, NUM_PLAYERS),
        "outcomes": (n, NUM_PLAYERS),
    }
    shapes = {key: tuple(examples[key].shape) for key in expected}
    if shapes != expected:
        raise ValueError(f"Bad merged shapes: {shapes}")

    actions = examples["selected_actions"]
    masks = examples["legal_masks"]
    illegal = int((~masks[np.arange(n), actions]).sum())
    families = Counter(_family(int(action)) for action in actions)
    family_frac = {
        name: families[name] / n for name in sorted(families, key=families.get, reverse=True)
    }
    end_turn_frac = family_frac.get("END_TURN", 0.0)
    buy_frac = family_frac.get("BUY_PROPERTY", 0.0)
    accept_frac = family_frac.get("ACCEPT_TRADE", 0.0)
    decline_frac = family_frac.get("DECLINE_TRADE", 0.0)
    build_frac = family_frac.get("improve_house", 0.0) + family_frac.get("improve_hotel", 0.0)
    auction_frac = sum(v for k, v in family_frac.items() if k.startswith("auction"))

    values = examples["values"].astype(np.float64, copy=False)
    outcomes = examples["outcomes"].astype(np.float64, copy=False)
    value_row_sums = values.sum(axis=1)
    outcome_row_sums = outcomes.sum(axis=1)

    soft_rows = sum(int(item["soft"]["soft_rows"]) for item in inventory)
    one_hot_rows = sum(int(item["soft"]["one_hot_rows"]) for item in inventory)
    mode_mismatch = sum(int(item["soft"]["mode_vs_selected_mismatch"]) for item in inventory)

    report = {
        "n_labels": n,
        "n_shards": len(inventory),
        "inventory": inventory,
        "schema_ok": True,
        "illegal_selected_actions": illegal,
        "finite_states": bool(np.isfinite(examples["states"]).all()),
        "finite_values": bool(np.isfinite(values).all()),
        "finite_outcomes": bool(np.isfinite(outcomes).all()),
        "value_row_sum": {
            "mean": float(value_row_sums.mean()),
            "min": float(value_row_sums.min()),
            "max": float(value_row_sums.max()),
        },
        "outcome_row_sum": {
            "mean": float(outcome_row_sums.mean()),
            "min": float(outcome_row_sums.min()),
            "max": float(outcome_row_sums.max()),
        },
        "family_fraction": family_frac,
        "checkpoint_mix": {
            "BUY_PROPERTY": buy_frac,
            "improve_*": build_frac,
            "ACCEPT_TRADE": accept_frac,
            "DECLINE_TRADE": decline_frac,
            "auction_*": auction_frac,
            "END_TURN": end_turn_frac,
        },
        "soft_policy": {
            "soft_rows": soft_rows,
            "one_hot_rows": one_hot_rows,
            "mode_vs_selected_mismatch": mode_mismatch,
            "soft_coverage": soft_rows / n if n else 0.0,
        },
        "gates": {
            "schema_matches_expert_core": illegal == 0
            and bool(np.isfinite(examples["states"]).all())
            and bool(np.isfinite(values).all()),
            "not_pathological_end_turn": end_turn_frac < 0.20,
            "has_buy_build_trade_auction_mass": (buy_frac + build_frac + accept_frac + decline_frac + auction_frac)
            >= 0.50,
            # New coverage labeling always emits soft visits; reject friend one-hot dumps.
            "soft_coverage_ok": (soft_rows / n if n else 0.0) >= 0.90,
            "decline_present": decline_frac >= 0.01 or accept_frac < 0.10,
        },
        "warnings": {
            "accept_trade_heavy": accept_frac >= 0.15,
            "accept_among_trade_decisions": (
                accept_frac / (accept_frac + decline_frac)
                if (accept_frac + decline_frac) > 0
                else None
            ),
            "one_hot_rows": one_hot_rows,
            "note": (
                "Coverage labeling records all incoming accept/decline once per round "
                "plus a routine subsample. Soft visit distributions are required "
                "(keep paired .jsonl / refuse one-hot-only shards)."
            ),
        },
        "ruleset": RULESET_VERSION,
        "binary_action_ids": {
            name: int(ActionType[name])
            for name in ("END_TURN", "BUY_PROPERTY", "ACCEPT_TRADE", "DECLINE_TRADE")
        },
        "offsets": {name: int(start) for name, start in OFFSETS.items()},
    }
    report["gates"]["all_pass"] = all(report["gates"].values())
    return report


def save_merged(path: Path, examples: dict[str, np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **examples, ruleset=np.asarray(RULESET_VERSION))
    temporary.replace(path)
    return path


def save_asu_compat(path: Path, examples: dict[str, np.ndarray]) -> Path:
    """ASU loader shape (hard CE + winner value). Uses teachers=0 sentinel."""
    from monopoly_bench.training import save_asu_examples

    payload = {
        "states": examples["states"].astype(np.float32, copy=False),
        "legal_masks": examples["legal_masks"].astype(np.bool_, copy=False),
        "selected_actions": examples["selected_actions"].astype(np.int64, copy=False),
        "actors": examples["actors"].astype(np.int64, copy=False),
        "outcomes": examples["outcomes"].astype(np.float32, copy=False),
        "teachers": np.zeros(len(examples["states"]), dtype=np.uint8),
    }
    return save_asu_examples(path, payload)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=Path("artifacts_scratch/colab_shards"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts_scratch/oracle_hybrid_merged/labels_hybrid_merged.npz"),
    )
    parser.add_argument(
        "--asu-compat-output",
        type=Path,
        default=Path("artifacts_scratch/oracle_hybrid_merged/labels_hybrid_asu_compat.npz"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts_scratch/oracle_hybrid_merged/sanity_report.json"),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    examples, inventory = merge_shards(args.shard_dir)
    report = sanity_check(examples, inventory)
    out = save_merged(args.output, examples)
    asu_out = save_asu_compat(args.asu_compat_output, examples)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report["outputs"] = {"merged": str(out), "asu_compat": str(asu_out)}
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["gates"]["all_pass"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
