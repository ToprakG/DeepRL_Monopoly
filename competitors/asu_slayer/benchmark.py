"""Paired benchmark: the challenger and ASU face identical lineups and seeds.

Every lineup is run twice, once with the challenger as the focus seat and once
with ``asu-value-v1`` in that role, on the same seeds. Reporting only the
challenger's win rate would be meaningless without that control, because the
lineups differ wildly in difficulty.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ASU_FROZEN_TEACHER.evaluate import evaluate_lineup, wilson_interval


LINEUPS: dict[str, tuple[str, str, str]] = {
    "asu-value-sweep": ("asu-value-v1", "asu-value-v1", "asu-value-v1"),
    "asu-mixed": ("asu-value-v1", "asu-value-v1", "fixed-a"),
    "asu-rollout": ("asu-rollout-v1", "asu-value-v1", "asu-value-v1"),
    "fixed": ("fixed-a", "fixed-b", "fixed-c"),
}
DEFAULT_SUITE = ("fixed", "asu-mixed", "asu-value-sweep")
CONTROL = "asu-value-v1"


def _summary(result: dict[str, Any], policy_id: str) -> dict[str, Any]:
    record = result["win_rates"][policy_id]
    return {
        "wins": record["wins"],
        "games": record["games"],
        "win_rate": record["win_rate"],
        "wilson_95": record["wilson_95"],
        "mean_net_worth": result["final_net_worth"][policy_id]["mean"],
        "truncations": result["truncations"],
        "elapsed_seconds": result["elapsed_seconds"],
    }


def run_suite(
    focus: str,
    lineups: tuple[str, ...],
    seeds: tuple[int, ...],
    max_decisions: int,
    control: bool = True,
) -> dict[str, Any]:
    """Run each lineup for the focus policy and, optionally, for the control."""

    started = time.perf_counter()
    report: dict[str, Any] = {"focus": focus, "seeds": list(seeds), "lineups": {}}
    for name in lineups:
        opponents = LINEUPS[name]
        entry: dict[str, Any] = {"opponents": list(opponents)}
        result = evaluate_lineup(focus, opponents, seeds, max_decisions)
        entry["focus"] = _summary(result, focus)
        if control and focus != CONTROL:
            baseline = evaluate_lineup(CONTROL, opponents, seeds, max_decisions)
            entry["control"] = _summary(baseline, CONTROL)
            focus_rate = entry["focus"]["win_rate"]
            control_rate = entry["control"]["win_rate"]
            if focus_rate is not None and control_rate is not None:
                entry["win_rate_delta"] = focus_rate - control_rate
        report["lineups"][name] = entry
        _print_row(name, entry)

    wins = sum(
        entry["focus"]["wins"] for entry in report["lineups"].values()
    )
    games = sum(entry["focus"]["games"] for entry in report["lineups"].values())
    report["overall"] = {
        "wins": wins,
        "games": games,
        "win_rate": (wins / games) if games else None,
        "wilson_95": list(wilson_interval(wins, games)),
    }
    report["elapsed_seconds"] = time.perf_counter() - started
    return report


def _print_row(name: str, entry: dict[str, Any]) -> None:
    focus = entry["focus"]
    rate = focus["win_rate"]
    text = "n/a" if rate is None else f"{100 * rate:5.1f}%"
    low, high = focus["wilson_95"]
    line = (
        f"{name:<18} focus {focus['wins']:>3}/{focus['games']:<3} {text} "
        f"[{100 * low:4.1f}, {100 * high:5.1f}]  nw={focus['mean_net_worth']:>9,.0f}"
    )
    control = entry.get("control")
    if control and control["win_rate"] is not None:
        line += f"   asu-control {100 * control['win_rate']:5.1f}%"
    print(line, flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--focus", default="slayer-v1")
    parser.add_argument(
        "--lineups", nargs="+", choices=sorted(LINEUPS), default=list(DEFAULT_SUITE)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2, 3))
    parser.add_argument("--max-decisions", type=int, default=20_000)
    parser.add_argument("--no-control", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_suite(
        args.focus,
        tuple(args.lineups),
        tuple(args.seeds),
        args.max_decisions,
        control=not args.no_control,
    )
    overall = report["overall"]
    rate = overall["win_rate"]
    print(
        f"\noverall {overall['wins']}/{overall['games']} = "
        f"{'n/a' if rate is None else f'{100 * rate:.1f}%'} "
        f"(chance is 25.0%)",
        flush=True,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CONTROL", "LINEUPS", "main", "run_suite"]
