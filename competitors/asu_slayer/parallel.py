"""Resumable parallel benchmark runner for the challenger policies.

Designed for a hosted 4-core session: work is a flat list of independent games,
every finished game is appended to a JSONL ledger, and a restart replays the
ledger and skips what is already done. A session that is cut off at the time
limit therefore loses at most one game per worker.

Two stages run in order.

``sweep``    every candidate config plays the tuning seeds.
``validate`` the winning config replays a held-out seed block, alongside an
             ``asu-value-v1`` control that occupies the same focus seat on the
             same seeds.

The stages use disjoint seed ranges on purpose. Reporting a win rate on the
seeds a config was selected on measures the selection, not the policy.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable

from ASU_FROZEN_TEACHER.evaluate import (
    AgentFactory,
    _new_seeded_game,
    parse_agent_spec,
    wilson_interval,
)
from monopoly_game_engine.constants import NUM_PLAYERS


CONTROL = "asu-value-v1"
DEFAULT_MAX_DECISIONS = 20_000

LINEUPS: dict[str, tuple[str, ...]] = {
    "asu-value-sweep": (CONTROL, CONTROL, CONTROL),
    "asu-mixed": (CONTROL, CONTROL, "fixed-a"),
    "fixed": ("fixed-a", "fixed-b", "fixed-c"),
}

# Candidate reserves, swept over survival risk rather than over a cash floor.
# A 128-game measurement showed the policy winning every game it survived
# (15 of 15) while going bankrupt in the other 113, so the only knob that
# matters is how much ruin risk it carries per turn and over how long a
# horizon. A flat cash floor was tried first and is not the right shape.
CONFIG_GRID: dict[str, dict[str, float]] = {
    "baseline": {},
    "balanced": {"target_survival": 0.50},
    "very-safe": {"target_survival": 0.90},
    "short-horizon": {"expected_game_length": 30.0},
    "long-horizon": {"expected_game_length": 60.0},
    "half-threat": {"threat_multiple": 0.5},
}


def task_key(task: dict[str, Any]) -> str:
    return "|".join(
        str(task[field])
        for field in ("stage", "focus", "config", "lineup", "seed", "seat")
    )


def _build_agent(spec: str, seat: int, overrides: dict, factory: AgentFactory):
    if spec == "slayer-v1":
        from ASU_SLAYER.policy import DEFAULT_CONFIG, SlayerV1

        return SlayerV1(seat, DEFAULT_CONFIG.evolve(**overrides))
    if spec == "slayer-rollout-v1":
        from ASU_SLAYER.policy import DEFAULT_CONFIG
        from ASU_SLAYER.search import SlayerRolloutV1

        return SlayerRolloutV1(seat, DEFAULT_CONFIG.evolve(**overrides))
    return factory.build(parse_agent_spec(spec), seat)


def play(task: dict[str, Any]) -> dict[str, Any]:
    """Run one seat-balanced game. Must stay importable for the worker pool."""

    started = time.perf_counter()
    seat = task["seat"]
    seats: list[str] = list(LINEUPS[task["lineup"]])
    seats.insert(seat, task["focus"])

    game = _new_seeded_game(task["seed"])
    factory = AgentFactory()
    overrides = CONFIG_GRID.get(task["config"], {})
    agents = [_build_agent(spec, index, overrides, factory) for index, spec in enumerate(seats)]

    decisions = 0
    limit = task.get("max_decisions", DEFAULT_MAX_DECISIONS)
    while not game.env.done and decisions < limit:
        actor = game.env.whose_turn()
        allowed = game.env.get_allowed_actions(actor)
        action = agents[actor].choose_action(game.env)
        if action not in allowed:
            raise RuntimeError(
                f"{seats[actor]} returned illegal action {action} for seat {actor}"
            )
        game.step(action)
        decisions += 1

    truncated = not game.env.done
    return {
        **task,
        "won": bool(not truncated and game.env.winner() == seat),
        "truncated": truncated,
        "rounds": game.env.round,
        "decisions": decisions,
        "net_worth": float(game.env.players[seat].net_worth()),
        "final_net_worth": [float(player.net_worth()) for player in game.env.players],
        "elapsed_seconds": time.perf_counter() - started,
    }


def build_tasks(
    stage: str,
    focus: str,
    configs: Iterable[str],
    lineups: Iterable[str],
    seeds: Iterable[int],
    max_decisions: int,
) -> list[dict[str, Any]]:
    return [
        {
            "stage": stage,
            "focus": focus,
            "config": config,
            "lineup": lineup,
            "seed": int(seed),
            "seat": seat,
            "max_decisions": max_decisions,
        }
        for config in configs
        for lineup in lineups
        for seed in seeds
        for seat in range(NUM_PLAYERS)
    ]


def load_ledger(path: Path) -> tuple[list[dict], set[str]]:
    """Replay a partial run so an interrupted session resumes where it stopped."""

    if not path.exists():
        return [], set()
    records: list[dict] = []
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn final line from a killed session
        records.append(record)
        done.add(task_key(record))
    return records, done


def run_tasks(
    tasks: list[dict[str, Any]], ledger: Path, workers: int
) -> list[dict[str, Any]]:
    records, done = load_ledger(ledger)
    wanted = {task_key(task) for task in tasks}
    pending = [task for task in tasks if task_key(task) not in done]
    mine = [record for record in records if task_key(record) in wanted]
    if not pending:
        print(f"  all {len(tasks)} games already in the ledger", flush=True)
        return mine

    print(
        f"  {len(pending)} games to play ({len(tasks) - len(pending)} resumed), "
        f"{workers} workers",
        flush=True,
    )
    started = time.perf_counter()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle, Pool(workers) as pool:
        for index, record in enumerate(pool.imap_unordered(play, pending), 1):
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            mine.append(record)
            if index % max(1, len(pending) // 40) == 0 or index == len(pending):
                elapsed = time.perf_counter() - started
                rate = elapsed / index
                print(
                    f"  [{index}/{len(pending)}] {elapsed / 60:.1f} min elapsed, "
                    f"{rate * (len(pending) - index) / 60:.1f} min left",
                    flush=True,
                )
    return mine


def summarize(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["stage"], record["focus"], record["config"], record["lineup"])].append(
            record
        )
    summary = {}
    for key, group in sorted(grouped.items()):
        scored = [item for item in group if not item["truncated"]]
        wins = sum(item["won"] for item in scored)
        low, high = wilson_interval(wins, len(scored))
        summary["/".join(str(part) for part in key)] = {
            "stage": key[0],
            "focus": key[1],
            "config": key[2],
            "lineup": key[3],
            "wins": wins,
            "games": len(scored),
            "truncated": len(group) - len(scored),
            "win_rate": wins / len(scored) if scored else None,
            "wilson_95": [low, high],
            "wilson_low": low,
            "mean_net_worth": sum(item["net_worth"] for item in group) / len(group),
            "seeds": sorted({item["seed"] for item in group}),
        }
    return summary


def print_summary(summary: dict[str, dict[str, Any]]) -> None:
    print(f"\n{'group':<52} {'wins':>9} {'rate':>7} {'wilson 95%':>16} {'mean nw':>10}")
    print("-" * 100)
    for name, entry in summary.items():
        rate = entry["win_rate"]
        text = " n/a" if rate is None else f"{100 * rate:5.1f}%"
        low, high = entry["wilson_95"]
        print(
            f"{name:<52} {entry['wins']:>4}/{entry['games']:<4} {text} "
            f"[{100 * low:5.1f},{100 * high:6.1f}] {entry['mean_net_worth']:>10,.0f}"
        )


def _seed_list(text: str) -> list[int]:
    """Accept ``0-7`` ranges and ``3,9`` lists interchangeably."""

    seeds: list[int] = []
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part[1:]:
            start, _, end = part.partition("-")
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(part))
    return seeds


def run(
    output_dir: Path,
    tuning_seeds: list[int],
    holdout_seeds: list[int],
    lineups: list[str],
    workers: int,
    max_decisions: int,
    skip_sweep: bool = False,
    forced_config: str | None = None,
    control: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = output_dir / "games.jsonl"
    overlap = set(tuning_seeds) & set(holdout_seeds)
    if overlap:
        raise ValueError(
            f"tuning and holdout seeds overlap on {sorted(overlap)}; "
            "a held-out block must be disjoint to mean anything"
        )

    started = time.perf_counter()
    report: dict[str, Any] = {
        "tuning_seeds": tuning_seeds,
        "holdout_seeds": holdout_seeds,
        "lineups": lineups,
        "config_grid": CONFIG_GRID,
    }

    best = forced_config
    if not skip_sweep:
        print(f"\n=== stage 1: sweep ({len(CONFIG_GRID)} configs) ===", flush=True)
        sweep_tasks = build_tasks(
            "sweep", "slayer-v1", CONFIG_GRID, ["asu-value-sweep"], tuning_seeds, max_decisions
        )
        sweep_records = run_tasks(sweep_tasks, ledger, workers)
        sweep_summary = summarize(sweep_records)
        print_summary(sweep_summary)
        report["sweep"] = sweep_summary
        ranked = sorted(
            sweep_summary.values(),
            key=lambda entry: (
                -(entry["win_rate"] or 0.0),
                -entry["wilson_low"],
                -entry["mean_net_worth"],
            ),
        )
        best = ranked[0]["config"]
        print(
            f"\nselected config: {best} "
            f"({100 * (ranked[0]['win_rate'] or 0):.1f}% on tuning seeds)",
            flush=True,
        )
    report["selected_config"] = best

    print(f"\n=== stage 2: validate '{best}' on held-out seeds ===", flush=True)
    validate_tasks = build_tasks(
        "validate", "slayer-v1", [best], lineups, holdout_seeds, max_decisions
    )
    # A lineup of three identical ASU seats needs no measured control: with four
    # copies of one policy the win rate is 25% by symmetry. Only asymmetric
    # lineups are worth spending half the budget on.
    control_lineups = [name for name in lineups if name != "asu-value-sweep"]
    if control and control_lineups:
        validate_tasks += build_tasks(
            "validate", CONTROL, ["control"], control_lineups, holdout_seeds, max_decisions
        )
    validate_records = run_tasks(validate_tasks, ledger, workers)
    validate_summary = summarize(validate_records)
    print_summary(validate_summary)
    report["validate"] = validate_summary
    report["elapsed_seconds"] = time.perf_counter() - started

    verdict = {}
    for lineup in lineups:
        focus = next(
            (
                entry
                for entry in validate_summary.values()
                if entry["focus"] == "slayer-v1" and entry["lineup"] == lineup
            ),
            None,
        )
        control = next(
            (
                entry
                for entry in validate_summary.values()
                if entry["focus"] == CONTROL and entry["lineup"] == lineup
            ),
            None,
        )
        if focus and control and focus["win_rate"] is not None:
            verdict[lineup] = {
                "slayer_win_rate": focus["win_rate"],
                "asu_control_win_rate": control["win_rate"],
                "beats_chance": focus["wilson_low"] > 0.25,
                "beats_asu_control": focus["win_rate"] > (control["win_rate"] or 0.0),
            }
    report["verdict"] = verdict
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=Path("slayer_results"))
    parser.add_argument("--tuning-seeds", type=_seed_list, default=_seed_list("0-7"))
    parser.add_argument(
        "--holdout-seeds", type=_seed_list, default=_seed_list("1000-1031")
    )
    parser.add_argument(
        "--lineups", nargs="+", choices=sorted(LINEUPS), default=sorted(LINEUPS)
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    parser.add_argument("--max-decisions", type=int, default=DEFAULT_MAX_DECISIONS)
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--config", dest="forced_config", default=None)
    parser.add_argument("--no-control", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.skip_sweep and not args.forced_config:
        args.forced_config = "baseline"
    report = run(
        args.output_dir,
        args.tuning_seeds,
        args.holdout_seeds,
        args.lineups,
        args.workers,
        args.max_decisions,
        args.skip_sweep,
        args.forced_config,
        control=not args.no_control,
    )
    path = args.output_dir / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {path}")
    for lineup, entry in report.get("verdict", {}).items():
        print(
            f"  {lineup:<18} slayer {100 * entry['slayer_win_rate']:5.1f}%  "
            f"asu {100 * (entry['asu_control_win_rate'] or 0):5.1f}%  "
            f"beats_asu={entry['beats_asu_control']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CONFIG_GRID", "LINEUPS", "build_tasks", "main", "play", "run", "summarize"]
