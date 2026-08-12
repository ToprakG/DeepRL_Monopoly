"""Seat-balanced ASU+ vs ASU head-to-head harness (reuses ASU evaluate runner)."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ASU_FROZEN_TEACHER.core import ASUValueV1, preserve_global_rng  # noqa: E402
from ASU_FROZEN_TEACHER.evaluate import (  # noqa: E402
    DEFAULT_MAX_DECISIONS,
    RESULTS_DISCLAIMER,
    _run_game,
    wilson_interval,
)
from ASU_FROZEN_TEACHER.spec import FROZEN_SPEC_HASH  # noqa: E402
from monopoly_game_engine.agents_fixed import FPAgentA, FPAgentB  # noqa: E402
from monopoly_game_engine.constants import NUM_PLAYERS, RULESET_VERSION  # noqa: E402

from .agent import ASU_PLUS_V1, ASUPlusV1
from .value import ASUPlusWeights

ASU_VALUE_ID = "asu-value-v1"
FIXED_A_ID = "fixed-a"
FIXED_B_ID = "fixed-b"
DEFAULT_LINEUP = (ASU_PLUS_V1, ASU_VALUE_ID, FIXED_A_ID, FIXED_B_ID)
DEFAULT_WORKERS = max(1, os.cpu_count() or 1)


class _Spec:
    __slots__ = ("policy_id",)

    def __init__(self, policy_id: str):
        self.policy_id = policy_id


class _ScriptedAdapter:
    """Match evaluate.py's fixed-agent compatibility fallback."""

    def __init__(self, agent, player_id: int):
        self.agent = agent
        self.player_id = player_id
        self.fallbacks = 0

    def choose_action(self, env) -> int:
        from monopoly_game_engine.actions import ActionType

        allowed = env.get_allowed_actions(self.player_id)
        action = self.agent.choose_action(env)
        if action in allowed:
            return action
        self.fallbacks += 1
        if int(ActionType.END_TURN) in allowed:
            return int(ActionType.END_TURN)
        return allowed[0]


class _H2HFactory:
    """Build ASU+ / ASU / fixed policies for a seat (mirrors evaluate.AgentFactory)."""

    def __init__(self, weights: ASUPlusWeights):
        self.weights = weights

    def build(self, spec: _Spec, player_id: int):
        if spec.policy_id == ASU_PLUS_V1:
            return ASUPlusV1(player_id, self.weights)
        if spec.policy_id == ASU_VALUE_ID:
            return ASUValueV1(player_id)
        if spec.policy_id == FIXED_A_ID:
            return _ScriptedAdapter(FPAgentA(player_id), player_id)
        if spec.policy_id == FIXED_B_ID:
            return _ScriptedAdapter(FPAgentB(player_id), player_id)
        raise ValueError(f"Unsupported H2H policy {spec.policy_id!r}")


def rotate_lineup(base: tuple[str, ...], game_index: int) -> tuple[_Spec, ...]:
    """Rotate the four policy slots so ASU+ and ASU cover every seat evenly."""

    if len(base) != NUM_PLAYERS:
        raise ValueError(f"lineup must have {NUM_PLAYERS} policies")
    shift = game_index % NUM_PLAYERS
    rotated = base[-shift:] + base[:-shift] if shift else base
    return tuple(_Spec(policy_id) for policy_id in rotated)


def _game_job(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    weights = ASUPlusWeights(**payload["weights"])
    factory = _H2HFactory(weights)
    specs = tuple(_Spec(policy_id) for policy_id in payload["policies"])
    with preserve_global_rng():
        result = _run_game(
            specs,
            focus_seat=payload["focus_seat"],
            seed=payload["seed"],
            max_decisions=payload["max_decisions"],
            factory=factory,
        )
    return int(payload["index"]), result


def summarize_policies(games: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    identifiers = sorted({policy for game in games for policy in game["policies"]})
    summaries = {}
    for identifier in identifiers:
        appearances = [
            (game, seat)
            for game in games
            if not game["truncated"]
            for seat, policy in enumerate(game["policies"])
            if policy == identifier
        ]
        wins = sum(game["winner"] == seat for game, seat in appearances)
        interval = wilson_interval(wins, len(appearances))
        rate = wins / len(appearances) if appearances else None
        summaries[identifier] = {
            "wins": wins,
            "games": len(appearances),
            "win_rate": rate,
            "win_rate_percent": None if rate is None else 100.0 * rate,
            "wilson_95": list(interval),
        }
    return summaries


def paired_net_worth_margins(games: list[dict[str, Any]]) -> list[float]:
    """Per-game ASU+ final net worth minus ASU final net worth (paired)."""

    margins: list[float] = []
    for game in games:
        policies = game["policies"]
        try:
            plus_seat = policies.index(ASU_PLUS_V1)
            asu_seat = policies.index(ASU_VALUE_ID)
        except ValueError as exc:
            raise RuntimeError("H2H game missing ASU+ or ASU seat") from exc
        worth = game["final_net_worth"]
        margins.append(float(worth[plus_seat]) - float(worth[asu_seat]))
    return margins


def summarize_net_worth_margin(games: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean paired net-worth margin ± standard error (ASU+ − ASU)."""

    margins = paired_net_worth_margins(games)
    n = len(margins)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "se": None,
            "std": None,
            "mean_minus_se": None,
            "mean_plus_se": None,
            "asu_plus_richer": 0,
            "asu_plus_richer_rate": None,
            "beats_asu_on_margin": False,
            "margins": [],
        }
    mean = sum(margins) / n
    if n > 1:
        variance = sum((value - mean) ** 2 for value in margins) / (n - 1)
        std = math.sqrt(variance)
        se = std / math.sqrt(n)
    else:
        std = 0.0
        se = 0.0
    richer = sum(value > 0.0 for value in margins)
    return {
        "n": n,
        "mean": mean,
        "se": se,
        "std": std,
        "mean_minus_se": mean - se,
        "mean_plus_se": mean + se,
        "asu_plus_richer": richer,
        "asu_plus_richer_rate": richer / n,
        # Paired continuous signal: mean margin exceeds one SE above zero.
        "beats_asu_on_margin": mean - se > 0.0,
        "margins": margins,
    }


def asu_plus_vs_asu(
    summaries: dict[str, dict[str, Any]],
    games: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    plus = summaries.get(ASU_PLUS_V1)
    asu = summaries.get(ASU_VALUE_ID)
    if plus is None or asu is None:
        raise RuntimeError("H2H summary missing ASU+ or ASU")
    plus_rate = plus["win_rate"] or 0.0
    asu_rate = asu["win_rate"] or 0.0
    plus_lower = plus["wilson_95"][0]
    margin = summarize_net_worth_margin(games or [])
    return {
        "asu_plus_win_rate": plus_rate,
        "asu_win_rate": asu_rate,
        "asu_plus_wilson_95": plus["wilson_95"],
        "asu_wilson_95": asu["wilson_95"],
        "asu_plus_wilson_lower": plus_lower,
        "beats_asu": plus_lower > asu_rate,
        "rate_gap": plus_rate - asu_rate,
        "net_worth_margin": {
            key: value
            for key, value in margin.items()
            if key != "margins"  # keep public JSON small; full list stays in game path
        },
        "net_worth_margins": margin["margins"],
        "beats_asu_on_margin": margin["beats_asu_on_margin"],
    }


def run_h2h(
    *,
    games: int,
    seed: int,
    weights: ASUPlusWeights,
    lineup: tuple[str, ...] = DEFAULT_LINEUP,
    max_decisions: int = DEFAULT_MAX_DECISIONS,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    if games < 1:
        raise ValueError("games must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")

    jobs = []
    for index in range(games):
        specs = rotate_lineup(lineup, index)
        plus_seat = next(
            seat for seat, spec in enumerate(specs) if spec.policy_id == ASU_PLUS_V1
        )
        jobs.append(
            {
                "index": index,
                "policies": [spec.policy_id for spec in specs],
                "focus_seat": plus_seat,
                "seed": seed + index,
                "max_decisions": max_decisions,
                "weights": asdict(weights),
            }
        )

    results: list[dict[str, Any] | None] = [None] * len(jobs)
    if workers == 1:
        for job in jobs:
            index, result = _game_job(job)
            results[index] = result
            print(f"h2h progress {index + 1}/{games}", flush=True)
    else:
        # Unordered: report as each game finishes instead of stalling on job 0.
        with mp.get_context("spawn").Pool(workers) as pool:
            done = 0
            for index, result in pool.imap_unordered(_game_job, jobs, chunksize=1):
                results[index] = result
                done += 1
                print(f"h2h progress {done}/{games}", flush=True)
    completed = [result for result in results if result is not None]
    if len(completed) != len(jobs):
        raise RuntimeError("H2H pool returned incomplete results")

    summaries = summarize_policies(completed)
    comparison = asu_plus_vs_asu(summaries, completed)
    return {
        "ruleset": RULESET_VERSION,
        "frozen_spec_hash": FROZEN_SPEC_HASH,
        "lineup": list(lineup),
        "games": games,
        "seed": seed,
        "weights": asdict(weights),
        "workers": workers,
        "truncations": sum(game["truncated"] for game in completed),
        "win_rates": summaries,
        "asu_plus_vs_asu": comparison,
        "game_records": completed,
        "disclaimer": RESULTS_DISCLAIMER,
    }


def run_ablation(
    *,
    games: int,
    seed: int,
    weights: ASUPlusWeights,
    max_decisions: int,
    workers: int,
) -> dict[str, Any]:
    terms = ("endgame", "block", "liq")
    reports = {
        "full": run_h2h(
            games=games,
            seed=seed,
            weights=weights,
            max_decisions=max_decisions,
            workers=workers,
        )
    }
    for term in terms:
        reports[f"ablate_{term}"] = run_h2h(
            games=games,
            seed=seed,
            weights=weights.ablate(term),
            max_decisions=max_decisions,
            workers=workers,
        )
    summary = {
        name: {
            "weights": report["weights"],
            "asu_plus_vs_asu": report["asu_plus_vs_asu"],
            "win_rates": report["win_rates"],
        }
        for name, report in reports.items()
    }
    return {"ablation": summary, "games": games, "seed": seed}


def _parser() -> argparse.ArgumentParser:
    defaults = ASUPlusWeights()
    parser = argparse.ArgumentParser(
        description="Seat-balanced ASU+ vs ASU head-to-head on ppo-plus-v2"
    )
    parser.add_argument("--games", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-decisions", type=int, default=DEFAULT_MAX_DECISIONS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--w-endgame", type=float, default=defaults.w_endgame)
    parser.add_argument(
        "--endgame-start-round",
        type=int,
        default=defaults.endgame_start_round,
    )
    parser.add_argument("--w-block", type=float, default=defaults.w_block)
    parser.add_argument("--w-liq", type=float, default=defaults.w_liq)
    parser.add_argument(
        "--ablate",
        action="store_true",
        help="Run full weights plus each new term zeroed one at a time",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--save-games",
        action="store_true",
        help="Include per-game records in --output JSON (large)",
    )
    parser.add_argument("--pretty", action="store_true")
    return parser


def _public_result(result: dict[str, Any], *, save_games: bool) -> dict[str, Any]:
    trimmed = dict(result)
    if not save_games:
        trimmed.pop("game_records", None)
    comparison = trimmed.get("asu_plus_vs_asu")
    if isinstance(comparison, dict):
        # Always keep summary margin stats; drop raw per-game vector unless saving games.
        comparison = dict(comparison)
        if not save_games:
            comparison.pop("net_worth_margins", None)
        trimmed["asu_plus_vs_asu"] = comparison
    return trimmed


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    weights = ASUPlusWeights(
        w_endgame=args.w_endgame,
        endgame_start_round=args.endgame_start_round,
        w_block=args.w_block,
        w_liq=args.w_liq,
    )
    try:
        if args.ablate:
            result = run_ablation(
                games=args.games,
                seed=args.seed,
                weights=weights,
                max_decisions=args.max_decisions,
                workers=args.workers,
            )
        else:
            result = run_h2h(
                games=args.games,
                seed=args.seed,
                weights=weights,
                max_decisions=args.max_decisions,
                workers=args.workers,
            )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    public = _public_result(result, save_games=args.save_games)
    payload = json.dumps(public, indent=2 if args.pretty else None, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload, flush=True)

    if not args.ablate:
        comparison = result["asu_plus_vs_asu"]
        margin = comparison["net_worth_margin"]
        mean = margin["mean"]
        se = margin["se"]
        print(
            (
                f"\nASU+ {comparison['asu_plus_win_rate']:.3f} "
                f"(Wilson [{comparison['asu_plus_wilson_95'][0]:.3f}, "
                f"{comparison['asu_plus_wilson_95'][1]:.3f}]) "
                f"vs ASU {comparison['asu_win_rate']:.3f} "
                f"| beats_asu={comparison['beats_asu']}"
            ),
            flush=True,
        )
        if mean is None or se is None:
            print("net_worth_margin: n=0", flush=True)
        else:
            print(
                (
                    f"net_worth_margin ASU+-ASU: mean={mean:.1f} ± SE {se:.1f} "
                    f"(mean±SE [{mean - se:.1f}, {mean + se:.1f}]) "
                    f"richer={margin['asu_plus_richer']}/{margin['n']} "
                    f"| beats_asu_on_margin={comparison['beats_asu_on_margin']}"
                ),
                flush=True,
            )
        return 0 if comparison["beats_asu_on_margin"] else 2

    print("\nAblation ASU+ vs ASU:", flush=True)
    for name, block in result["ablation"].items():
        cmp_ = block["asu_plus_vs_asu"]
        margin = cmp_["net_worth_margin"]
        mean = margin["mean"]
        se = margin["se"]
        mean_txt = "n/a" if mean is None or se is None else f"{mean:.1f}±{se:.1f}"
        print(
            f"  {name}: ASU+={cmp_['asu_plus_win_rate']:.3f} "
            f"(lo={cmp_['asu_plus_wilson_lower']:.3f}) "
            f"ASU={cmp_['asu_win_rate']:.3f} beats={cmp_['beats_asu']} "
            f"margin={mean_txt} beats_margin={cmp_['beats_asu_on_margin']}",
            flush=True,
        )
    full_beats = result["ablation"]["full"]["asu_plus_vs_asu"]["beats_asu_on_margin"]
    return 0 if full_beats else 2


if __name__ == "__main__":
    raise SystemExit(main())
