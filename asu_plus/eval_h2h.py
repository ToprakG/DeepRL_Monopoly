"""Seat-balanced ASU+ vs ASU head-to-head harness (reuses ASU evaluate runner)."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
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


def _game_job(payload: dict[str, Any]) -> dict[str, Any]:
    weights = ASUPlusWeights(**payload["weights"])
    factory = _H2HFactory(weights)
    specs = tuple(_Spec(policy_id) for policy_id in payload["policies"])
    with preserve_global_rng():
        return _run_game(
            specs,
            focus_seat=payload["focus_seat"],
            seed=payload["seed"],
            max_decisions=payload["max_decisions"],
            factory=factory,
        )


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


def asu_plus_vs_asu(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    plus = summaries.get(ASU_PLUS_V1)
    asu = summaries.get(ASU_VALUE_ID)
    if plus is None or asu is None:
        raise RuntimeError("H2H summary missing ASU+ or ASU")
    plus_rate = plus["win_rate"] or 0.0
    asu_rate = asu["win_rate"] or 0.0
    plus_lower = plus["wilson_95"][0]
    return {
        "asu_plus_win_rate": plus_rate,
        "asu_win_rate": asu_rate,
        "asu_plus_wilson_95": plus["wilson_95"],
        "asu_wilson_95": asu["wilson_95"],
        "asu_plus_wilson_lower": plus_lower,
        "beats_asu": plus_lower > asu_rate,
        "rate_gap": plus_rate - asu_rate,
    }


def run_h2h(
    *,
    games: int,
    seed: int,
    weights: ASUPlusWeights,
    lineup: tuple[str, ...] = DEFAULT_LINEUP,
    max_decisions: int = DEFAULT_MAX_DECISIONS,
    workers: int = 1,
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
                "policies": [spec.policy_id for spec in specs],
                "focus_seat": plus_seat,
                "seed": seed + index,
                "max_decisions": max_decisions,
                "weights": asdict(weights),
            }
        )

    results: list[dict[str, Any] | None] = [None] * len(jobs)
    if workers == 1:
        for index, job in enumerate(jobs):
            results[index] = _game_job(job)
            if (index + 1) % max(1, min(8, games)) == 0 or index + 1 == games:
                print(f"h2h progress {index + 1}/{games}", flush=True)
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            for index, result in enumerate(pool.imap(_game_job, jobs, chunksize=1)):
                results[index] = result
                if (index + 1) % max(1, min(8, games)) == 0 or index + 1 == games:
                    print(f"h2h progress {index + 1}/{games}", flush=True)
    completed = [result for result in results if result is not None]
    if len(completed) != len(jobs):
        raise RuntimeError("H2H pool returned incomplete results")

    summaries = summarize_policies(completed)
    comparison = asu_plus_vs_asu(summaries)
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
    terms = ("cash", "endgame", "block", "liq")
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
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-decisions", type=int, default=DEFAULT_MAX_DECISIONS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--w-cash", type=float, default=defaults.w_cash)
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
    if save_games or "game_records" not in result:
        return result
    trimmed = dict(result)
    trimmed.pop("game_records", None)
    return trimmed


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    weights = ASUPlusWeights(
        w_cash=args.w_cash,
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
        return 0 if comparison["beats_asu"] else 2

    print("\nAblation ASU+ vs ASU:", flush=True)
    for name, block in result["ablation"].items():
        cmp_ = block["asu_plus_vs_asu"]
        print(
            f"  {name}: ASU+={cmp_['asu_plus_win_rate']:.3f} "
            f"(lo={cmp_['asu_plus_wilson_lower']:.3f}) "
            f"ASU={cmp_['asu_win_rate']:.3f} beats={cmp_['beats_asu']}",
            flush=True,
        )
    full_beats = result["ablation"]["full"]["asu_plus_vs_asu"]["beats_asu"]
    return 0 if full_beats else 2


if __name__ == "__main__":
    raise SystemExit(main())
