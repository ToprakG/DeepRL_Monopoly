"""Seat-balanced oracle vs ASU head-to-head with margin + sims sweep."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ASU_FROZEN_TEACHER.core import ASUValueV1, preserve_global_rng
from ASU_FROZEN_TEACHER.evaluate import (
    DEFAULT_MAX_DECISIONS,
    RESULTS_DISCLAIMER,
    _run_game,
    wilson_interval,
)
from ASU_FROZEN_TEACHER.spec import FROZEN_SPEC_HASH
from monopoly_game_engine.agents_fixed import FPAgentA, FPAgentB, FPAgentC
from monopoly_game_engine.constants import NUM_PLAYERS, RULESET_VERSION

from .agent import ORACLE_V1, OracleAgent, OracleConfig

ASU_VALUE_ID = "asu-value-v1"
FIXED_A_ID = "fixed-a"
FIXED_B_ID = "fixed-b"
FIXED_C_ID = "fixed-c"
DEFAULT_LINEUP = (ORACLE_V1, ASU_VALUE_ID, FIXED_A_ID, FIXED_B_ID)
FIXED_LINEUP = (ORACLE_V1, FIXED_A_ID, FIXED_B_ID, FIXED_C_ID)
DEFAULT_WORKERS = max(1, os.cpu_count() or 1)


def _is_oracle_policy(policy_id: str) -> bool:
    return policy_id in {ORACLE_V1, "oracle-fast-v1"}


class _Spec:
    __slots__ = ("policy_id",)

    def __init__(self, policy_id: str):
        self.policy_id = policy_id


class _ScriptedAdapter:
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
    def __init__(
        self,
        config: OracleConfig,
        seed: int,
        *,
        live: bool = False,
        turn_deadline_s: float | None = None,
    ):
        self.config = config
        self.seed = seed
        self.live = live
        self.turn_deadline_s = turn_deadline_s

    def build(self, spec: _Spec, player_id: int):
        if spec.policy_id == ORACLE_V1:
            return OracleAgent(player_id, self.config, seed=self.seed + player_id)
        if spec.policy_id == "oracle-fast-v1":
            from oracle_v2.agent import OracleV2Agent

            return OracleV2Agent(
                player_id,
                self.config,
                seed=self.seed + player_id,
                live=self.live,
                turn_deadline_s=self.turn_deadline_s,
            )
        if spec.policy_id == ASU_VALUE_ID:
            return ASUValueV1(player_id)
        if spec.policy_id == FIXED_A_ID:
            return _ScriptedAdapter(FPAgentA(player_id), player_id)
        if spec.policy_id == FIXED_B_ID:
            return _ScriptedAdapter(FPAgentB(player_id), player_id)
        if spec.policy_id == FIXED_C_ID:
            return _ScriptedAdapter(FPAgentC(player_id), player_id)
        raise ValueError(f"Unsupported H2H policy {spec.policy_id!r}")


def rotate_lineup(base: tuple[str, ...], game_index: int) -> tuple[_Spec, ...]:
    if len(base) != NUM_PLAYERS:
        raise ValueError(f"lineup must have {NUM_PLAYERS} policies")
    shift = game_index % NUM_PLAYERS
    rotated = base[-shift:] + base[:-shift] if shift else base
    return tuple(_Spec(policy_id) for policy_id in rotated)


def _game_job(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    config = OracleConfig(**payload["config"])
    factory = _H2HFactory(
        config,
        seed=payload["seed"],
        live=bool(payload.get("live", False)),
        turn_deadline_s=payload.get("turn_deadline_s"),
    )
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
    margins: list[float] = []
    for game in games:
        policies = game["policies"]
        oracle_seat = policies.index(next(p for p in policies if _is_oracle_policy(p)))
        asu_seat = policies.index(ASU_VALUE_ID)
        worth = game["final_net_worth"]
        margins.append(float(worth[oracle_seat]) - float(worth[asu_seat]))
    return margins


def summarize_net_worth_margin(games: list[dict[str, Any]]) -> dict[str, Any]:
    margins = paired_net_worth_margins(games)
    n = len(margins)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "se": None,
            "std": None,
            "oracle_richer": 0,
            "oracle_richer_rate": None,
            "beats_asu_on_margin": False,
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
        "oracle_richer": richer,
        "oracle_richer_rate": richer / n,
        "beats_asu_on_margin": mean - se > 0.0,
        "margins": margins,
    }


def oracle_vs_asu(
    summaries: dict[str, dict[str, Any]],
    games: list[dict[str, Any]],
) -> dict[str, Any]:
    oracle = summaries[next(name for name in summaries if _is_oracle_policy(name))]
    asu = summaries[ASU_VALUE_ID]
    margin = summarize_net_worth_margin(games)
    return {
        "oracle_win_rate": oracle["win_rate"] or 0.0,
        "asu_win_rate": asu["win_rate"] or 0.0,
        "oracle_wilson_95": oracle["wilson_95"],
        "asu_wilson_95": asu["wilson_95"],
        "oracle_wilson_lower": oracle["wilson_95"][0],
        "beats_asu": oracle["wilson_95"][0] > (asu["win_rate"] or 0.0),
        "rate_gap": (oracle["win_rate"] or 0.0) - (asu["win_rate"] or 0.0),
        "net_worth_margin": {
            key: value for key, value in margin.items() if key != "margins"
        },
        "beats_asu_on_margin": margin["beats_asu_on_margin"],
    }


def run_h2h(
    *,
    games: int,
    seed: int,
    config: OracleConfig,
    lineup: tuple[str, ...] = DEFAULT_LINEUP,
    max_decisions: int = DEFAULT_MAX_DECISIONS,
    workers: int = DEFAULT_WORKERS,
    live: bool = False,
    turn_deadline_s: float | None = None,
) -> dict[str, Any]:
    if games < 1:
        raise ValueError("games must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")

    jobs = []
    for index in range(games):
        specs = rotate_lineup(lineup, index)
        oracle_seat = next(
            seat for seat, spec in enumerate(specs) if _is_oracle_policy(spec.policy_id)
        )
        jobs.append(
            {
                "index": index,
                "policies": [spec.policy_id for spec in specs],
                "focus_seat": oracle_seat,
                "seed": seed + index,
                "max_decisions": max_decisions,
                "config": asdict(config),
                "live": live,
                "turn_deadline_s": turn_deadline_s,
            }
        )

    results: list[dict[str, Any] | None] = [None] * len(jobs)
    if workers == 1:
        for job in jobs:
            index, result = _game_job(job)
            results[index] = result
            print(f"h2h progress {index + 1}/{games}", flush=True)
    else:
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
    payload: dict[str, Any] = {
        "ruleset": RULESET_VERSION,
        "frozen_spec_hash": FROZEN_SPEC_HASH,
        "lineup": list(lineup),
        "games": games,
        "seed": seed,
        "oracle_config": asdict(config),
        "live": live,
        "turn_deadline_s": turn_deadline_s,
        "workers": workers,
        "truncations": sum(game["truncated"] for game in completed),
        "win_rates": summaries,
        "game_records": completed,
        "disclaimer": RESULTS_DISCLAIMER,
    }
    if ASU_VALUE_ID in lineup:
        payload["oracle_vs_asu"] = oracle_vs_asu(summaries, completed)
    oracle = summaries.get(ORACLE_V1) or summaries.get("oracle-fast-v1") or {}
    payload["oracle_focus"] = {
        "wins": oracle.get("wins"),
        "games": oracle.get("games"),
        "win_rate": oracle.get("win_rate"),
        "wilson_95": oracle.get("wilson_95"),
    }
    return payload


def run_sims_sweep(
    *,
    games: int,
    seed: int,
    sims_list: list[int],
    horizon: int,
    rollouts: int,
    workers: int,
    max_decisions: int,
    checkpoint_dir: Path | None = None,
    deadline_s: float | None = None,
    early_stop_visit_lead: int | None = None,
    early_stop_min_sims: int = 16,
) -> dict[str, Any]:
    rows = []
    for sims in sims_list:
        print(f"\n=== sweep sims={sims} ===", flush=True)
        config = OracleConfig(
            simulations=sims,
            rollout_horizon=horizon,
            rollouts_per_leaf=rollouts,
            deadline_s=deadline_s,
            early_stop_visit_lead=early_stop_visit_lead,
            early_stop_min_sims=early_stop_min_sims,
        )
        report = run_h2h(
            games=games,
            seed=seed,
            config=config,
            workers=workers,
            max_decisions=max_decisions,
        )
        cmp_ = report["oracle_vs_asu"]
        margin = cmp_["net_worth_margin"]
        row = {
            "sims": sims,
            "oracle_win_rate": cmp_["oracle_win_rate"],
            "asu_win_rate": cmp_["asu_win_rate"],
            "rate_gap": cmp_["rate_gap"],
            "margin_mean": margin["mean"],
            "margin_se": margin["se"],
            "oracle_richer": margin["oracle_richer"],
            "beats_asu_on_margin": cmp_["beats_asu_on_margin"],
            "beats_asu": cmp_["beats_asu"],
        }
        rows.append(row)
        print(
            (
                f"sims={sims} oracle={cmp_['oracle_win_rate']:.3f} "
                f"asu={cmp_['asu_win_rate']:.3f} "
                f"margin={margin['mean']:.1f} ± SE {margin['se']:.1f} "
                f"richer={margin['oracle_richer']}/{margin['n']} "
                f"beats_m={cmp_['beats_asu_on_margin']}"
            ),
            flush=True,
        )
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            public = dict(report)
            public.pop("game_records", None)
            path = checkpoint_dir / f"oracle_h2h_{games}_sims{sims}.json"
            path.write_text(
                json.dumps(public, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"checkpoint wrote {path}", flush=True)
    return {"games": games, "seed": seed, "sweep": rows}


def _parse_sims(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values or any(value < 1 for value in values):
        raise ValueError("--sims must be a positive int or comma list")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seat-balanced oracle (Max-N + rollout leaf) vs ASU"
    )
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sims",
        type=str,
        default="32",
        help="Simulation budget, or comma list for a sweep (e.g. 50,100,200)",
    )
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--margin-temperature", type=float, default=2000.0)
    parser.add_argument(
        "--deadline-s",
        type=float,
        default=5.0,
        help="Per-decision search wall (seconds). 0 disables. Labels stay unlimited.",
    )
    parser.add_argument(
        "--early-stop-lead",
        type=int,
        default=8,
        help="Stop when top action leads by this many visits. 0 disables.",
    )
    parser.add_argument("--early-stop-min-sims", type=int, default=16)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--max-decisions", type=int, default=DEFAULT_MAX_DECISIONS)
    parser.add_argument(
        "--lineup",
        type=str,
        default=",".join(DEFAULT_LINEUP),
        help=(
            "Comma-separated 4-seat lineup. Use "
            f"{','.join(FIXED_LINEUP)} for oracle vs three fixed agents."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="oracle-fast-v1: 4s/turn clock, search only buy/build/trade/auction-open.",
    )
    parser.add_argument(
        "--turn-deadline-s",
        type=float,
        default=4.0,
        help="Per-turn wall for --live (seconds). Ignored unless --live.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument(
        "--save-games",
        action="store_true",
        help="Include per-game records in --output JSON",
    )
    return parser


def _parse_lineup(text: str) -> tuple[str, ...]:
    lineup = tuple(part.strip() for part in text.split(",") if part.strip())
    if len(lineup) != NUM_PLAYERS:
        raise ValueError(f"lineup must have {NUM_PLAYERS} policies, got {len(lineup)}")
    if not any(_is_oracle_policy(policy) for policy in lineup):
        raise ValueError("lineup must include an oracle policy")
    return lineup


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        sims_list = _parse_sims(args.sims)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        lineup = _parse_lineup(args.lineup)
    except ValueError as exc:
        parser.error(str(exc))

    if len(sims_list) == 1:
        config = OracleConfig(
            simulations=sims_list[0],
            rollout_horizon=args.horizon,
            rollouts_per_leaf=args.rollouts,
            margin_temperature=args.margin_temperature,
            deadline_s=None if args.deadline_s <= 0 else args.deadline_s,
            early_stop_visit_lead=None if args.early_stop_lead < 1 else args.early_stop_lead,
            early_stop_min_sims=args.early_stop_min_sims,
        )
        result = run_h2h(
            games=args.games,
            seed=args.seed,
            config=config,
            lineup=lineup,
            workers=args.workers,
            max_decisions=args.max_decisions,
            live=args.live,
            turn_deadline_s=args.turn_deadline_s if args.live else None,
        )
        public = dict(result)
        if not args.save_games:
            public.pop("game_records", None)
        payload = json.dumps(public, indent=2 if args.pretty else None, sort_keys=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n", encoding="utf-8")
        print(payload, flush=True)
        focus = result["oracle_focus"]
        print(
            (
                f"\noracle focus WR={focus['win_rate']:.3f} "
                f"({focus['wins']}/{focus['games']}) "
                f"Wilson {focus['wilson_95']} lineup={list(lineup)}"
            ),
            flush=True,
        )
        if "oracle_vs_asu" in result:
            cmp_ = result["oracle_vs_asu"]
            margin = cmp_["net_worth_margin"]
            print(
                (
                    f"oracle {cmp_['oracle_win_rate']:.3f} "
                    f"(Wilson [{cmp_['oracle_wilson_95'][0]:.3f}, "
                    f"{cmp_['oracle_wilson_95'][1]:.3f}]) "
                    f"vs ASU {cmp_['asu_win_rate']:.3f} "
                    f"| beats_asu={cmp_['beats_asu']}"
                ),
                flush=True,
            )
            print(
                (
                    f"net_worth_margin oracle-ASU: mean={margin['mean']:.1f} "
                    f"± SE {margin['se']:.1f} "
                    f"richer={margin['oracle_richer']}/{margin['n']} "
                    f"| beats_asu_on_margin={cmp_['beats_asu_on_margin']}"
                ),
                flush=True,
            )
            return 0 if cmp_["beats_asu_on_margin"] else 2
        return 0

    sweep = run_sims_sweep(
        games=args.games,
        seed=args.seed,
        sims_list=sims_list,
        horizon=args.horizon,
        rollouts=args.rollouts,
        workers=args.workers,
        max_decisions=args.max_decisions,
        checkpoint_dir=args.output.parent if args.output is not None else None,
        deadline_s=None if args.deadline_s <= 0 else args.deadline_s,
        early_stop_visit_lead=None if args.early_stop_lead < 1 else args.early_stop_lead,
        early_stop_min_sims=args.early_stop_min_sims,
    )
    payload = json.dumps(sweep, indent=2 if args.pretty else None, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload, flush=True)
    print("\nsims sweep (oracle vs ASU):", flush=True)
    print(
        f"{'sims':>6}  {'ora_wr':>7}  {'asu_wr':>7}  {'margin':>10}  {'se':>10}  {'beats_m':>7}",
        flush=True,
    )
    for row in sweep["sweep"]:
        print(
            (
                f"{row['sims']:6d}  {row['oracle_win_rate']:7.3f}  "
                f"{row['asu_win_rate']:7.3f}  {row['margin_mean']:10.1f}  "
                f"{row['margin_se']:10.1f}  {str(row['beats_asu_on_margin']):>7}"
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
