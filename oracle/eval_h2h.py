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
from competitors.factory import COMPETITOR_IDS, FIELD_COMPETITOR_IDS, build_competitor
from monopoly_game_engine.agents_fixed import FPAgentA, FPAgentB, FPAgentC
from monopoly_game_engine.constants import NUM_PLAYERS, RULESET_VERSION

from .agent import ORACLE_V1, OracleAgent, OracleConfig
from .leaves import LEAF_KINDS
from .jail import JAIL_ID, JailAgent
from .plus_agent import ORACLE_PLUS_ID, OraclePlusAgent
from toprakthegoat import GOAT_ID, GoatAgent

ASU_VALUE_ID = "asu-value-v1"
FIXED_A_ID = "fixed-a"
FIXED_B_ID = "fixed-b"
FIXED_C_ID = "fixed-c"
ORACLE_V2_ID = "oracle-fast-v1"
DEFAULT_LINEUP = (ORACLE_V1, ASU_VALUE_ID, FIXED_A_ID, FIXED_B_ID)
FIXED_LINEUP = (ORACLE_V1, FIXED_A_ID, FIXED_B_ID, FIXED_C_ID)
COMPETITOR_LINEUP = (ORACLE_V2_ID, *FIELD_COMPETITOR_IDS)
PLUS_FIELD_LINEUP = (ORACLE_PLUS_ID, "alinebidal-final", "slayer-v1", "inncenta-heuristic")
PLUS_EXPO_LINEUP = (ORACLE_PLUS_ID, "alinebidal-final", "slayer-v1", "expo-heuristic-v1")
JAIL_FIELD_LINEUP = (JAIL_ID, "slayer-v1", "underdog-v1", "inncenta-heuristic")
GOAT_FIELD_LINEUP = (GOAT_ID, "slayer-v1", "underdog-v1", "inncenta-heuristic")
DEFAULT_WORKERS = max(1, os.cpu_count() or 1)


def _is_oracle_policy(policy_id: str) -> bool:
    return policy_id in {ORACLE_V1, ORACLE_V2_ID, ORACLE_PLUS_ID, JAIL_ID, GOAT_ID}


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
        if spec.policy_id == ORACLE_V2_ID:
            from oracle_v2.agent import OracleV2Agent

            return OracleV2Agent(
                player_id,
                self.config,
                seed=self.seed + player_id,
                live=self.live,
                turn_deadline_s=self.turn_deadline_s,
            )
        if spec.policy_id == ORACLE_PLUS_ID:
            return OraclePlusAgent(
                player_id, self.config, seed=self.seed + player_id
            )
        if spec.policy_id == JAIL_ID:
            return JailAgent(player_id, self.config, seed=self.seed + player_id)
        if spec.policy_id == GOAT_ID:
            return GoatAgent(player_id, self.config, seed=self.seed + player_id)
        if spec.policy_id == ASU_VALUE_ID:
            return ASUValueV1(player_id)
        if spec.policy_id == FIXED_A_ID:
            return _ScriptedAdapter(FPAgentA(player_id), player_id)
        if spec.policy_id == FIXED_B_ID:
            return _ScriptedAdapter(FPAgentB(player_id), player_id)
        if spec.policy_id == FIXED_C_ID:
            return _ScriptedAdapter(FPAgentC(player_id), player_id)
        if spec.policy_id in COMPETITOR_IDS:
            return _ScriptedAdapter(build_competitor(spec.policy_id, player_id), player_id)
        raise ValueError(f"Unsupported H2H policy {spec.policy_id!r}")


def rotate_lineup(base: tuple[str, ...], game_index: int) -> tuple[_Spec, ...]:
    if len(base) != NUM_PLAYERS:
        raise ValueError(f"lineup must have {NUM_PLAYERS} policies")
    shift = game_index % NUM_PLAYERS
    rotated = base[-shift:] + base[:-shift] if shift else base
    return tuple(_Spec(policy_id) for policy_id in rotated)


def _focus_seat(specs: tuple[_Spec, ...]) -> int:
    for seat, spec in enumerate(specs):
        if _is_oracle_policy(spec.policy_id):
            return seat
    return 0


def _lineup_slug(lineup: tuple[str, ...]) -> str:
    return "_".join(lineup)


def _checkpoint_game_path(checkpoint_dir: Path, index: int) -> Path:
    return checkpoint_dir / f"game_{index:04d}.json"


def _load_h2h_checkpoint(checkpoint_dir: Path) -> dict[int, dict[str, Any]]:
    loaded: dict[int, dict[str, Any]] = {}
    if not checkpoint_dir.is_dir():
        return loaded
    for path in sorted(checkpoint_dir.glob("game_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        loaded[int(payload["index"])] = payload["result"]
    return loaded


def _write_h2h_game(checkpoint_dir: Path, index: int, result: dict[str, Any]) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_game_path(checkpoint_dir, index)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"index": index, "result": result}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    completed = sorted(
        int(p.stem.split("_")[1]) for p in checkpoint_dir.glob("game_*.json")
    )
    (checkpoint_dir / "manifest.json").write_text(
        json.dumps({"completed": completed, "n": len(completed)}, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_h2h_summary(checkpoint_dir: Path, payload: dict[str, Any]) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    public = dict(payload)
    public.pop("game_records", None)
    (checkpoint_dir / "summary.json").write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _assemble_report(
    *,
    lineup: tuple[str, ...],
    games: int,
    seed: int,
    config: OracleConfig,
    live: bool,
    turn_deadline_s: float | None,
    workers: int,
    game_timeout_s: float | None,
    completed: list[dict[str, Any]],
) -> dict[str, Any]:
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
        "game_timeout_s": game_timeout_s,
        "workers": workers,
        "truncations": sum(game["truncated"] for game in completed),
        "timeouts": sum(bool(game.get("timed_out")) for game in completed),
        "completed_games": len(completed),
        "win_rates": summaries,
        "game_records": completed,
        "disclaimer": RESULTS_DISCLAIMER,
    }
    if ASU_VALUE_ID in lineup and any(_is_oracle_policy(policy) for policy in lineup):
        payload["oracle_vs_asu"] = oracle_vs_asu(summaries, completed)
    if any(_is_oracle_policy(policy) for policy in lineup) and any(
        policy in COMPETITOR_IDS for policy in lineup
    ):
        payload["oracle_vs_field"] = oracle_vs_field(summaries, completed)
    oracle = next(
        (summaries[name] for name in summaries if _is_oracle_policy(name)),
        {},
    )
    payload["oracle_focus"] = {
        "wins": oracle.get("wins"),
        "games": oracle.get("games"),
        "win_rate": oracle.get("win_rate"),
        "wilson_95": oracle.get("wilson_95"),
    }
    return payload


def _game_job(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    config = OracleConfig(**payload["config"])
    factory = _H2HFactory(
        config,
        seed=payload["seed"],
        live=bool(payload.get("live", False)),
        turn_deadline_s=payload.get("turn_deadline_s"),
    )
    specs = tuple(_Spec(policy_id) for policy_id in payload["policies"])
    timeout = payload.get("game_timeout_s")
    extra = {}
    if timeout not in (None, 0):
        extra["game_timeout_s"] = float(timeout)
    with preserve_global_rng():
        result = _run_game(
            specs,
            focus_seat=payload["focus_seat"],
            seed=payload["seed"],
            max_decisions=payload["max_decisions"],
            factory=factory,
            **extra,
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


def paired_net_worth_margins(
    games: list[dict[str, Any]],
    other_id: str = ASU_VALUE_ID,
) -> list[float]:
    margins: list[float] = []
    for game in games:
        policies = game["policies"]
        if other_id not in policies:
            continue
        oracle_seat = policies.index(next(p for p in policies if _is_oracle_policy(p)))
        other_seat = policies.index(other_id)
        worth = game["final_net_worth"]
        margins.append(float(worth[oracle_seat]) - float(worth[other_seat]))
    return margins


def summarize_net_worth_margin(
    games: list[dict[str, Any]],
    other_id: str = ASU_VALUE_ID,
) -> dict[str, Any]:
    margins = paired_net_worth_margins(games, other_id)
    n = len(margins)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "se": None,
            "std": None,
            "oracle_richer": 0,
            "oracle_richer_rate": None,
            "beats_other_on_margin": False,
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
        "beats_other_on_margin": mean - se > 0.0,
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


def oracle_vs_field(
    summaries: dict[str, dict[str, Any]],
    games: list[dict[str, Any]],
) -> dict[str, Any]:
    oracle_name = next(name for name in summaries if _is_oracle_policy(name))
    oracle = summaries[oracle_name]
    others = {}
    for identifier, summary in summaries.items():
        if _is_oracle_policy(identifier):
            continue
        margin = summarize_net_worth_margin(games, identifier)
        others[identifier] = {
            "win_rate": summary["win_rate"] or 0.0,
            "wins": summary["wins"],
            "games": summary["games"],
            "wilson_95": summary["wilson_95"],
            "rate_gap": (oracle["win_rate"] or 0.0) - (summary["win_rate"] or 0.0),
            "oracle_wilson_lower_beats": oracle["wilson_95"][0] > (summary["win_rate"] or 0.0),
            "net_worth_margin": {
                key: value for key, value in margin.items() if key != "margins"
            },
        }
    return {
        "oracle_id": oracle_name,
        "oracle_win_rate": oracle["win_rate"] or 0.0,
        "oracle_wilson_95": oracle["wilson_95"],
        "opponents": others,
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
    game_timeout_s: float | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    if games < 1:
        raise ValueError("games must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")

    cached = _load_h2h_checkpoint(checkpoint_dir) if resume and checkpoint_dir else {}
    if cached:
        print(f"resume: skipping {len(cached)} completed games", flush=True)

    jobs = []
    for index in range(games):
        if index in cached:
            continue
        specs = rotate_lineup(lineup, index)
        jobs.append(
            {
                "index": index,
                "policies": [spec.policy_id for spec in specs],
                "focus_seat": _focus_seat(specs),
                "seed": seed + index,
                "max_decisions": max_decisions,
                "config": asdict(config),
                "live": live,
                "turn_deadline_s": turn_deadline_s,
                "game_timeout_s": game_timeout_s,
            }
        )

    results: list[dict[str, Any] | None] = [None] * games
    for index, result in cached.items():
        if 0 <= index < games:
            results[index] = result

    def _record(index: int, result: dict[str, Any], done: int) -> None:
        results[index] = result
        if checkpoint_dir is not None:
            _write_h2h_game(checkpoint_dir, index, result)
            finished = [game for game in results if game is not None]
            _write_h2h_summary(
                checkpoint_dir,
                _assemble_report(
                    lineup=lineup,
                    games=games,
                    seed=seed,
                    config=config,
                    live=live,
                    turn_deadline_s=turn_deadline_s,
                    workers=workers,
                    game_timeout_s=game_timeout_s,
                    completed=finished,
                ),
            )
        print(f"h2h progress {done}/{games}", flush=True)

    done = sum(game is not None for game in results)
    if workers == 1 or not jobs:
        for job in jobs:
            index, result = _game_job(job)
            done += 1
            _record(index, result, done)
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            for index, result in pool.imap_unordered(_game_job, jobs, chunksize=1):
                done += 1
                _record(index, result, done)

    completed = [result for result in results if result is not None]
    if len(completed) != games:
        raise RuntimeError("H2H pool returned incomplete results")

    payload = _assemble_report(
        lineup=lineup,
        games=games,
        seed=seed,
        config=config,
        live=live,
        turn_deadline_s=turn_deadline_s,
        workers=workers,
        game_timeout_s=game_timeout_s,
        completed=completed,
    )
    if checkpoint_dir is not None:
        _write_h2h_summary(checkpoint_dir, payload)
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
    parser.add_argument(
        "--leaf",
        choices=LEAF_KINDS,
        default="rollout",
        help="Max-N leaf: rollout (default), networth, asu, asu_plus, clone.",
    )
    parser.add_argument(
        "--leaf-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint for --leaf clone (default: new25k hybrid BC clone).",
    )
    parser.add_argument(
        "--one-ply",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="oracle-plus-v1: 1-ply shortlist search (default on).",
    )
    parser.add_argument(
        "--solvency",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="oracle-plus-v1: prefer solvent 1-ply successors (default on).",
    )
    parser.add_argument(
        "--denial",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="oracle-plus-v1: Alinebidal denial term on acquisitions (default on).",
    )
    parser.add_argument(
        "--completing-trade",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="oracle-plus-v1: take monopoly-completing buy-trades first (default on).",
    )
    parser.add_argument(
        "--denial-weight",
        type=float,
        default=1.0,
        help="Scale on the plus denial term.",
    )
    parser.add_argument(
        "--auction",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="oracle-plus-v1: Inncenta auction heuristic before 1-ply (default on).",
    )
    parser.add_argument(
        "--inncenta-trade",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="oracle-plus-v1: Inncenta completing-trade (quality, pay-up first; default on).",
    )
    parser.add_argument(
        "--networth-mix",
        type=float,
        default=0.0,
        help="Blend plus own-score toward raw net worth (0=off, 1=net worth only).",
    )
    parser.add_argument(
        "--auction-kind",
        choices=("inncenta", "asu_delta"),
        default="inncenta",
        help="oracle-plus-v1: inncenta worth or our ASU-delta auction.",
    )
    parser.add_argument(
        "--cash-gate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="oracle-plus-v1: penalize 1-ply futures below live opponent rent.",
    )
    parser.add_argument(
        "--build-first",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="oracle-plus-v1: cheapest legal house/hotel above the live-rent floor.",
    )
    parser.add_argument(
        "--race-buy",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="oracle-plus-v1: buy when a colour is already contested.",
    )
    parser.add_argument(
        "--lethal-jail",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="oracle-plus-v1: skip bail when published rent is a large cash slice.",
    )
    parser.add_argument(
        "--one-ply-trades",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="oracle-plus-v1: 1-ply only on trade decisions (default off).",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--max-decisions", type=int, default=DEFAULT_MAX_DECISIONS)
    parser.add_argument(
        "--lineup",
        action="append",
        default=None,
        help=(
            "Comma-separated 4-team field. Repeat --lineup or separate fields "
            "with ';' to run several 4-team matches. Oracle is optional. "
            f"Examples: {','.join(FIXED_LINEUP)} or {','.join(COMPETITOR_LINEUP)}. "
            "Registered competitors: "
            f"{','.join(COMPETITOR_IDS)}."
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
    parser.add_argument(
        "--game-timeout-s",
        type=float,
        default=0.0,
        help="Per-game wall (seconds). 0 disables. Timed-out games are truncated.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Write each finished game here so Colab/session death can resume.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip games already recorded in --checkpoint-dir.",
    )
    return parser


def _parse_lineup(text: str) -> tuple[str, ...]:
    lineup = tuple(part.strip() for part in text.split(",") if part.strip())
    if len(lineup) != NUM_PLAYERS:
        raise ValueError(f"lineup must have {NUM_PLAYERS} policies, got {len(lineup)}")
    return lineup


def _parse_lineups(texts: list[str]) -> list[tuple[str, ...]]:
    fields: list[tuple[str, ...]] = []
    for text in texts:
        for chunk in text.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            fields.append(_parse_lineup(chunk))
    if not fields:
        raise ValueError("at least one 4-team lineup is required")
    return fields


def _checkpoint_for_field(
    checkpoint_dir: Path | None,
    lineup: tuple[str, ...],
    field_index: int,
    n_fields: int,
) -> Path | None:
    if checkpoint_dir is None:
        return None
    if n_fields == 1:
        return checkpoint_dir
    return checkpoint_dir / f"field_{field_index:02d}_{_lineup_slug(lineup)}"


def _print_h2h_result(result: dict[str, Any], lineup: tuple[str, ...]) -> None:
    print(f"\nlineup={list(lineup)}", flush=True)
    for name, row in (result.get("win_rates") or {}).items():
        wr = row.get("win_rate")
        wr_s = "n/a" if wr is None else f"{wr:.3f}"
        print(
            (
                f"  {name}: WR={wr_s} ({row.get('wins')}/{row.get('games')}) "
                f"Wilson {row.get('wilson_95')}"
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
        return
    if "oracle_vs_field" in result:
        field = result["oracle_vs_field"]
        print(
            f"oracle WR={field['oracle_win_rate']:.3f} Wilson {field['oracle_wilson_95']}",
            flush=True,
        )
        for name, row in field["opponents"].items():
            margin = row["net_worth_margin"]
            print(
                (
                    f"  vs {name}: WR={row['win_rate']:.3f} "
                    f"({row['wins']}/{row['games']}) "
                    f"gap={row['rate_gap']:+.3f} "
                    f"margin={margin['mean']:.1f} ± SE {margin['se']:.1f}"
                ),
                flush=True,
            )


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        sims_list = _parse_sims(args.sims)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        lineups = _parse_lineups(args.lineup or [",".join(DEFAULT_LINEUP)])
    except ValueError as exc:
        parser.error(str(exc))

    if len(sims_list) > 1 and len(lineups) > 1:
        parser.error("sims sweep cannot be combined with multiple --lineup fields")

    if len(sims_list) == 1:
        config = OracleConfig(
            simulations=sims_list[0],
            rollout_horizon=args.horizon,
            rollouts_per_leaf=args.rollouts,
            margin_temperature=args.margin_temperature,
            deadline_s=None if args.deadline_s <= 0 else args.deadline_s,
            early_stop_visit_lead=None if args.early_stop_lead < 1 else args.early_stop_lead,
            early_stop_min_sims=args.early_stop_min_sims,
            leaf=args.leaf,
            leaf_checkpoint=None if args.leaf_checkpoint is None else str(args.leaf_checkpoint),
            one_ply=args.one_ply,
            solvency=args.solvency,
            denial=args.denial,
            completing_trade=args.completing_trade,
            denial_weight=args.denial_weight,
            auction=args.auction,
            inncenta_trade=args.inncenta_trade,
            networth_mix=args.networth_mix,
            auction_kind=args.auction_kind,
            family_body="one_ply",
            one_ply_trades=args.one_ply_trades,
            phase_switch=None,
            cash_gate=args.cash_gate,
            build_first=args.build_first,
            race_buy=args.race_buy,
            lethal_jail=args.lethal_jail,
        )
        reports = []
        for field_index, lineup in enumerate(lineups):
            if len(lineups) > 1:
                print(
                    f"\n=== field {field_index + 1}/{len(lineups)} {list(lineup)} ===",
                    flush=True,
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
                game_timeout_s=None if args.game_timeout_s <= 0 else args.game_timeout_s,
                checkpoint_dir=_checkpoint_for_field(
                    args.checkpoint_dir, lineup, field_index, len(lineups)
                ),
                resume=args.resume,
            )
            reports.append(result)
            _print_h2h_result(result, lineup)
        public_reports = []
        for result in reports:
            public = dict(result)
            if not args.save_games:
                public.pop("game_records", None)
            public_reports.append(public)
        payload_obj: dict[str, Any] | list[dict[str, Any]]
        if len(public_reports) == 1:
            payload_obj = public_reports[0]
        else:
            payload_obj = {"fields": public_reports}
        payload = json.dumps(
            payload_obj, indent=2 if args.pretty else None, sort_keys=True
        )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n", encoding="utf-8")
        print(payload, flush=True)
        if len(reports) == 1 and "oracle_vs_asu" in reports[0]:
            return 0 if reports[0]["oracle_vs_asu"]["beats_asu_on_margin"] else 2
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
