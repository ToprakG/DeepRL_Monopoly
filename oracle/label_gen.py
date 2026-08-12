"""Offline oracle label generation (embarrassingly parallel across games)."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from ASU_FROZEN_TEACHER.core import ASUValueV1, preserve_global_rng
from monopoly_bench.engine import (
    ACTION_SPACE_SIZE,
    MAX_DECISIONS_PER_TURN,
    NUM_PLAYERS,
    SharedGame,
    legal_mask,
)
from monopoly_game_engine.agents_fixed import FPAgentA, FPAgentB, FPAgentD
from monopoly_game_engine.constants import RULESET_VERSION

from .agent import ORACLE_V1, OracleAgent, OracleConfig, build_oracle_search, oracle_config_from_args
from .hybrid_config import (
    HybridLabelConfig,
    checkpoint_kind,
    is_event_checkpoint,
    lineup_kind_for_game,
)
from .rollout_policy import greedy_rollout_action

DEFAULT_WORKERS = max(1, os.cpu_count() or 1)


class _Scripted:
    def __init__(self, agent):
        self.agent = agent
        self.player_id = agent.player_id

    def choose_action(self, env) -> int:
        from monopoly_game_engine.actions import ActionType

        allowed = env.get_allowed_actions(self.player_id)
        action = self.agent.choose_action(env)
        if action in allowed:
            return action
        if int(ActionType.END_TURN) in allowed:
            return int(ActionType.END_TURN)
        return allowed[0]


class _GreedySeat:
    def __init__(self, player_id: int):
        self.player_id = player_id

    def choose_action(self, env) -> int:
        return greedy_rollout_action(env, self.player_id)


def _build_lineup(mode: str, config: OracleConfig, seed: int) -> list[Any]:
    if mode == "self":
        return [OracleAgent(seat, config, seed=seed + seat) for seat in range(NUM_PLAYERS)]
    if mode == "vs_asu":
        return [
            OracleAgent(0, config, seed=seed),
            ASUValueV1(1),
            _Scripted(FPAgentA(2)),
            _Scripted(FPAgentB(3)),
        ]
    if mode == "vs_greedy":
        return [
            OracleAgent(0, config, seed=seed),
            _GreedySeat(1),
            _GreedySeat(2),
            _GreedySeat(3),
        ]
    raise ValueError(f"Unknown label mode {mode!r}")


def _build_hybrid_players(
    kind: str,
    seed: int,
    *,
    label_one_seat_only: bool = True,
) -> tuple[list[Any], tuple[int, ...]]:
    """Play policies + which seats produce checkpoint labels."""

    if kind == "self":
        players = [_GreedySeat(seat) for seat in range(NUM_PLAYERS)]
        if label_one_seat_only:
            return players, (seed % NUM_PLAYERS,)
        return players, tuple(range(NUM_PLAYERS))
    if kind == "pool":
        focus = seed % NUM_PLAYERS
        # One hybrid label seat + ASU + two frozen scripted bots (same mix as H2H).
        pool_factories = [ASUValueV1, FPAgentA, FPAgentB]
        players: list[Any] = []
        pool_idx = 0
        for seat in range(NUM_PLAYERS):
            if seat == focus:
                players.append(_GreedySeat(seat))
                continue
            factory = pool_factories[pool_idx]
            pool_idx += 1
            raw = factory(seat)
            players.append(raw if factory is ASUValueV1 else _Scripted(raw))
        return players, (focus,)
    raise ValueError(f"Unknown hybrid lineup kind {kind!r}")


def _oracle_config_dict(config: OracleConfig) -> dict[str, Any]:
    return {
        "simulations": config.simulations,
        "c_puct": config.c_puct,
        "max_depth": config.max_depth,
        "max_width": config.max_width,
        "rollout_horizon": config.rollout_horizon,
        "rollouts_per_leaf": config.rollouts_per_leaf,
        "margin_temperature": config.margin_temperature,
        "prior_peak": config.prior_peak,
    }


def _record_label(
    *,
    game_id: int,
    seed: int,
    step: int,
    actor: int,
    env,
    legal: tuple[int, ...],
    result,
) -> dict[str, Any]:
    visits = {int(k): int(v) for k, v in result.visits.items()}
    total = float(sum(visits.values())) or 1.0
    return {
        "game_id": game_id,
        "seed": seed,
        "step": step,
        "actor": actor,
        "state": env._get_state(actor).astype(np.float32),
        "legal_mask": legal_mask(legal).astype(np.bool_),
        "visit_distribution": {str(k): v / total for k, v in visits.items()},
        "visits": {str(k): v for k, v in visits.items()},
        "backed_up_value_vector": list(result.root_value),
        "selected_action": int(result.chosen_action),
        "simulations": result.simulations,
        "checkpoint": True,
    }


def _collect_game(payload: dict[str, Any]) -> dict[str, Any]:
    config = OracleConfig(**payload["config"])
    seed = int(payload["seed"])
    game_id = int(payload["game_id"])
    mode = payload["mode"]
    max_rounds = int(payload["max_rounds"])
    subsample = int(payload["subsample"])
    hybrid = payload.get("hybrid")

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    with preserve_global_rng():
        game = SharedGame.new(seed, max_rounds=max_rounds)
        step = 0
        labeled = 0
        checkpoints_seen = 0

        if hybrid is not None:
            hybrid_cfg = HybridLabelConfig(**hybrid)
            kind = payload["lineup_kind"]
            players, label_seats = _build_hybrid_players(
                kind,
                seed,
                label_one_seat_only=hybrid_cfg.label_one_seat_only,
            )
            search = build_oracle_search(hybrid_cfg.oracle_config())
            label_seat_set = set(label_seats)
            decision_seed = seed * 1_000_003
            # One build label per pre-roll menu; one trade label per round.
            build_menu_labeled: set[tuple[int, int]] = set()
            trade_round_labeled: set[tuple[int, int]] = set()
            last_phase = game.env.phase
            last_round = game.env.round

            while not game.env.done and step < max_rounds * NUM_PLAYERS * MAX_DECISIONS_PER_TURN:
                actor = game.env.whose_turn()
                legal = tuple(game.env.get_allowed_actions(actor))
                phase = game.env.phase
                if game.env.round != last_round:
                    trade_round_labeled.clear()
                    last_round = game.env.round
                if phase != last_phase:
                    if phase != "pre_roll":
                        build_menu_labeled.clear()
                    last_phase = phase
                if len(legal) == 1:
                    action = legal[0]
                else:
                    build_key = (game.env.round, actor)
                    trade_key = (game.env.round, actor)
                    kind = checkpoint_kind(game.env, legal)
                    checkpoint = is_event_checkpoint(
                        game.env,
                        legal,
                        already_labeled_build_menu=build_key in build_menu_labeled,
                        already_labeled_trade_round=trade_key in trade_round_labeled,
                    )
                    if checkpoint:
                        checkpoints_seen += 1
                    if (
                        checkpoint
                        and actor in label_seat_set
                        and hybrid_cfg.label_checkpoints_only
                    ):
                        result = search.choose_action(game.env, actor, decision_seed + step)
                        records.append(
                            _record_label(
                                game_id=game_id,
                                seed=seed,
                                step=step,
                                actor=actor,
                                env=game.env,
                                legal=legal,
                                result=result,
                            )
                        )
                        labeled += 1
                        if kind == "build":
                            build_menu_labeled.add(build_key)
                        elif kind == "trade":
                            trade_round_labeled.add(trade_key)
                    # Hybrid (or pool bot) drives the actual move.
                    action = int(players[actor].choose_action(game.env))
                if action not in legal:
                    # Pool bots can return illegal; fall back safely.
                    action = int(greedy_rollout_action(game.env, actor))
                    if action not in legal:
                        action = legal[0]
                game.step(action)
                step += 1
        else:
            agents = _build_lineup(mode, config, seed)
            while not game.env.done and step < max_rounds * NUM_PLAYERS * MAX_DECISIONS_PER_TURN:
                actor = game.env.whose_turn()
                legal = tuple(game.env.get_allowed_actions(actor))
                agent = agents[actor]
                if len(legal) == 1:
                    action = legal[0]
                elif isinstance(agent, OracleAgent) and (
                    subsample <= 1 or (labeled % subsample == 0)
                ):
                    result = agent.search_action(game.env)
                    action = int(result.chosen_action)
                    records.append(
                        _record_label(
                            game_id=game_id,
                            seed=seed,
                            step=step,
                            actor=actor,
                            env=game.env,
                            legal=legal,
                            result=result,
                        )
                    )
                    labeled += 1
                else:
                    action = int(agent.choose_action(game.env))
                    if isinstance(agent, OracleAgent):
                        labeled += 1
                if action not in legal:
                    raise RuntimeError(f"illegal action {action} at step {step}")
                game.step(action)
                step += 1
            checkpoints_seen = labeled

    winner = game.env.winner() if game.env.done else None
    outcome = np.zeros(NUM_PLAYERS, dtype=np.float32)
    if winner is not None:
        outcome[winner] = 1.0
    for record in records:
        record["outcome"] = outcome.tolist()
        record["winner"] = winner
    return {
        "game_id": game_id,
        "seed": seed,
        "winner": winner,
        "truncated": not game.env.done,
        "records": records,
        "n_labels": len(records),
        "checkpoints_seen": checkpoints_seen,
        "lineup_kind": payload.get("lineup_kind"),
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_label_gen(
    *,
    games: int,
    seed: int,
    config: OracleConfig,
    mode: str = "vs_asu",
    workers: int = DEFAULT_WORKERS,
    max_rounds: int = 200,
    subsample: int = 1,
    hybrid: HybridLabelConfig | None = None,
) -> dict[str, Any]:
    jobs = []
    for index in range(games):
        if hybrid is not None:
            kind = lineup_kind_for_game(index, hybrid)
            job_mode = f"hybrid_{kind}"
            hybrid_dict = hybrid.as_dict()
        else:
            kind = None
            job_mode = mode
            hybrid_dict = None
        jobs.append(
            {
                "game_id": index,
                "seed": seed + index,
                "config": _oracle_config_dict(config),
                "mode": job_mode,
                "lineup_kind": kind,
                "hybrid": hybrid_dict,
                "max_rounds": max_rounds,
                "subsample": subsample,
            }
        )

    wall_started = time.perf_counter()
    results: list[dict[str, Any] | None] = [None] * games
    if workers == 1:
        for job in jobs:
            result = _collect_game(job)
            results[result["game_id"]] = result
            print(
                f"label progress {result['game_id'] + 1}/{games} "
                f"labels={result['n_labels']} kind={result.get('lineup_kind')}",
                flush=True,
            )
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            done = 0
            for result in pool.imap_unordered(_collect_game, jobs, chunksize=1):
                results[result["game_id"]] = result
                done += 1
                print(
                    f"label progress {done}/{games} "
                    f"labels={result['n_labels']} kind={result.get('lineup_kind')}",
                    flush=True,
                )
    wall_elapsed = time.perf_counter() - wall_started
    completed = [result for result in results if result is not None]
    labels = [record for game in completed for record in game["records"]]
    labels_per_game = [game["n_labels"] for game in completed]
    mean_labels = float(sum(labels_per_game) / len(labels_per_game)) if labels_per_game else 0.0
    games_per_hour = (len(completed) / wall_elapsed) * 3600.0 if wall_elapsed > 0 else 0.0
    return {
        "ruleset": RULESET_VERSION,
        "teacher": ORACLE_V1,
        "action_dim": ACTION_SPACE_SIZE,
        "games": games,
        "seed": seed,
        "mode": mode if hybrid is None else "hybrid",
        "hybrid_config": None if hybrid is None else hybrid.as_dict(),
        "config": _oracle_config_dict(config),
        "n_labels": len(labels),
        "throughput": {
            "wall_seconds": wall_elapsed,
            "games_per_hour": games_per_hour,
            "mean_labels_per_game": mean_labels,
            "labels_per_hour": (len(labels) / wall_elapsed) * 3600.0 if wall_elapsed > 0 else 0.0,
            "workers": workers,
        },
        "game_summaries": [
            {
                "game_id": game["game_id"],
                "seed": game["seed"],
                "winner": game["winner"],
                "truncated": game["truncated"],
                "n_labels": game["n_labels"],
                "checkpoints_seen": game.get("checkpoints_seen"),
                "lineup_kind": game.get("lineup_kind"),
                "elapsed_seconds": game.get("elapsed_seconds"),
            }
            for game in completed
        ],
        "labels": labels,
    }


def _save(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = report["labels"]
    meta = {key: value for key, value in report.items() if key != "labels"}
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if labels:
        np.savez_compressed(
            path.with_suffix(".npz"),
            states=np.stack([label["state"] for label in labels]),
            legal_masks=np.stack([label["legal_mask"] for label in labels]),
            actors=np.asarray([label["actor"] for label in labels], dtype=np.int16),
            selected_actions=np.asarray(
                [label["selected_action"] for label in labels], dtype=np.int32
            ),
            values=np.stack(
                [np.asarray(label["backed_up_value_vector"], dtype=np.float32) for label in labels]
            ),
            outcomes=np.stack(
                [np.asarray(label["outcome"], dtype=np.float32) for label in labels]
            ),
        )
        with path.with_suffix(".jsonl").open("w", encoding="utf-8") as handle:
            for label in labels:
                row = {
                    "game_id": label["game_id"],
                    "seed": label["seed"],
                    "step": label["step"],
                    "actor": label["actor"],
                    "selected_action": label["selected_action"],
                    "visit_distribution": label["visit_distribution"],
                    "backed_up_value_vector": label["backed_up_value_vector"],
                    "simulations": label["simulations"],
                    "winner": label["winner"],
                    "checkpoint": label.get("checkpoint", False),
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate oracle MCTS teacher labels")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--margin-temperature", type=float, default=2000.0)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--mode",
        choices=("self", "vs_asu", "vs_greedy", "hybrid"),
        default="vs_asu",
        help="hybrid = DealBuilder play + 128-sim labels at buy/build/trade/auction",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Use HybridLabelConfig defaults (128 sims, checkpoint-only, self+pool mix)",
    )
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    hybrid_cfg: HybridLabelConfig | None = None
    if args.calibrate or args.mode == "hybrid":
        hybrid_cfg = HybridLabelConfig()
        if not args.calibrate:
            # Allow CLI overrides when not forcing calibrate defaults.
            hybrid_cfg = HybridLabelConfig(
                simulations=int(args.sims),
                rollout_horizon=int(args.horizon),
                rollouts_per_leaf=int(args.rollouts),
                margin_temperature=float(args.margin_temperature),
            )
        config = hybrid_cfg.oracle_config()
        mode = "hybrid"
    else:
        config = oracle_config_from_args(args)
        mode = args.mode

    report = run_label_gen(
        games=args.games,
        seed=args.seed,
        config=config,
        mode=mode,
        workers=args.workers,
        subsample=args.subsample,
        hybrid=hybrid_cfg,
    )
    _save(report, args.output)
    thr = report["throughput"]
    print(
        f"wrote {report['n_labels']} labels from {args.games} games -> {args.output}",
        flush=True,
    )
    print(
        (
            f"throughput: {thr['games_per_hour']:.2f} games/hour | "
            f"{thr['mean_labels_per_game']:.1f} labels/game | "
            f"{thr['labels_per_hour']:.1f} labels/hour | "
            f"wall={thr['wall_seconds']:.1f}s workers={thr['workers']}"
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
