"""Measure live 4s/turn oracle: wall per turn + action overlap vs full search."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from monopoly_bench.adapters import ActionDecision, FixedAdapter
from monopoly_bench.arena import play_game
from monopoly_bench.engine import SharedGame, clone_env, unwrap
from monopoly_game_engine.agents_fixed import FPAgentA, FPAgentB, FPAgentC
from oracle.hybrid_config import checkpoint_kind
from oracle.rollout_policy import greedy_rollout_action
from oracle_v2.agent import LIVE_TURN_DEADLINE_S, OracleV2Agent, _turn_id, default_v2_config


class TimingOracle(OracleV2Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.turn_times: dict[tuple, float] = defaultdict(float)
        self.searches = 0
        self.greedies = 0
        self.kinds: dict[str, int] = defaultdict(int)

    def choose_action(self, env):
        t0 = time.perf_counter()
        action = super().choose_action(env)
        dt = time.perf_counter() - t0
        # Attribute time to the 4s budget key (seat/auction/oot).
        self.turn_times[_turn_id(env)] += dt
        self.kinds[str(self.last_kind)] += 1
        if self.last_used_search:
            self.searches += 1
        elif self.last_kind != "forced":
            self.greedies += 1
        return action


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sims", type=int, default=128)
    parser.add_argument("--deadline", type=float, default=LIVE_TURN_DEADLINE_S)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = default_v2_config(simulations=args.sims)
    turn_walls = []
    over = 0
    searches = greedies = 0
    kinds: dict[str, int] = defaultdict(int)
    for game_id in range(args.games):
        oracle = TimingOracle(
            0,
            cfg,
            seed=args.seed + game_id,
            live=True,
            turn_deadline_s=args.deadline,
        )
        policies = {
            0: oracle,
            1: FixedAdapter(FPAgentA),
            2: FixedAdapter(FPAgentB),
            3: FixedAdapter(FPAgentC),
        }
        # play_game expects adapters with choose_action(game, player_id, seed)
        # Wrap oracle.

        class _Wrap:
            def __init__(self, inner):
                self.inner = inner

            def choose_action(self, game, player_id, decision_seed):
                del player_id, decision_seed
                started = time.perf_counter()
                action = self.inner.choose_action(unwrap(game))
                return ActionDecision(int(action), time.perf_counter() - started)

        wrapped = {0: _Wrap(oracle), 1: policies[1], 2: policies[2], 3: policies[3]}
        result = play_game(
            game_id=game_id,
            seed=args.seed + game_id,
            policies=wrapped,
            max_rounds=80,
            record_seats=set(),
        )
        if result.crashes:
            raise RuntimeError(f"fit_turn game {game_id} crashed: {result.error}")
        walls = list(oracle.turn_times.values())
        turn_walls.extend(walls)
        over += sum(1 for w in walls if w > args.deadline + 0.05)
        searches += oracle.searches
        greedies += oracle.greedies
        for k, v in oracle.kinds.items():
            kinds[k] += v
        max_s = max(walls) if walls else 0.0
        mean_s = sum(walls) / max(len(walls), 1)
        print(
            f"game {game_id} turns={len(walls)} max={max_s:.2f}s "
            f"mean={mean_s:.2f}s over={sum(w>args.deadline+0.05 for w in walls)} "
            f"search={oracle.searches} greedy={oracle.greedies} "
            f"completed={result.completed} winner={result.winner}",
            flush=True,
        )

    # Overlap vs full search on a few event roots (accuracy proxy).
    full = OracleV2Agent(0, cfg, seed=1, live=False)
    live = OracleV2Agent(0, cfg, seed=1, live=True, turn_deadline_s=args.deadline)
    agree_event = agree_all = n_event = n_all = 0
    game = SharedGame.new(args.seed + 99, max_rounds=40)
    steps = 0

    while not game.env.done and steps < 250 and n_all < 40:
        actor = game.env.whose_turn()
        legal = game.env.get_allowed_actions(actor)
        if actor == 0 and len(legal) > 1:
            a_env = clone_env(game.env)
            b_env = clone_env(game.env)
            live._turn_key = None
            live._turn_deadline_at = None
            a = live.choose_action(a_env)
            b = full.choose_action(b_env)
            n_all += 1
            match = int(a == b)
            agree_all += match
            kind = checkpoint_kind(game.env, legal)
            if kind in {"buy", "build", "trade", "auction"}:
                n_event += 1
                agree_event += match
        action = (
            greedy_rollout_action(game.env, actor)
            if len(legal) > 1
            else legal[0]
        )
        if action not in legal:
            action = legal[0]
        game.step(action)
        steps += 1

    report = {
        "deadline_s": args.deadline,
        "games": args.games,
        "sims": args.sims,
        "n_turns": len(turn_walls),
        "mean_turn_s": sum(turn_walls) / max(len(turn_walls), 1),
        "max_turn_s": max(turn_walls) if turn_walls else None,
        "p95_turn_s": sorted(turn_walls)[int(0.95 * (len(turn_walls) - 1))] if turn_walls else None,
        "over_deadline": over,
        "searches": searches,
        "greedies": greedies,
        "kinds": dict(kinds),
        "agreement_all_multi": None if not n_all else agree_all / n_all,
        "agreement_events": None if not n_event else agree_event / n_event,
        "n_overlap_all": n_all,
        "n_overlap_events": n_event,
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
