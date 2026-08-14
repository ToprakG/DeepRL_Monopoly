"""
Seat-balanced arena for the EXPO heuristic.

Follows the evaluation discipline described in REPO_STUDY_NOTES.md section 11:
every seed produces a four-game block in which the focus policy occupies each
physical seat exactly once, so seat and turn-order advantages cancel out.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent / "DeepRL_Monopoly"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monopoly_game_engine.actions import ActionType          # noqa: E402
from monopoly_game_engine.agents_fixed import FP_AGENT_CLASSES  # noqa: E402
from monopoly_game_engine.constants import NUM_PLAYERS       # noqa: E402
from monopoly_game_engine.env import MonopolyEnv             # noqa: E402

from EXPOSURE_HEURISTIC.agent import ExpoHeuristicAgent      # noqa: E402

FIXED_IDS = {f"fixed-{c}": cls for c, cls in zip("abcdef", FP_AGENT_CLASSES)}
DEFAULT_MAX_DECISIONS = 20_000


def build_agent(policy_id: str, seat: int):
    if policy_id == "expo":
        return ExpoHeuristicAgent(seat), True
    if policy_id in FIXED_IDS:
        return FIXED_IDS[policy_id](seat), False
    if policy_id in ("asu-value-v1", "asu-rollout-v1"):
        from ASU_FROZEN_TEACHER import ASURolloutV1, ASUValueV1
        cls = ASUValueV1 if policy_id == "asu-value-v1" else ASURolloutV1
        return cls(player_id=seat), True
    raise ValueError(f"unknown policy id: {policy_id}")


def wilson(wins: int, games: int) -> tuple[float, float]:
    if games <= 0:
        return (0.0, 1.0)
    z = 1.959963984540054
    rate = wins / games
    denom = 1 + z * z / games
    center = (rate + z * z / (2 * games)) / denom
    radius = z * math.sqrt(
        rate * (1 - rate) / games + z * z / (4 * games * games)
    ) / denom
    return (max(0.0, center - radius), min(1.0, center + radius))


def play_game(policy_ids, seed: int, max_decisions: int,
              max_rounds: int = 200) -> dict:
    random.seed(seed)
    env = MonopolyEnv(agent_ids=[0], max_rounds=max_rounds)
    agents = []
    strict = []
    for seat, pid in enumerate(policy_ids):
        agent, is_strict = build_agent(pid, seat)
        agents.append(agent)
        strict.append(is_strict)

    decisions = 0
    fallbacks = 0
    while not env.done and decisions < max_decisions:
        actor = env.whose_turn()
        allowed = env.get_allowed_actions(actor)
        action = agents[actor].choose_action(env)
        if action not in allowed:
            if strict[actor]:
                raise RuntimeError(
                    f"{policy_ids[actor]} (seat {actor}) returned illegal "
                    f"action {action}; legal={allowed[:12]}"
                )
            # Documented compatibility fallback for the scripted policies.
            fallbacks += 1
            action = (
                int(ActionType.END_TURN)
                if int(ActionType.END_TURN) in allowed
                else allowed[0]
            )
        env.step(action)
        decisions += 1

    truncated = decisions >= max_decisions and not env.done
    return {
        "policies": list(policy_ids),
        "seed": seed,
        "rounds": env.round,
        "decisions": decisions,
        "fallbacks": fallbacks,
        "truncated": truncated,
        "winner": None if truncated else env.winner(),
        "net_worth": [p.net_worth() for p in env.players],
        "bankrupt": [p.bankrupt for p in env.players],
    }


def run(focus: str, opponents, seeds, max_decisions: int,
        progress: bool = False, focus_seats=None, max_rounds: int = 200) -> dict:
    games = []
    start = time.time()
    for seed in seeds:
        for focus_seat in (focus_seats or range(NUM_PLAYERS)):
            seats = [None] * NUM_PLAYERS
            seats[focus_seat] = focus
            others = [s for s in range(NUM_PLAYERS) if s != focus_seat]
            for seat, opponent in zip(others, opponents):
                seats[seat] = opponent
            t0 = time.time()
            record = play_game(seats, seed, max_decisions, max_rounds)
            record["focus_seat"] = focus_seat
            games.append(record)
            if progress:
                win = record["winner"]
                print(
                    f"  seed={seed} focus_seat={focus_seat} "
                    f"rounds={record['rounds']:3d} "
                    f"winner={'TRUNC' if win is None else record['policies'][win]:14s} "
                    f"{time.time() - t0:6.1f}s",
                    file=sys.stderr, flush=True,
                )

    scored = [g for g in games if not g["truncated"]]
    summary = {"focus": focus, "opponents": list(opponents),
               "seeds": list(seeds), "games": len(games),
               "scored_games": len(scored),
               "truncated": len(games) - len(scored),
               "elapsed_s": round(time.time() - start, 1)}

    per_policy = {}
    for policy in {focus, *opponents}:
        appearances = [
            (g, seat)
            for g in scored
            for seat, pid in enumerate(g["policies"])
            if pid == policy
        ]
        wins = sum(1 for g, seat in appearances if g["winner"] == seat)
        survivals = sum(
            1 for g, seat in appearances if not g["bankrupt"][seat]
        )
        worth = [g["net_worth"][seat] for g, seat in appearances]
        low, high = wilson(wins, len(appearances))
        per_policy[policy] = {
            "appearances": len(appearances),
            "wins": wins,
            "win_rate": round(wins / len(appearances), 4) if appearances else None,
            "wilson_95": [round(low, 4), round(high, 4)],
            "survival_rate": round(survivals / len(appearances), 4)
            if appearances else None,
            "mean_net_worth": round(sum(worth) / len(worth), 1) if worth else None,
        }
    summary["policies"] = per_policy
    summary["mean_rounds"] = round(
        sum(g["rounds"] for g in scored) / max(1, len(scored)), 1
    )
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="EXPO seat-balanced arena")
    parser.add_argument("--focus", default="expo")
    parser.add_argument("--opponents", nargs=3,
                        default=["asu-value-v1", "fixed-b", "fixed-c"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-decisions", type=int,
                        default=DEFAULT_MAX_DECISIONS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--focus-seats", type=int, nargs="+")
    parser.add_argument("--max-rounds", type=int, default=200)
    args = parser.parse_args(argv)

    summary = run(args.focus, args.opponents, args.seeds,
                  args.max_decisions, progress=args.progress,
                  focus_seats=args.focus_seats, max_rounds=args.max_rounds)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
