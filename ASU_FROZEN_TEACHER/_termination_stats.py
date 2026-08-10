"""Cap-vs-bankruptcy termination stats for asu-value-v1 self-play and vs fixed bots.

Classifies each finished game by how it ended:
  - "elimination": bankruptcy left exactly one solvent player (env.winner() is
    unambiguous, decided by survival).
  - "round_cap": env.round reached max_rounds with 2+ solvent players still
    standing; env.winner() fell back to the net-worth tiebreak.
  - "decision_cap": the max_decisions safety valve fired before env.done (not
    a real ending; excluded from the elimination/round_cap split).

Not part of the public CLI -- reuses ASU_FROZEN_TEACHER.evaluate's private
game-seeding helpers with an extra bankruptcy-count observation that
_run_game does not expose.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ASU_FROZEN_TEACHER.core import preserve_global_rng
from ASU_FROZEN_TEACHER.evaluate import (
    AgentFactory,
    AgentSpec,
    RESULTS_DISCLAIMER,
    _new_seeded_game,
    parse_agent_spec,
)
from ASU_FROZEN_TEACHER.spec import FROZEN_SPEC_HASH
from monopoly_game_engine.constants import NUM_PLAYERS

DEFAULT_MAX_DECISIONS = 20_000


def _run_game_with_ending(
    specs: tuple[AgentSpec, ...],
    seed: int,
    max_decisions: int,
    factory: AgentFactory,
) -> dict[str, Any]:
    started = time.perf_counter()
    game = _new_seeded_game(seed)
    agents = [factory.build(spec, seat) for seat, spec in enumerate(specs)]
    decisions = 0
    while not game.env.done and decisions < max_decisions:
        actor = game.env.whose_turn()
        allowed = game.env.get_allowed_actions(actor)
        action = agents[actor].choose_action(game.env)
        if action not in allowed:
            raise RuntimeError(
                f"{specs[actor].policy_id} returned illegal action {action} "
                f"for seat {actor}; legal actions are {allowed}"
            )
        game.step(action)
        decisions += 1

    active_at_end = sum(not player.bankrupt for player in game.env.players)
    if not game.env.done:
        ended_by = "decision_cap"
    elif active_at_end == 1:
        ended_by = "elimination"
    else:
        ended_by = "round_cap"
    winner = game.env.winner() if game.env.done else None
    return {
        "seed": seed,
        "policies": [spec.policy_id for spec in specs],
        "winner": winner,
        "winner_policy": None if winner is None else specs[winner].policy_id,
        "ended_by": ended_by,
        "active_players_at_end": active_at_end,
        "rounds": game.env.round,
        "decisions": decisions,
        "truncated": not game.env.done,
        "final_net_worth": [float(player.net_worth()) for player in game.env.players],
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_asu_self_play(seeds: tuple[int, ...], max_decisions: int) -> list[dict]:
    """Every seat is asu-value-v1; symmetric, so one game per seed suffices."""
    focus_spec = parse_agent_spec("asu-value-v1")
    specs = (focus_spec,) * NUM_PLAYERS
    factory = AgentFactory()
    games = []
    with preserve_global_rng():
        for seed in seeds:
            games.append(_run_game_with_ending(specs, int(seed), max_decisions, factory))
    return games


def run_asu_vs_fixed(
    seeds: tuple[int, ...],
    opponents: tuple[str, str, str],
    max_decisions: int,
) -> list[dict]:
    """Seat-balanced: asu-value-v1 rotates through all four physical seats."""
    focus_spec = parse_agent_spec("asu-value-v1")
    opponent_specs = tuple(parse_agent_spec(o) for o in opponents)
    factory = AgentFactory()
    games = []
    with preserve_global_rng():
        for seed in seeds:
            for focus_seat in range(NUM_PLAYERS):
                seats: list[AgentSpec | None] = [None] * NUM_PLAYERS
                seats[focus_seat] = focus_spec
                for seat, opponent in zip(
                    (seat for seat in range(NUM_PLAYERS) if seat != focus_seat),
                    opponent_specs,
                ):
                    seats[seat] = opponent
                games.append(
                    _run_game_with_ending(
                        tuple(s for s in seats if s is not None),
                        int(seed),
                        max_decisions,
                        factory,
                    )
                )
    return games


def summarize(games: list[dict]) -> dict:
    total = len(games)
    counts = {"elimination": 0, "round_cap": 0, "decision_cap": 0}
    for game in games:
        counts[game["ended_by"]] += 1
    rounds = [g["rounds"] for g in games]
    return {
        "games": total,
        "ended_by_counts": counts,
        "ended_by_rate": {k: (v / total if total else None) for k, v in counts.items()},
        "mean_rounds": sum(rounds) / total if total else None,
        "min_rounds": min(rounds) if rounds else None,
        "max_rounds": max(rounds) if rounds else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("self-play", "vs-fixed"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--opponents", nargs=3, default=("fixed-a", "fixed-b", "fixed-c"))
    parser.add_argument("--max-decisions", type=int, default=DEFAULT_MAX_DECISIONS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    if args.mode == "self-play":
        games = run_asu_self_play(tuple(args.seeds), args.max_decisions)
    else:
        games = run_asu_vs_fixed(tuple(args.seeds), tuple(args.opponents), args.max_decisions)

    result = {
        "mode": args.mode,
        "opponents": list(args.opponents) if args.mode == "vs-fixed" else None,
        "frozen_spec_hash": FROZEN_SPEC_HASH,
        "seeds": list(args.seeds),
        "games": games,
        "summary": summarize(games),
        "elapsed_seconds": time.perf_counter() - started,
        "disclaimer": RESULTS_DISCLAIMER,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
