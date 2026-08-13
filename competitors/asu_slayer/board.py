"""Exact short-horizon board mathematics for ``ppo-plus-v2``.

This module is self-contained: it duplicates no policy logic and imports only
the engine's constants, so the slayer package never depends on a teacher.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

from monopoly_game_engine.constants import (
    COLOR_GROUPS,
    GO_TO_JAIL_SQUARE,
    JAIL_SQUARE,
    MAX_JAIL_TURNS,
    PROPERTIES,
)


_DIE_P = 1.0 / 36.0
_TURN_STATE = tuple[int, bool, int]
_LANDING = tuple[int, int]


def _landing_state(square: int) -> _TURN_STATE:
    if square == GO_TO_JAIL_SQUARE:
        return (JAIL_SQUARE, True, 0)
    return (square, False, 0)


@lru_cache(maxsize=None)
def _free_turn(position: int, doubles: int) -> tuple[dict, dict]:
    """One complete free turn, following the engine's doubles chain."""

    transitions: dict[_TURN_STATE, float] = defaultdict(float)
    landings: dict[_LANDING, float] = defaultdict(float)
    for first in range(1, 7):
        for second in range(1, 7):
            total = first + second
            double = first == second
            if double and doubles + 1 >= 3:
                transitions[(JAIL_SQUARE, True, 0)] += _DIE_P
                continue
            square = (position + total) % 40
            landings[(square, total)] += _DIE_P
            state = _landing_state(square)
            if state[1] or not double:
                transitions[state] += _DIE_P
                continue
            later_states, later_landings = _free_turn(square, doubles + 1)
            for key, value in later_states.items():
                transitions[key] += value * _DIE_P
            for key, value in later_landings.items():
                landings[key] += value * _DIE_P
    return dict(transitions), dict(landings)


@lru_cache(maxsize=None)
def _complete_turn(position: int, in_jail: bool, jail_turns: int) -> tuple[dict, dict]:
    if not in_jail:
        return _free_turn(position, 0)

    transitions: dict[_TURN_STATE, float] = defaultdict(float)
    landings: dict[_LANDING, float] = defaultdict(float)
    for first in range(1, 7):
        for second in range(1, 7):
            total = first + second
            if first != second and jail_turns + 1 < MAX_JAIL_TURNS:
                transitions[(JAIL_SQUARE, True, jail_turns + 1)] += _DIE_P
                continue
            square = (position + total) % 40
            landings[(square, total)] += _DIE_P
            transitions[_landing_state(square)] += _DIE_P
    return dict(transitions), dict(landings)


@lru_cache(maxsize=None)
def _landings(
    position: int, in_jail: bool, jail_turns: int, turns: int
) -> tuple[tuple[_LANDING, float], ...]:
    states: dict[_TURN_STATE, float] = {(position, in_jail, jail_turns): 1.0}
    landings: dict[_LANDING, float] = defaultdict(float)
    for _ in range(turns):
        following: dict[_TURN_STATE, float] = defaultdict(float)
        for state, weight in states.items():
            transitions, turn_landings = _complete_turn(*state)
            for key, value in transitions.items():
                following[key] += value * weight
            for key, value in turn_landings.items():
                landings[key] += value * weight
        states = following
    return tuple(landings.items())


def landings(player, turns: int = 1) -> tuple[tuple[_LANDING, float], ...]:
    """Expected landings as ``((square, dice_total), probability)`` pairs."""

    return _landings(
        int(player.position), bool(player.in_jail), int(player.jail_turns), int(turns)
    )


@lru_cache(maxsize=None)
def group_of(square: int) -> tuple[int, ...]:
    """The full color group containing ``square``."""

    return tuple(COLOR_GROUPS[PROPERTIES[square]["color"]])


@lru_cache(maxsize=None)
def is_real_estate(square: int) -> bool:
    return PROPERTIES[square]["color"] not in ("railroad", "utility")


def owned_in_group(env, owner: int, square: int) -> int:
    return sum(env.properties[item].owner == owner for item in group_of(square))


def rent_at(env, square: int, dice_total: int = 7) -> int:
    """Rent owed for landing on ``square`` right now."""

    prop = env.properties.get(square)
    if prop is None or prop.owner is None or prop.mortgaged:
        return 0
    color = prop.color
    if color == "railroad":
        count = owned_in_group(env, prop.owner, square)
        return prop.data["rent"][min(count - 1, 3)]
    if color == "utility":
        count = owned_in_group(env, prop.owner, square)
        return prop.data["rent"][0 if count == 1 else 1] * dice_total
    if prop.houses:
        return prop.data["rent"][min(prop.houses, 5)]
    monopoly = owned_in_group(env, prop.owner, square) == len(group_of(square))
    return prop.data["rent"][0] * (2 if monopoly else 1)


def exposure(env, player_id: int, turns: int = 1) -> tuple[float, float]:
    """Expected and worst-case rent this player owes over ``turns`` turns."""

    expected = 0.0
    worst = 0.0
    for (square, dice_total), probability in landings(env.players[player_id], turns):
        prop = env.properties.get(square)
        if prop is None or prop.owner is None or prop.owner == player_id:
            continue
        rent = float(rent_at(env, square, dice_total))
        expected += probability * rent
        if rent > worst:
            worst = rent
    return expected, worst


def rent_quantile(env, player_id: int, quantile: float, turns: int = 1) -> float:
    """The rent this player survives with probability ``quantile`` next turn.

    A reserve built on the worst reachable rent is far too conservative — that
    square is usually one roll in thirty-six — while a reserve built on the
    mean ignores the single payment that actually causes bankruptcy. The
    quantile of the true landing distribution is the honest middle.
    """

    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be strictly between 0 and 1")
    outcomes: list[tuple[float, float]] = []
    total = 0.0
    for (square, dice_total), probability in landings(env.players[player_id], turns):
        prop = env.properties.get(square)
        rent = (
            0.0
            if prop is None or prop.owner is None or prop.owner == player_id
            else float(rent_at(env, square, dice_total))
        )
        outcomes.append((rent, probability))
        total += probability
    if total <= 0.0:
        return 0.0

    outcomes.sort()
    cumulative = 0.0
    for rent, probability in outcomes:
        cumulative += probability / total
        if cumulative >= quantile:
            return rent
    return outcomes[-1][0]


def income_by_square(env, player_id: int, turns: int = 1) -> dict[int, float]:
    """Expected rent each owned square collects, in one pass over opponents.

    Used to choose which deed to give up when cash must be raised: the cheapest
    deed to mortgage is the one earning the least, not the one with the lowest
    price.
    """

    earned: dict[int, float] = {}
    for opponent in env.players:
        if opponent.player_id == player_id or opponent.bankrupt:
            continue
        for (square, dice_total), probability in landings(opponent, turns):
            prop = env.properties.get(square)
            if prop is None or prop.owner != player_id:
                continue
            earned[square] = earned.get(square, 0.0) + probability * rent_at(
                env, square, dice_total
            )
    return earned


def income(env, player_id: int, turns: int = 1) -> float:
    """Expected rent this player collects from live opponents."""

    total = 0.0
    for opponent in env.players:
        if opponent.player_id == player_id or opponent.bankrupt:
            continue
        for (square, dice_total), probability in landings(opponent, turns):
            prop = env.properties.get(square)
            if prop is not None and prop.owner == player_id:
                total += probability * rent_at(env, square, dice_total)
    return total


__all__ = [
    "exposure",
    "group_of",
    "income",
    "income_by_square",
    "is_real_estate",
    "landings",
    "owned_in_group",
    "rent_at",
    "rent_quantile",
]
