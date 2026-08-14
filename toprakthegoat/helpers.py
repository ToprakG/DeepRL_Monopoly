"""Goat helpers that used to live under ``oracle.plus_*``.

Copied here so the submitted agent does not import the oracle package (that
package init pulls torch via monopoly_bench).
"""

from __future__ import annotations

from monopoly_game_engine.actions import OFFSETS, ActionType, AuctionAction
from monopoly_game_engine.constants import (
    COLOR_GROUPS,
    JAIL_BAIL,
    PROPERTIES,
    PROPERTY_IDS,
)
from monopoly_game_engine.env import MonopolyEnv

HOUSE_LO = OFFSETS["improve_house"]
HOTEL_LO = OFFSETS["improve_hotel"]
SELL_HOUSE_LO = OFFSETS["sell_house"]
SELL_PROP_LO = OFFSETS["sell_prop"]
MORTGAGE_LO = OFFSETS["mortgage"]
UNMORTGAGE_LO = OFFSETS["unmortgage"]
REAL_COLOURS = tuple(
    color for color in COLOR_GROUPS if color not in ("railroad", "utility")
)


def colour_is_open(env: MonopolyEnv, pid: int, color: str) -> bool:
    for square in COLOR_GROUPS.get(color) or ():
        owner = env.properties[square].owner
        if owner not in (pid, None):
            return False
    return True


def idle_action(legal: list[int]) -> int:
    if int(AuctionAction.PASS) in legal:
        return int(AuctionAction.PASS)
    if int(ActionType.ROLL_DICE) in legal:
        return int(ActionType.ROLL_DICE)
    if int(ActionType.END_TURN) in legal:
        return int(ActionType.END_TURN)
    return int(legal[0])


def max_live_rent(env: MonopolyEnv, pid: int) -> float:
    worst = 0.0
    for opp in env.players:
        if opp.player_id == pid or opp.bankrupt:
            continue
        rails = sum(1 for p in opp.properties if p.color == "railroad" and not p.mortgaged)
        utils = sum(1 for p in opp.properties if p.color == "utility" and not p.mortgaged)
        for prop in opp.properties:
            rent = float(prop.get_rent(7, max(rails, 1), max(utils, 1)))
            if rent > worst:
                worst = rent
    return worst


def next_roll_threat(env: MonopolyEnv, pid: int) -> float:
    player = env.players[pid]
    pos = int(player.position)
    worst = 0.0
    for total in range(2, 13):
        square = (pos + total) % 40
        prop = env.properties.get(square)
        if prop is None or prop.owner in (None, pid) or prop.mortgaged:
            continue
        opp = env.players[prop.owner]
        if opp.bankrupt:
            continue
        rails = sum(1 for p in opp.properties if p.color == "railroad" and not p.mortgaged)
        utils = sum(1 for p in opp.properties if p.color == "utility" and not p.mortgaged)
        rent = float(prop.get_rent(7, max(rails, 1), max(utils, 1)))
        if rent > worst:
            worst = rent
    return worst


def spend_floor(env: MonopolyEnv, pid: int) -> float:
    return max(float(JAIL_BAIL), next_roll_threat(env, pid))


def complete_floor(env: MonopolyEnv, pid: int) -> float:
    return float(next_roll_threat(env, pid))


def opponent_is_developed(env: MonopolyEnv, pid: int) -> bool:
    for opp in env.players:
        if opp.player_id == pid or opp.bankrupt:
            continue
        for prop in opp.properties:
            if int(getattr(prop, "houses", 0) or 0) >= 3:
                return True
    return False


def unowned_count(env: MonopolyEnv) -> int:
    return sum(1 for prop in env.properties.values() if prop.owner is None)


def would_complete(env: MonopolyEnv, pid: int, square: int) -> bool:
    data = PROPERTIES.get(square)
    if data is None:
        return False
    group = COLOR_GROUPS.get(data["color"]) or ()
    if not group:
        return False
    return all(env.properties[sq].owner == pid or sq == square for sq in group)


def lethal_jail_action(env: MonopolyEnv, pid: int, legal: list[int]) -> int | None:
    if int(ActionType.PAY_BAIL) not in legal:
        return None
    if not opponent_is_developed(env, pid):
        return None
    if int(ActionType.USE_GOOJ_CARD) in legal:
        return int(ActionType.USE_GOOJ_CARD)
    if int(ActionType.ROLL_DICE) in legal:
        return int(ActionType.ROLL_DICE)
    if int(ActionType.END_TURN) in legal:
        return int(ActionType.END_TURN)
    return None


def thaw_unmortgage_action(env: MonopolyEnv, pid: int, legal: list[int]) -> int | None:
    floor = spend_floor(env, pid)
    cash = float(env.players[pid].cash)
    best: tuple[float, int] | None = None
    for action in legal:
        if not (UNMORTGAGE_LO <= action < HOUSE_LO):
            continue
        square = PROPERTY_IDS[action - UNMORTGAGE_LO]
        prop = env.properties[square]
        color = prop.color
        if color not in REAL_COLOURS:
            continue
        group = COLOR_GROUPS[color]
        if not all(env.properties[sq].owner == pid for sq in group):
            continue
        cost = float(prop.mortgage_v) * 1.1
        if cash - cost < floor:
            continue
        if best is None or cost < best[0]:
            best = (cost, int(action))
    return None if best is None else best[1]
