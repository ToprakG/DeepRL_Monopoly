"""EnzeCbe hybrid-PPO heuristics (github.com/EnzeCbe/monopoly-boom).

The published repo has no usable trained checkpoint for the leftover
roll/end-turn head (their documented 2k-game run peaked at 2.5% WR).
ALGORITHM.md says the 9 heuristics are the policy; this agent runs those
in the same priority order, then falls back to roll / end-turn.
"""

from __future__ import annotations

from typing import List, Optional

from monopoly_game_engine.actions import (
    AUCTION_ACTION_TO_INCREMENT,
    OFFSETS,
    ActionType,
    AuctionAction,
)
from monopoly_game_engine.agents_fixed import (
    _buy_trade_action,
    _exchange_action,
    _sell_trade_action,
)
from monopoly_game_engine.constants import (
    COLOR_GROUPS,
    NUM_PLAYERS,
    PROPERTY_IDS,
    REAL_ESTATE_IDS,
)


def fixed_buy_decision(env, pid: int) -> bool:
    player = env.players[pid]
    sq = player.position
    if sq not in env.properties:
        return False
    prop = env.properties[sq]
    if prop.owner is not None or not player.can_afford(prop.price):
        return False

    color = prop.color
    group = COLOR_GROUPS.get(color, [])
    if group:
        owned = sum(1 for s in group if env.properties[s].owner == pid)
        if owned + 1 == len(group):
            return True

    buffer = 20 if color == "orange" else 100
    return player.cash >= prop.price + buffer


def fixed_accept_trade_decision(env, pid: int) -> bool:
    offer = env._incoming_trade(pid)
    if offer is None:
        return False

    if offer.offered_prop:
        color = offer.offered_prop.color
        group = COLOR_GROUPS.get(color, [])
        if group:
            owned_after = sum(
                1
                for s in group
                if env.properties[s].owner == pid
                or env.properties[s] == offer.offered_prop
            )
            if owned_after == len(group):
                return True

    nwo = offer.net_worth()
    if nwo < 0:
        return False

    player = env.players[pid]
    if player.cash - offer.cash_requested < 100:
        return False

    return True


def fixed_build_decision(env, pid: int, allowed) -> Optional[int]:
    player = env.players[pid]
    build_floor = 100
    for i, sq in enumerate(REAL_ESTATE_IDS):
        prop = env.properties[sq]
        if prop.owner != pid or not prop.is_monopoly:
            continue
        house_price = prop.data["house_price"]
        if player.cash < house_price + build_floor:
            continue
        hotel_action = OFFSETS["improve_hotel"] + i
        house_action = OFFSETS["improve_house"] + i
        if hotel_action in allowed:
            return hotel_action
        if house_action in allowed:
            return house_action
    return None


def fixed_auction_decision(env, pid: int, allowed) -> Optional[int]:
    prop = env.properties.get(env.auction_property_id)
    player = env.players[pid]
    if prop is None:
        return None

    color = prop.color
    group = COLOR_GROUPS.get(color, [])
    completes_monopoly = bool(group) and (
        sum(1 for s in group if env.properties[s].owner == pid) + 1 == len(group)
    )
    ceiling = prop.price * 1.75 if completes_monopoly else prop.price * 0.9

    safety_buffer = 100
    max_bid = min(ceiling, player.cash - safety_buffer)

    if env.auction_high_bid >= max_bid:
        return int(AuctionAction.PASS)

    candidates = sorted(
        (
            (action, increment)
            for action, increment in AUCTION_ACTION_TO_INCREMENT.items()
            if int(action) in allowed and env.auction_high_bid + increment <= max_bid
        ),
        key=lambda pair: pair[1],
    )
    if not candidates:
        return int(AuctionAction.PASS)
    return int(candidates[0][0])


def fixed_mortgage_decision(env, pid: int, allowed) -> Optional[int]:
    player = env.players[pid]
    if player.cash >= 200:
        return None
    candidates = sorted(
        (
            (sq, env.properties[sq])
            for sq in PROPERTY_IDS
            if env.properties[sq].owner == pid
            and not env.properties[sq].mortgaged
            and not env.properties[sq].is_monopoly
            and env.properties[sq].houses == 0
        ),
        key=lambda pair: pair[1].price,
    )
    for sq, prop in candidates:
        idx = PROPERTY_IDS.index(sq)
        action = OFFSETS["mortgage"] + idx
        if action in allowed:
            return action
    return None


def fixed_unmortgage_decision(env, pid: int, allowed) -> Optional[int]:
    player = env.players[pid]
    if player.cash < 500:
        return None
    candidates = sorted(
        (
            (sq, env.properties[sq])
            for sq in PROPERTY_IDS
            if env.properties[sq].owner == pid and env.properties[sq].mortgaged
        ),
        key=lambda pair: pair[1].mortgage_v,
    )
    for sq, prop in candidates:
        cost = int(prop.mortgage_v * 1.1)
        if player.cash - cost < 300:
            continue
        idx = PROPERTY_IDS.index(sq)
        action = OFFSETS["unmortgage"] + idx
        if action in allowed:
            return action
    return None


def fixed_liquidation_decision(env, pid: int, allowed) -> Optional[int]:
    if env.debt_player != pid and env.players[pid].cash >= 50:
        return None
    for i, sq in enumerate(REAL_ESTATE_IDS):
        prop = env.properties[sq]
        if prop.owner != pid:
            continue
        action = OFFSETS["sell_hotel"] + i
        if action in allowed:
            return action
    cheapest_house = sorted(
        (
            (i, sq)
            for i, sq in enumerate(REAL_ESTATE_IDS)
            if env.properties[sq].owner == pid and env.properties[sq].houses > 0
        ),
        key=lambda pair: env.properties[pair[1]].price,
    )
    for i, sq in cheapest_house:
        action = OFFSETS["sell_house"] + i
        if action in allowed:
            return action

    cheapest_prop = sorted(
        (
            (idx, sq)
            for idx, sq in enumerate(PROPERTY_IDS)
            if env.properties[sq].owner == pid
        ),
        key=lambda pair: env.properties[pair[1]].price,
    )
    for idx, sq in cheapest_prop:
        action = OFFSETS["sell_prop"] + idx
        if action in allowed:
            return action
    return None


def fixed_jail_decision(env, pid: int, allowed) -> Optional[int]:
    if int(ActionType.USE_GOOJ_CARD) in allowed:
        return int(ActionType.USE_GOOJ_CARD)
    return None


def fixed_trade_offer_decision(env, pid: int, allowed) -> Optional[int]:
    others = [
        i for i in range(NUM_PLAYERS) if i != pid and not env.players[i].bankrupt
    ]
    if not others:
        return None

    for color, group in COLOR_GROUPS.items():
        if color in ("railroad", "utility"):
            continue
        owned = [s for s in group if env.properties[s].owner == pid]
        need = [
            s
            for s in group
            if env.properties[s].owner not in (pid, None)
            and not env.players[env.properties[s].owner].bankrupt
        ]
        if len(owned) + 1 == len(group) and need:
            sq = need[0]
            target = env.properties[sq].owner
            action = _buy_trade_action(pid, target, sq, 0, env, allowed)
            if action is not None:
                return action

    for color, group in COLOR_GROUPS.items():
        if color in ("railroad", "utility"):
            continue
        owned_here = [s for s in group if env.properties[s].owner == pid]
        if not owned_here:
            continue
        need = [
            s
            for s in group
            if env.properties[s].owner not in (pid, None)
            and not env.players[env.properties[s].owner].bankrupt
        ]
        for req_sq in need:
            target = env.properties[req_sq].owner
            for offer_sq in owned_here:
                if env.properties[offer_sq].is_monopoly:
                    continue
                action = _exchange_action(pid, target, offer_sq, req_sq, env, allowed)
                if action is not None:
                    return action

    for sq in PROPERTY_IDS:
        prop = env.properties[sq]
        if prop.owner != pid or prop.is_monopoly or prop.houses > 0:
            continue
        for target in others:
            action = _sell_trade_action(pid, target, sq, 2, env, allowed)
            if action is not None:
                return action

    return None


def _is_heuristic_family(action: int) -> bool:
    if action in (
        int(ActionType.BUY_PROPERTY),
        int(ActionType.ACCEPT_TRADE),
        int(ActionType.PAY_BAIL),
        int(ActionType.USE_GOOJ_CARD),
    ):
        return True
    if OFFSETS["improve_house"] <= action < OFFSETS["sell_house"]:
        return True
    if OFFSETS["buy_trade"] <= action < OFFSETS["auction"]:
        return True
    if OFFSETS["auction"] <= action < OFFSETS["auction"] + 5:
        return True
    if OFFSETS["mortgage"] <= action < OFFSETS["improve_house"]:
        return True
    if OFFSETS["sell_house"] <= action < OFFSETS["buy_trade"]:
        return True
    return False


class BoomHybridAgent:
    """Nine EnzeCbe heuristics + roll/end-turn leftover, H2H-compatible."""

    policy_id = "boom-hybrid"

    def __init__(self, player_id: int):
        self.player_id = player_id

    def choose_action(self, env) -> int:
        pid = self.player_id
        allowed: List[int] = list(env.get_allowed_actions(pid))
        if not allowed:
            return int(ActionType.END_TURN)
        if len(allowed) == 1:
            return int(allowed[0])

        if int(ActionType.BUY_PROPERTY) in allowed and fixed_buy_decision(env, pid):
            return int(ActionType.BUY_PROPERTY)

        if int(ActionType.ACCEPT_TRADE) in allowed:
            pending = next(
                (o for o in env.pending_trades.values() if o.to_player == pid),
                None,
            )
            if pending is not None:
                if fixed_accept_trade_decision(env, pid):
                    return int(ActionType.ACCEPT_TRADE)
                return int(ActionType.DECLINE_TRADE)

        build_action = fixed_build_decision(env, pid, allowed)
        if build_action is not None:
            return int(build_action)

        offer_action = fixed_trade_offer_decision(env, pid, allowed)
        if offer_action is not None:
            return int(offer_action)

        jail_action = fixed_jail_decision(env, pid, allowed)
        if jail_action is not None:
            return int(jail_action)

        auction_action = fixed_auction_decision(env, pid, allowed)
        if auction_action is not None:
            return int(auction_action)

        mort_action = fixed_mortgage_decision(env, pid, allowed)
        if mort_action is not None:
            return int(mort_action)
        unmort_action = fixed_unmortgage_decision(env, pid, allowed)
        if unmort_action is not None:
            return int(unmort_action)

        liq_action = fixed_liquidation_decision(env, pid, allowed)
        if liq_action is not None:
            return int(liq_action)

        leftover = [a for a in allowed if not _is_heuristic_family(a)]
        if not leftover:
            leftover = allowed
        if int(ActionType.ROLL_DICE) in leftover:
            return int(ActionType.ROLL_DICE)
        if int(ActionType.END_TURN) in leftover:
            return int(ActionType.END_TURN)
        if int(ActionType.DO_NOTHING) in leftover:
            return int(ActionType.DO_NOTHING)
        return int(leftover[0])


__all__ = ["BoomHybridAgent"]
