"""Closed-form loop for ``oracle-plus-v1``.

No colour is the plan. Each spend is one engine-book step: a deed is 2.5×
list, a completed group reprices every deed in it to 5×, a house books at
``houses * house_price * (1 + 0.5 * houses)``. Buy / auction / build / trade
if that delta beats the cash spent. Blocked groups are discounted because we
cannot build there. This is not deepcopy 1-ply and not a colour ranking.
"""

from __future__ import annotations

from monopoly_game_engine.actions import (
    AUCTION_ACTION_TO_INCREMENT,
    OFFSETS,
    ActionType,
    AuctionAction,
)
from monopoly_game_engine.agents_fixed import (
    _buy_trade_action,
    _exchange_action,
)
from monopoly_game_engine.constants import (
    COLOR_GROUPS,
    PROPERTIES,
    PROPERTY_IDS,
    REAL_ESTATE_IDS,
    TRADE_CASH_LEVELS,
)
from monopoly_game_engine.env import PHASE_AUCTION, MonopolyEnv

from oracle.plus_steals import (
    HOUSE_LO,
    HOTEL_LO,
    MORTGAGE_LO,
    SELL_HOUSE_LO,
    SELL_PROP_LO,
    UNMORTGAGE_LO,
    REAL_COLOURS,
    complete_floor,
    spend_floor,
    would_complete,
)

# Kept for jail-v1, which still ranks a colour. Plus does not use it.
_JAIL_CORRIDOR = frozenset(range(11, 20))
_RAIL = "railroad"
OPENING_RACE = frozenset(("brown", "lightblue", "pink"))
SOLO_BOOK = 2.5
MONO_BOOK = 5.0
BLOCKED_WEIGHT = 0.25
DENY_FRAC = 0.22
AUCTION_FRAC = 0.62
BUILD_RESERVE = 0.25


def _group(color: str) -> list[int]:
    return list(COLOR_GROUPS.get(color) or ())


def colour_is_open(env: MonopolyEnv, pid: int, color: str) -> bool:
    """True if we can still complete ``color`` (no opponent piece in it)."""

    for square in _group(color):
        owner = env.properties[square].owner
        if owner not in (pid, None):
            return False
    return True


def _rent_at_three(color: str) -> float:
    rents = [float((PROPERTIES[sq].get("rent") or [0])[3]) for sq in _group(color)]
    return min(rents) if rents else 0.0


def _cost_to_three(env: MonopolyEnv, pid: int, color: str) -> float:
    group = _group(color)
    house = float(PROPERTIES[group[0]]["house_price"])
    deeds = sum(
        float(PROPERTIES[sq]["price"])
        for sq in group
        if env.properties[sq].owner != pid
    )
    return deeds + house * 3.0 * len(group)


def _owned_count(env: MonopolyEnv, pid: int, color: str) -> int:
    return sum(env.properties[sq].owner == pid for sq in _group(color))


def colour_score(env: MonopolyEnv, pid: int, color: str) -> float:
    """Jail-v1 ranking only. Plus ignores this."""

    if not colour_is_open(env, pid, color):
        return 0.0
    group = _group(color)
    if not group:
        return 0.0
    owned = _owned_count(env, pid, color)
    if owned == len(group):
        return 0.0
    weapon = _rent_at_three(color)
    cost = max(_cost_to_three(env, pid, color), 1.0)
    score = weapon / cost
    if owned == len(group) - 1:
        score *= 4.0
    elif owned >= 1:
        score *= 2.0
    if any(sq in _JAIL_CORRIDOR for sq in group):
        score *= 1.6
    if color in OPENING_RACE:
        score *= 2.5
        if len(group) == 2:
            score *= 2.0
    return score


def active_colour(env: MonopolyEnv, pid: int) -> str | None:
    """Jail-v1's one colour. Plus does not call this."""

    best: tuple[float, str] | None = None
    for color in REAL_COLOURS:
        score = colour_score(env, pid, color)
        if score <= 0.0:
            continue
        if best is None or score > best[0]:
            best = (score, color)
    return None if best is None else best[1]


def _squares(square: int) -> list[int]:
    data = PROPERTIES.get(square)
    if data is None:
        return []
    return _group(data["color"])


def _is_real(square: int) -> bool:
    data = PROPERTIES.get(square)
    return bool(data) and data["color"] not in (_RAIL, "utility")


def _deed_book(square: int, houses: int, mortgaged: bool, monopoly: bool) -> float:
    data = PROPERTIES[square]
    price = float(data["price"])
    mort = float(data["mortgage"]) if mortgaged else 0.0
    base = (price - mort) * (MONO_BOOK if monopoly else SOLO_BOOK)
    if not _is_real(square) or houses <= 0:
        return base
    hp = float(data["house_price"])
    count = 5 if houses >= 5 else houses
    return base + count * hp * (1.0 + 0.5 * houses)


def _group_book(env: MonopolyEnv, owner: int, squares: list[int]) -> float:
    monopoly = bool(squares) and all(env.properties[s].owner == owner for s in squares)
    total = 0.0
    for square in squares:
        prop = env.properties[square]
        if prop.owner != owner:
            continue
        total += _deed_book(square, int(prop.houses), bool(prop.mortgaged), monopoly)
    return total


def acquire_gain(env: MonopolyEnv, pid: int, square: int) -> float:
    """Engine book added by holding ``square``, ignoring the price paid."""

    squares = _squares(square)
    if not squares:
        return 0.0
    before = _group_book(env, pid, squares)
    completes = all(
        item == square or env.properties[item].owner == pid for item in squares
    )
    after = 0.0
    for item in squares:
        prop = env.properties[item]
        if item != square and prop.owner != pid:
            continue
        houses = 0 if item == square else int(prop.houses)
        mortgaged = False if item == square else bool(prop.mortgaged)
        after += _deed_book(item, houses, mortgaged, completes)
    return after - before


def disposal_loss(env: MonopolyEnv, pid: int, square: int) -> float:
    """Engine book lost by giving ``square`` away, including a broken group."""

    squares = _squares(square)
    if not squares:
        return 0.0
    without = 0.0
    for item in squares:
        if item == square:
            continue
        prop = env.properties[item]
        if prop.owner != pid:
            continue
        without += _deed_book(item, int(prop.houses), bool(prop.mortgaged), False)
    return _group_book(env, pid, squares) - without


def _blocked(env: MonopolyEnv, pid: int, square: int) -> bool:
    if not _is_real(square):
        return False
    for item in _squares(square):
        if item == square:
            continue
        owner = env.properties[item].owner
        if owner not in (pid, None):
            return True
    return False


def acquire_value(env: MonopolyEnv, pid: int, square: int, *, deny: bool = True) -> float:
    """Book gain we keep, plus a slice of the best opponent's gain."""

    own = acquire_gain(env, pid, square)
    if _blocked(env, pid, square):
        own *= BLOCKED_WEIGHT
    if not deny:
        return own
    rival = 0.0
    for opp in env.players:
        if opp.player_id == pid or opp.bankrupt:
            continue
        rival = max(rival, acquire_gain(env, opp.player_id, square))
    return own + DENY_FRAC * rival


def house_gain(env: MonopolyEnv, square: int, *, hotel: bool) -> float:
    """Book added by one build step, excluding cash spent."""

    prop = env.properties[square]
    current = int(prop.houses)
    following = 5 if hotel else current + 1
    monopoly = _is_our_monopoly(env, int(prop.owner), square) if prop.owner is not None else False
    return _deed_book(square, following, bool(prop.mortgaged), monopoly) - _deed_book(
        square, current, bool(prop.mortgaged), monopoly
    )


def _is_our_monopoly(env: MonopolyEnv, pid: int, square: int) -> bool:
    squares = _squares(square)
    return bool(squares) and all(env.properties[s].owner == pid for s in squares)


def wanted_deed(env: MonopolyEnv, pid: int, square: int, plan: str | None = None) -> bool:
    """True if holding ``square`` raises our book after the blocked-group haircut."""

    del plan
    return acquire_value(env, pid, square, deny=False) > 0.0


def giveable_deed(
    env: MonopolyEnv,
    pid: int,
    square: int,
    plan: str | None = None,
    *,
    request_score: float = 0.0,
) -> bool:
    """True if we can swap ``square`` without breaking a monopoly."""

    del plan
    prop = env.properties.get(square)
    if prop is None or prop.owner != pid or int(getattr(prop, "houses", 0) or 0):
        return False
    if _is_our_monopoly(env, pid, square):
        return False
    loss = disposal_loss(env, pid, square)
    return request_score + 1e-9 >= loss


def plan_buy_action(
    env: MonopolyEnv,
    pid: int,
    legal: list[int],
    plan: str | None = None,
    *,
    deny: bool = True,
) -> int | None:
    """Buy if the book step is positive after the next-roll cash floor."""

    del plan
    if int(ActionType.BUY_PROPERTY) not in legal:
        return None
    square = int(env.players[pid].position)
    prop = env.properties.get(square)
    if prop is None or prop.owner is not None:
        return None
    price = float(prop.price)
    if acquire_value(env, pid, square, deny=deny) - price <= 0.0:
        return None
    floor = (
        complete_floor(env, pid)
        if would_complete(env, pid, square) or not _blocked(env, pid, square)
        else spend_floor(env, pid)
    )
    if float(env.players[pid].cash) - price < floor:
        return None
    return int(ActionType.BUY_PROPERTY)


def plan_auction_ceiling(
    env: MonopolyEnv,
    pid: int,
    square: int,
    plan: str | None = None,
    *,
    deny: bool = True,
) -> float:
    """Bid up to 0.62 of book value, never through the next-roll floor."""

    del plan
    if PROPERTIES.get(square) is None:
        return 0.0
    value = acquire_value(env, pid, square, deny=deny)
    if value <= 0.0:
        return 0.0
    floor = (
        complete_floor(env, pid)
        if would_complete(env, pid, square) or not _blocked(env, pid, square)
        else spend_floor(env, pid)
    )
    return max(0.0, min(AUCTION_FRAC * value, float(env.players[pid].cash) - floor))


def plan_auction_action(
    env: MonopolyEnv,
    pid: int,
    legal: list[int],
    plan: str | None = None,
    *,
    deny: bool = True,
) -> int | None:
    if env.phase != PHASE_AUCTION:
        return None
    bids = [a for a in legal if a != int(AuctionAction.PASS)]
    if not bids:
        return int(AuctionAction.PASS) if int(AuctionAction.PASS) in legal else None
    square = getattr(env, "auction_property_id", None)
    if square is None:
        return None
    ceiling = plan_auction_ceiling(env, pid, int(square), plan, deny=deny)
    high = float(getattr(env, "auction_high_bid", 0) or 0)
    affordable: list[tuple[int, int]] = []
    for action, step in AUCTION_ACTION_TO_INCREMENT.items():
        if int(action) in bids and high + step <= ceiling:
            affordable.append((step, int(action)))
    if not affordable:
        return int(AuctionAction.PASS) if int(AuctionAction.PASS) in legal else None
    affordable.sort()
    return affordable[0][1]


def plan_build_action(
    env: MonopolyEnv, pid: int, legal: list[int], plan: str | None = None
) -> int | None:
    """Build the step with the most book per dollar, down to 0.25 of next-roll."""

    del plan
    floor = BUILD_RESERVE * spend_floor(env, pid)
    cash = float(env.players[pid].cash)
    best: tuple[float, float, int] | None = None
    for action in legal:
        if HOUSE_LO <= action < HOTEL_LO:
            square = REAL_ESTATE_IDS[action - HOUSE_LO]
            cost = float(PROPERTIES[square]["house_price"])
            gain = house_gain(env, square, hotel=False) - cost
        elif HOTEL_LO <= action < SELL_HOUSE_LO:
            square = REAL_ESTATE_IDS[action - HOTEL_LO]
            cost = float(PROPERTIES[square]["house_price"])
            gain = house_gain(env, square, hotel=True) - cost
        else:
            continue
        if gain <= 0.0 or cash - cost < floor:
            continue
        key = (gain / max(cost, 1.0), -cost, action)
        if best is None or key > (best[0], best[1], best[2]):
            best = key
    return None if best is None else int(best[2])


def debt_action(env: MonopolyEnv, pid: int, legal: list[int], plan: str | None = None) -> int:
    """Mortgage blocked colours first. Never break a set while junk remains."""

    del plan
    if int(ActionType.DECLARE_BANKRUPT) in legal and len(legal) == 1:
        return int(ActionType.DECLARE_BANKRUPT)

    def _mortgage_rank(action: int) -> tuple[int, float] | None:
        if not (MORTGAGE_LO <= action < UNMORTGAGE_LO):
            return None
        square = PROPERTY_IDS[action - MORTGAGE_LO]
        prop = env.properties[square]
        if prop.houses:
            return None
        if _is_our_monopoly(env, pid, square):
            return None
        if not _is_real(square):
            return (2, -float(prop.mortgage_v))
        if _blocked(env, pid, square):
            return (0, -float(prop.mortgage_v))
        return (1, -float(prop.mortgage_v))

    ranked = []
    for action in legal:
        key = _mortgage_rank(action)
        if key is not None:
            ranked.append((key, action))
    if ranked:
        ranked.sort()
        return int(ranked[0][1])

    houses = [a for a in legal if SELL_HOUSE_LO <= a < SELL_PROP_LO]
    if houses:
        return int(min(houses))
    sells = [a for a in legal if SELL_PROP_LO <= a < OFFSETS["buy_trade"]]
    if sells:
        return int(min(sells))
    return int(min(legal))


def needs_plan_cash(
    env: MonopolyEnv, pid: int, legal: list[int], plan: str | None = None, *, deny: bool = True
) -> bool:
    cash = float(env.players[pid].cash)
    if cash < max(200.0, 1.5 * spend_floor(env, pid)):
        return True
    if int(ActionType.BUY_PROPERTY) not in legal:
        return False
    square = int(env.players[pid].position)
    prop = env.properties.get(square)
    if prop is None or prop.owner is not None:
        return False
    price = float(prop.price)
    if acquire_value(env, pid, square, deny=deny) - price <= 0.0:
        return False
    return cash - price < complete_floor(env, pid)


def _trade_gain(
    env: MonopolyEnv, pid: int, give_sq: int | None, take_sq: int | None, cash_delta: float
) -> float:
    gain = cash_delta
    if take_sq is not None:
        gain += acquire_value(env, pid, take_sq, deny=False)
    if give_sq is not None:
        gain -= disposal_loss(env, pid, give_sq)
    return gain


def plan_incoming_action(
    env: MonopolyEnv, pid: int, legal: list[int], plan: str | None = None
) -> int | None:
    """Accept only if we gain book and we gain more than the proposer."""

    del plan
    accept = int(ActionType.ACCEPT_TRADE)
    decline = int(ActionType.DECLINE_TRADE)
    if accept not in legal and decline not in legal:
        return None
    offer = env._incoming_trade(pid)
    if offer is None:
        return None
    requested = offer.requested_prop
    offered = offer.offered_prop
    req_sq = None if requested is None else int(requested.square_id)
    off_sq = None if offered is None else int(offered.square_id)
    cash_in = float(offer.cash_offered or 0) - float(offer.cash_requested or 0)
    cost = float(offer.cash_requested or 0)

    if req_sq is not None and would_complete(env, offer.from_player, req_sq):
        return decline if decline in legal else None
    if req_sq is not None and _is_our_monopoly(env, pid, req_sq):
        return decline if decline in legal else None

    mine = _trade_gain(env, pid, req_sq, off_sq, cash_in)
    theirs = _trade_gain(
        env,
        offer.from_player,
        off_sq,
        req_sq,
        -cash_in,
    )
    floor = complete_floor(env, pid) if off_sq is not None and would_complete(env, pid, off_sq) else spend_floor(env, pid)
    if mine > 0.0 and mine > theirs and float(env.players[pid].cash) - cost >= floor:
        return accept if accept in legal else None
    return decline if decline in legal else None


def plan_trade_action(
    env: MonopolyEnv,
    pid: int,
    legal: list[int],
    plan: str | None = None,
    *,
    pay_up: bool = True,
) -> int | None:
    """Swap when both sides gain book and we gain more. Cash-buy only a finish."""

    del plan
    del pay_up
    if pid in getattr(env, "pending_trades", {}):
        return None
    me = env.players[pid]
    ours = [
        int(prop.square_id)
        for prop in me.properties
        if int(getattr(prop, "houses", 0) or 0) == 0
    ]
    best_exch: tuple[tuple[float, float], int] | None = None
    for opp in env.players:
        if opp.player_id == pid or opp.bankrupt:
            continue
        theirs = [
            int(prop.square_id)
            for prop in opp.properties
            if int(getattr(prop, "houses", 0) or 0) == 0
        ]
        for req in theirs:
            for offer_sq in ours:
                if _is_our_monopoly(env, pid, offer_sq):
                    continue
                mine = _trade_gain(env, pid, offer_sq, req, 0.0)
                theirs_gain = _trade_gain(env, opp.player_id, req, offer_sq, 0.0)
                if mine <= 0.0 or theirs_gain <= 0.0 or mine <= theirs_gain:
                    continue
                action = _exchange_action(
                    pid, opp.player_id, offer_sq, req, env, legal
                )
                if action is None:
                    continue
                key = (1.0 if would_complete(env, pid, req) else 0.0, mine)
                if best_exch is None or key > best_exch[0]:
                    best_exch = (key, int(action))
    if best_exch is not None:
        return best_exch[1]

    best_buy: tuple[tuple[float, float], int] | None = None
    price_idx = TRADE_CASH_LEVELS.index(1.0) if 1.0 in TRADE_CASH_LEVELS else 1
    for opp in env.players:
        if opp.player_id == pid or opp.bankrupt:
            continue
        for req in (
            int(prop.square_id)
            for prop in opp.properties
            if int(getattr(prop, "houses", 0) or 0) == 0
        ):
            if not would_complete(env, pid, req):
                continue
            cost = float(PROPERTIES[req]["price"]) * TRADE_CASH_LEVELS[price_idx]
            if float(me.cash) - cost < complete_floor(env, pid):
                continue
            mine = acquire_value(env, pid, req, deny=False) - cost
            if mine <= 0.0:
                continue
            action = _buy_trade_action(pid, opp.player_id, req, price_idx, env, legal)
            if action is None:
                continue
            key = (mine, -cost)
            if best_buy is None or key > best_buy[0]:
                best_buy = (key, int(action))
    if best_buy is not None:
        return best_buy[1]
    return None


def idle_action(legal: list[int]) -> int:
    if int(AuctionAction.PASS) in legal:
        return int(AuctionAction.PASS)
    if int(ActionType.ROLL_DICE) in legal:
        return int(ActionType.ROLL_DICE)
    if int(ActionType.END_TURN) in legal:
        return int(ActionType.END_TURN)
    return int(legal[0])


__all__ = [
    "AUCTION_FRAC",
    "OPENING_RACE",
    "acquire_gain",
    "acquire_value",
    "active_colour",
    "colour_is_open",
    "colour_score",
    "debt_action",
    "giveable_deed",
    "idle_action",
    "needs_plan_cash",
    "plan_auction_action",
    "plan_auction_ceiling",
    "plan_build_action",
    "plan_buy_action",
    "plan_incoming_action",
    "plan_trade_action",
    "wanted_deed",
]
