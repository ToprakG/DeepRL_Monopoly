"""Closed-form body for ``oracle-jail-v1``.

Four rules, no 1-ply:
- buy the jail half (light blue / pink / orange / red) and cheap scrap
- 5-9 dice window decides offense vs defense
- even-build to three houses; mortgage before selling houses
- trades that finish us *and* leave them too poor to reach three
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
    _sell_trade_action,
)
from monopoly_game_engine.constants import (
    COLOR_GROUPS,
    PROPERTIES,
    PROPERTY_IDS,
    REAL_ESTATE_IDS,
    TRADE_CASH_LEVELS,
)
from monopoly_game_engine.env import PHASE_AUCTION, MonopolyEnv

from oracle.plus_loop import colour_is_open
from oracle.plus_steals import (
    HOUSE_LO,
    HOTEL_LO,
    MORTGAGE_LO,
    REAL_COLOURS,
    SELL_HOUSE_LO,
    SELL_PROP_LO,
    UNMORTGAGE_LO,
    complete_floor,
    max_live_rent,
    spend_floor,
    unowned_count,
    would_complete,
)

JAIL_COLOURS = frozenset({"lightblue", "pink", "orange", "red"})
GHOST_COLOURS = frozenset({"brown", "yellow", "green", "darkblue"})
THREE_HOUSES = 3
WINDOW = frozenset(range(5, 10))
SCRAP_FRAC = 0.5
DENY_FRAC = 1.15
SQUEEZE_HOUSES = 8
_RAIL = "railroad"
_UTIL = "utility"


def _group(color: str) -> list[int]:
    return list(COLOR_GROUPS.get(color) or ())


def _walk_from(player) -> int:
    if getattr(player, "in_jail", False):
        return 10
    return int(player.position)


def window_squares(pos: int) -> set[int]:
    return {(pos + dist) % 40 for dist in WINDOW}


def in_window(pos: int, squares) -> bool:
    hits = set(squares)
    return any((pos + dist) % 40 in hits for dist in WINDOW)


def hot_for(env: MonopolyEnv, pid: int, squares) -> bool:
    """True if an opponent is 5-9 away from any of ``squares``."""

    hits = set(squares)
    if not hits:
        return False
    for opp in env.players:
        if opp.player_id == pid or opp.bankrupt:
            continue
        if in_window(_walk_from(opp), hits):
            return True
    return False


def threat_squares(env: MonopolyEnv, pid: int) -> list[int]:
    out = []
    for opp in env.players:
        if opp.player_id == pid or opp.bankrupt:
            continue
        for prop in opp.properties:
            if int(getattr(prop, "houses", 0) or 0) >= THREE_HOUSES:
                out.append(int(prop.square_id))
    return out


def we_threatened(env: MonopolyEnv, pid: int) -> bool:
    return in_window(_walk_from(env.players[pid]), threat_squares(env, pid))


def _real_color(color: str | None) -> str | None:
    return color if color in REAL_COLOURS else None


def cost_to_three(env: MonopolyEnv, color: str) -> float:
    group = _group(color)
    if not group:
        return 0.0
    house = PROPERTIES[group[0]].get("house_price")
    if not house:
        return 0.0
    house = float(house)
    missing = 0
    for square in group:
        houses = int(env.properties[square].houses)
        if houses >= 5:
            continue
        missing += max(0, THREE_HOUSES - min(houses, 4))
    return missing * house


def undeveloped_mortgage(
    env: MonopolyEnv, pid: int, owned: set[int] | None = None, *, exclude=()
) -> float:
    blocked = set(exclude)
    if owned is None:
        owned = {int(prop.square_id) for prop in env.players[pid].properties}
    total = 0.0
    for square in owned:
        prop = env.properties[square]
        if prop.mortgaged or int(getattr(prop, "houses", 0) or 0):
            continue
        if prop.color in blocked:
            continue
        total += float(prop.mortgage_v)
    return total


def can_three(
    env: MonopolyEnv,
    pid: int,
    color: str,
    cash: float,
    owned: set[int] | None = None,
) -> bool:
    if _real_color(color) is None:
        return False
    liquid = cash + undeveloped_mortgage(env, pid, owned, exclude=(color,))
    return liquid + 1e-9 >= cost_to_three(env, color)


def our_sets(env: MonopolyEnv, pid: int) -> list[str]:
    out = []
    for color in REAL_COLOURS:
        group = _group(color)
        if group and all(env.properties[sq].owner == pid for sq in group):
            out.append(color)
    return out


def _opponent_one_away(env: MonopolyEnv, pid: int, square: int) -> bool:
    for opp in env.players:
        if opp.player_id == pid or opp.bankrupt:
            continue
        if would_complete(env, opp.player_id, square):
            return True
    return False


def _housing_squeeze(env: MonopolyEnv) -> bool:
    return int(getattr(env, "houses_available", 32) or 0) <= SQUEEZE_HOUSES


def deed_role(
    env: MonopolyEnv, pid: int, square: int, plan: str | None, *, deny: bool = True
) -> str:
    data = PROPERTIES.get(square)
    if data is None:
        return "junk"
    color = data["color"]
    if would_complete(env, pid, square):
        return "finish"
    if plan is not None and color == plan:
        return "plan"
    if deny and _opponent_one_away(env, pid, square):
        return "deny"
    if color == _RAIL:
        return "rail"
    if color in JAIL_COLOURS and colour_is_open(env, pid, color):
        return "jail"
    if color in GHOST_COLOURS or color == _UTIL:
        return "scrap"
    return "junk"


def buy_action(
    env: MonopolyEnv,
    pid: int,
    legal: list[int],
    plan: str | None,
    *,
    deny: bool = True,
) -> int | None:
    """Cash-buy jail-side / plan / deny. Ghost colours wait for a cheap auction."""

    if int(ActionType.BUY_PROPERTY) not in legal:
        return None
    square = int(env.players[pid].position)
    prop = env.properties.get(square)
    if prop is None or prop.owner is not None:
        return None
    role = deed_role(env, pid, square, plan, deny=deny)
    if role not in ("finish", "plan", "deny", "rail", "jail"):
        return None
    floor = complete_floor(env, pid) if role in ("finish", "plan") else spend_floor(env, pid)
    if float(env.players[pid].cash) - float(prop.price) < floor:
        return None
    return int(ActionType.BUY_PROPERTY)


def auction_ceiling(
    env: MonopolyEnv,
    pid: int,
    square: int,
    plan: str | None,
    *,
    deny: bool = True,
) -> float:
    data = PROPERTIES.get(square)
    if data is None:
        return 0.0
    price = float(data["price"])
    cash = float(env.players[pid].cash)
    role = deed_role(env, pid, square, plan, deny=deny)
    if role in ("finish", "plan"):
        return max(0.0, cash - complete_floor(env, pid))
    if role == "deny":
        return max(0.0, min(DENY_FRAC * price, cash - spend_floor(env, pid)))
    if role in ("rail", "jail"):
        return max(0.0, min(price, cash - spend_floor(env, pid)))
    return max(0.0, min(SCRAP_FRAC * price, cash - spend_floor(env, pid)))


def auction_action(
    env: MonopolyEnv,
    pid: int,
    legal: list[int],
    plan: str | None,
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
    ceiling = auction_ceiling(env, pid, int(square), plan, deny=deny)
    high = float(getattr(env, "auction_high_bid", 0) or 0)
    affordable: list[tuple[int, int]] = []
    for action, step in AUCTION_ACTION_TO_INCREMENT.items():
        if int(action) in bids and high + step <= ceiling:
            affordable.append((step, int(action)))
    if not affordable:
        return int(AuctionAction.PASS) if int(AuctionAction.PASS) in legal else None
    affordable.sort()
    return affordable[0][1]


def _build_ok(env: MonopolyEnv, pid: int, square: int, cost: float) -> bool:
    color = env.properties[square].color
    cash_after = float(env.players[pid].cash) - cost
    liquid = cash_after + undeveloped_mortgage(env, pid, exclude=(color,))
    blow = max_live_rent(env, pid)
    group = _group(color)
    incoming = hot_for(env, pid, group)
    if liquid + 1e-9 < blow and not incoming:
        return False
    if we_threatened(env, pid) and not incoming:
        return False
    return True


def build_action(env: MonopolyEnv, pid: int, legal: list[int], plan: str | None) -> int | None:
    """Even-build to three. Fourth house / hotel only in a housing squeeze."""

    cash = float(env.players[pid].cash)
    squeeze = _housing_squeeze(env)
    planned: list[tuple[int, int, float, int]] = []
    other: list[tuple[int, int, float, int]] = []
    for action in legal:
        if HOUSE_LO <= action < HOTEL_LO:
            square = REAL_ESTATE_IDS[action - HOUSE_LO]
            cost = float(PROPERTIES[square]["house_price"])
            houses = int(env.properties[square].houses)
            if houses >= THREE_HOUSES and not squeeze:
                continue
        elif HOTEL_LO <= action < SELL_HOUSE_LO:
            if not squeeze:
                continue
            square = REAL_ESTATE_IDS[action - HOTEL_LO]
            cost = float(PROPERTIES[square]["house_price"])
            houses = int(env.properties[square].houses)
        else:
            continue
        if cash - cost < spend_floor(env, pid):
            continue
        if not _build_ok(env, pid, square, cost):
            continue
        rents = PROPERTIES[square].get("rent") or [0]
        three = float(rents[3] if len(rents) > 3 else rents[-1])
        key = (houses, -(three / max(cost, 1.0)), -cost, action)
        if plan is not None and env.properties[square].color == plan:
            planned.append(key)
        else:
            other.append(key)
    pool = planned or other
    if not pool:
        return None
    pool.sort()
    return int(pool[0][3])


def debt_action(env: MonopolyEnv, pid: int, legal: list[int], plan: str | None) -> int:
    """Mortgage dead colours, then plan, then bank-sell. Houses last."""

    if int(ActionType.DECLARE_BANKRUPT) in legal and len(legal) == 1:
        return int(ActionType.DECLARE_BANKRUPT)

    def _mortgage_rank(action: int) -> tuple[int, float] | None:
        if not (MORTGAGE_LO <= action < UNMORTGAGE_LO):
            return None
        square = PROPERTY_IDS[action - MORTGAGE_LO]
        prop = env.properties[square]
        if prop.houses:
            return None
        color = prop.color
        if color not in REAL_COLOURS:
            return (2, -float(prop.mortgage_v))
        if plan is not None and color == plan:
            return (3, -float(prop.mortgage_v))
        if color in our_sets(env, pid):
            return None
        blocked = any(env.properties[sq].owner not in (pid, None) for sq in _group(color))
        if blocked or color in GHOST_COLOURS:
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

    sells = [a for a in legal if SELL_PROP_LO <= a < OFFSETS["buy_trade"]]
    if sells:
        return int(min(sells))
    houses = [a for a in legal if SELL_HOUSE_LO <= a < SELL_PROP_LO]
    if houses:
        return int(min(houses))
    return int(min(legal))


def needs_three_cash(env: MonopolyEnv, pid: int) -> bool:
    cash = float(env.players[pid].cash)
    for color in our_sets(env, pid):
        if cost_to_three(env, color) <= 0.0:
            continue
        if cash < cost_to_three(env, color) + spend_floor(env, pid):
            return True
    return False


def fund_three_action(env: MonopolyEnv, pid: int, legal: list[int], plan: str | None) -> int | None:
    """Mortgage ghost / blocked deeds to pay for three houses."""

    hungry = needs_three_cash(env, pid)
    if not hungry:
        cash = float(env.players[pid].cash)
        if cash >= max(200.0, 1.5 * spend_floor(env, pid)):
            return None
        if unowned_count(env) == 0:
            return None
    best: tuple[tuple[int, float], int] | None = None
    owned_sets = set(our_sets(env, pid))
    for action in legal:
        if not (MORTGAGE_LO <= action < UNMORTGAGE_LO):
            continue
        square = PROPERTY_IDS[action - MORTGAGE_LO]
        prop = env.properties[square]
        if prop.houses:
            continue
        color = prop.color
        if color in owned_sets or (plan is not None and color == plan):
            continue
        blocked = color not in REAL_COLOURS or any(
            env.properties[sq].owner not in (pid, None) for sq in _group(color)
        )
        ghost = color in GHOST_COLOURS
        if not blocked and not ghost:
            continue
        key = (0 if blocked or ghost else 1, -float(prop.mortgage_v))
        if best is None or key < best[0]:
            best = (key, int(action))
    return None if best is None else best[1]


def _owned_squares(player) -> set[int]:
    return {int(prop.square_id) for prop in player.properties}


def _color_of(square: int) -> str | None:
    data = PROPERTIES.get(square)
    return None if data is None else data["color"]


def _is_our_monopoly(env: MonopolyEnv, pid: int, square: int) -> bool:
    color = _color_of(square)
    if color is None:
        return False
    group = _group(color)
    return bool(group) and all(env.properties[sq].owner == pid for sq in group)


def _they_can_three_after(
    env: MonopolyEnv,
    opp_id: int,
    owned: set[int],
    cash: float,
    complete_color: str | None,
) -> bool:
    if complete_color is None:
        for color in REAL_COLOURS:
            group = _group(color)
            if group and all(sq in owned for sq in group) and can_three(
                env, opp_id, color, cash, owned
            ):
                return True
        return False
    return can_three(env, opp_id, complete_color, cash, owned)


def _third_party_hot(env: MonopolyEnv, pid: int, opp_id: int, squares) -> bool:
    hits = set(squares)
    for other in env.players:
        if other.player_id in (pid, opp_id) or other.bankrupt:
            continue
        if in_window(_walk_from(other), hits):
            return True
    return False


def _hijack_ok(
    env: MonopolyEnv,
    pid: int,
    opp_id: int,
    *,
    our_owned: set[int],
    their_owned: set[int],
    our_cash: float,
    their_cash: float,
    our_color: str | None,
    their_color: str | None,
) -> bool:
    if our_color is None:
        return False
    if not can_three(env, pid, our_color, our_cash, our_owned):
        return False
    if their_color is not None:
        if _they_can_three_after(env, opp_id, their_owned, their_cash, their_color):
            return False
        their_group = _group(their_color)
        our_group = _group(our_color)
        if _third_party_hot(env, pid, opp_id, their_group) and not hot_for(
            env, pid, our_group
        ):
            return False
    elif _they_can_three_after(env, opp_id, their_owned, their_cash, None):
        return False
    return True


def incoming_action(
    env: MonopolyEnv, pid: int, legal: list[int], plan: str | None
) -> int | None:
    """Accept a finish we can three-house. Decline a finish they can three-house."""

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
    cash = float(env.players[pid].cash)
    pay = float(offer.cash_requested or 0)
    get = float(offer.cash_offered or 0)
    opp_id = int(offer.from_player)
    our_owned = _owned_squares(env.players[pid])
    their_owned = _owned_squares(env.players[opp_id])
    if req_sq is not None:
        our_owned.discard(req_sq)
        their_owned.add(req_sq)
    if off_sq is not None:
        our_owned.add(off_sq)
        their_owned.discard(off_sq)
    our_cash = cash - pay + get
    their_cash = float(env.players[opp_id].cash) + pay - get

    if req_sq is not None and _is_our_monopoly(env, pid, req_sq):
        return decline if decline in legal else None

    our_color = (
        _real_color(_color_of(off_sq))
        if off_sq is not None and would_complete(env, pid, off_sq)
        else None
    )
    if our_color is None:
        for color in (*JAIL_COLOURS, *REAL_COLOURS):
            group = _group(color)
            if group and all(sq in our_owned for sq in group):
                our_color = color
                break
    their_color = (
        _real_color(_color_of(req_sq))
        if req_sq is not None and would_complete(env, opp_id, req_sq)
        else None
    )

    if our_color is not None and _hijack_ok(
        env,
        pid,
        opp_id,
        our_owned=our_owned,
        their_owned=their_owned,
        our_cash=our_cash,
        their_cash=their_cash,
        our_color=our_color,
        their_color=their_color,
    ):
        return accept if accept in legal else None

    if their_color is not None:
        return decline if decline in legal else None
    if off_sq is not None and plan is not None and _color_of(off_sq) == plan:
        if our_cash >= spend_floor(env, pid):
            return accept if accept in legal else None
    return decline if decline in legal else None


def trade_action(env: MonopolyEnv, pid: int, legal: list[int], plan: str | None) -> int | None:
    """Finish us. Completing them is allowed only if they cannot reach three houses."""

    if pid in getattr(env, "pending_trades", {}):
        return None
    me = env.players[pid]
    ours = [
        int(prop.square_id)
        for prop in me.properties
        if int(getattr(prop, "houses", 0) or 0) == 0
    ]
    best: tuple[tuple, int] | None = None

    def _consider(
        action: int | None,
        opp_id: int,
        our_owned: set[int],
        their_owned: set[int],
        our_cash: float,
        their_cash: float,
        our_color: str | None,
        their_color: str | None,
        disguise: float,
    ) -> None:
        nonlocal best
        if action is None:
            return
        if not _hijack_ok(
            env,
            pid,
            opp_id,
            our_owned=our_owned,
            their_owned=their_owned,
            our_cash=our_cash,
            their_cash=their_cash,
            our_color=our_color,
            their_color=their_color,
        ):
            return
        hot = 1.0 if our_color and hot_for(env, pid, _group(our_color)) else 0.0
        jail = 1.0 if our_color in JAIL_COLOURS else 0.0
        key = (disguise, hot, jail, -cost_to_three(env, our_color or ""))
        if best is None or key > best[0]:
            best = (key, int(action))

    for opp in env.players:
        if opp.player_id == pid or opp.bankrupt:
            continue
        theirs = [
            int(prop.square_id)
            for prop in opp.properties
            if int(getattr(prop, "houses", 0) or 0) == 0
        ]
        for req in theirs:
            our_color = (
                _real_color(_color_of(req)) if would_complete(env, pid, req) else None
            )
            if our_color is None:
                continue
            for offer_sq in ours:
                they_complete = would_complete(env, opp.player_id, offer_sq)
                their_color = (
                    _real_color(_color_of(offer_sq)) if they_complete else None
                )
                our_owned = set(ours) - {offer_sq} | {req}
                their_owned = set(theirs) - {req} | {offer_sq}
                action = _exchange_action(pid, opp.player_id, offer_sq, req, env, legal)
                _consider(
                    action,
                    opp.player_id,
                    our_owned,
                    their_owned,
                    float(me.cash),
                    float(opp.cash),
                    our_color,
                    their_color,
                    1.0 if they_complete else 2.0,
                )

            for price_idx, frac in enumerate(TRADE_CASH_LEVELS):
                cost = float(PROPERTIES[req]["price"]) * frac
                our_cash = float(me.cash) - cost
                if our_cash < 0:
                    continue
                their_owned = set(theirs) - {req}
                our_owned = set(ours) | {req}
                if _they_can_three_after(
                    env, opp.player_id, their_owned, float(opp.cash) + cost, None
                ):
                    continue
                action = _buy_trade_action(pid, opp.player_id, req, price_idx, env, legal)
                _consider(
                    action,
                    opp.player_id,
                    our_owned,
                    their_owned,
                    our_cash,
                    float(opp.cash) + cost,
                    our_color,
                    None,
                    0.0,
                )
                break

        have = our_sets(env, pid)
        if not have:
            continue
        our_color = next((c for c in have if c in JAIL_COLOURS), have[0])
        for offer_sq in ours:
            if not would_complete(env, opp.player_id, offer_sq):
                continue
            their_color = _real_color(_color_of(offer_sq))
            if their_color is None:
                continue
            for price_idx in (2, 1, 0):
                cash_amt = float(PROPERTIES[offer_sq]["price"]) * TRADE_CASH_LEVELS[price_idx]
                if float(opp.cash) < cash_amt:
                    continue
                our_owned = set(ours) - {offer_sq}
                their_owned = set(theirs) | {offer_sq}
                action = _sell_trade_action(
                    pid, opp.player_id, offer_sq, price_idx, env, legal
                )
                _consider(
                    action,
                    opp.player_id,
                    our_owned,
                    their_owned,
                    float(me.cash) + cash_amt,
                    float(opp.cash) - cash_amt,
                    our_color,
                    their_color,
                    1.0,
                )
                break

    return None if best is None else best[1]


__all__ = [
    "GHOST_COLOURS",
    "JAIL_COLOURS",
    "THREE_HOUSES",
    "WINDOW",
    "auction_action",
    "auction_ceiling",
    "build_action",
    "buy_action",
    "can_three",
    "cost_to_three",
    "debt_action",
    "deed_role",
    "fund_three_action",
    "hot_for",
    "in_window",
    "incoming_action",
    "needs_three_cash",
    "trade_action",
    "we_threatened",
    "window_squares",
]
