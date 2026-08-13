"""One-weapon plan loop for ``oracle-plus-v1``.

Not a net-worth argmax over every legal action, and not a stack of stolen
priority rules. Each decision answers: which colour can we still finish, and
does this action serve that colour?

The ranking is board geometry we already use elsewhere (jail-exit corridor
11–19, 3-house rent vs remaining cash to get there). Dead colours are fuel.
Completing auctions spend down to the next-roll floor — no fraction cap.
"""

from __future__ import annotations

from monopoly_game_engine.actions import (
    AUCTION_ACTION_TO_INCREMENT,
    OFFSETS,
    ActionType,
    AuctionAction,
)
from monopoly_game_engine.constants import COLOR_GROUPS, PROPERTIES, PROPERTY_IDS, REAL_ESTATE_IDS
from monopoly_game_engine.env import PHASE_AUCTION, MonopolyEnv

from oracle.plus_steals import (
    HOUSE_LO,
    HOTEL_LO,
    MORTGAGE_LO,
    SELL_HOUSE_LO,
    SELL_PROP_LO,
    UNMORTGAGE_LO,
    REAL_COLOURS,
    cap_weight,
    complete_floor,
    spend_floor,
    unowned_count,
    would_complete,
)

# Squares a player leaving jail actually rolls onto (6–9 from square 10).
_JAIL_CORRIDOR = frozenset(range(11, 20))
_RAIL = "railroad"


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
    """Higher = better single weapon. One-away and jail-exit beat empty dark blue."""

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
    return score


def active_colour(env: MonopolyEnv, pid: int) -> str | None:
    """The one real-estate colour we are trying to finish. None if all blocked."""

    best: tuple[float, str] | None = None
    for color in REAL_COLOURS:
        score = colour_score(env, pid, color)
        if score <= 0.0:
            continue
        if best is None or score > best[0]:
            best = (score, color)
    return None if best is None else best[1]


def _opponent_one_away(env: MonopolyEnv, pid: int, square: int) -> bool:
    for opp in env.players:
        if opp.player_id == pid or opp.bankrupt:
            continue
        if would_complete(env, opp.player_id, square):
            return True
    return False


def _rails_owned(env: MonopolyEnv, pid: int) -> int:
    return sum(
        env.properties[sq].owner == pid for sq in (COLOR_GROUPS.get(_RAIL) or ())
    )


def _deed_role(
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
    if color == _RAIL and _rails_owned(env, pid) >= 1:
        return "rail"
    if deny and _opponent_one_away(env, pid, square):
        return "deny"
    return "junk"


def plan_buy_action(
    env: MonopolyEnv,
    pid: int,
    legal: list[int],
    plan: str | None,
    *,
    deny: bool = True,
) -> int | None:
    """Buy the landed deed only if it is the weapon, a finish, a rail, or a deny."""

    if int(ActionType.BUY_PROPERTY) not in legal:
        return None
    square = int(env.players[pid].position)
    prop = env.properties.get(square)
    if prop is None or prop.owner is not None:
        return None
    role = _deed_role(env, pid, square, plan, deny=deny)
    if role == "junk":
        return None
    price = float(prop.price)
    floor = complete_floor(env, pid) if role in ("finish", "plan") else spend_floor(env, pid)
    if float(env.players[pid].cash) - price < floor:
        return None
    return int(ActionType.BUY_PROPERTY)


def plan_auction_ceiling(
    env: MonopolyEnv,
    pid: int,
    square: int,
    plan: str | None,
    *,
    deny: bool = True,
) -> float:
    """Cash we will spend. Finish/plan go to the next-roll floor (no 0.62 cap)."""

    data = PROPERTIES.get(square)
    if data is None:
        return 0.0
    price = float(data["price"])
    cash = float(env.players[pid].cash)
    role = _deed_role(env, pid, square, plan, deny=deny)
    if role in ("finish", "plan"):
        return max(0.0, cash - complete_floor(env, pid))
    if role == "deny":
        return max(0.0, min(1.15 * price, cash - spend_floor(env, pid)))
    if role == "rail":
        return max(0.0, min(price, cash - spend_floor(env, pid)))
    if cap_weight(env) >= 0.5 or unowned_count(env) <= 2:
        return max(0.0, min(1.25 * price, cash - spend_floor(env, pid)))
    return 0.0


def plan_auction_action(
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
    env: MonopolyEnv, pid: int, legal: list[int], plan: str | None
) -> int | None:
    """Even-build the weapon first (3-house rent / house price), else any set."""

    floor = spend_floor(env, pid)
    cash = float(env.players[pid].cash)
    planned: list[tuple[int, float, int]] = []
    other: list[tuple[int, float, int]] = []
    for action in legal:
        if HOUSE_LO <= action < HOTEL_LO:
            square = REAL_ESTATE_IDS[action - HOUSE_LO]
            cost = float(PROPERTIES[square]["house_price"])
        elif HOTEL_LO <= action < SELL_HOUSE_LO:
            square = REAL_ESTATE_IDS[action - HOTEL_LO]
            cost = float(PROPERTIES[square]["house_price"])
        else:
            continue
        if cash - cost < floor:
            continue
        prop = env.properties[square]
        houses = int(prop.houses)
        rents = PROPERTIES[square].get("rent") or [0]
        three = float(rents[3] if len(rents) > 3 else rents[-1])
        key = (houses, -(three / max(cost, 1.0)), action)
        color = prop.color
        if plan is not None and color == plan:
            planned.append(key)
        else:
            other.append(key)
    pool = planned or other
    if not pool:
        return None
    pool.sort()
    return int(pool[0][2])


def debt_action(env: MonopolyEnv, pid: int, legal: list[int], plan: str | None) -> int:
    """Raise cash from dead colours first. Never break the weapon while junk remains."""

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
        blocked = any(
            env.properties[sq].owner not in (pid, None) for sq in _group(color)
        )
        if blocked:
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
    env: MonopolyEnv, pid: int, legal: list[int], plan: str | None, *, deny: bool = True
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
    if _deed_role(env, pid, square, plan, deny=deny) == "junk":
        return False
    return cash - float(prop.price) < complete_floor(env, pid)


def idle_action(legal: list[int]) -> int:
    if int(AuctionAction.PASS) in legal:
        return int(AuctionAction.PASS)
    if int(ActionType.ROLL_DICE) in legal:
        return int(ActionType.ROLL_DICE)
    if int(ActionType.END_TURN) in legal:
        return int(ActionType.END_TURN)
    return int(legal[0])


__all__ = [
    "active_colour",
    "colour_is_open",
    "colour_score",
    "debt_action",
    "idle_action",
    "needs_plan_cash",
    "plan_auction_action",
    "plan_auction_ceiling",
    "plan_build_action",
    "plan_buy_action",
]
