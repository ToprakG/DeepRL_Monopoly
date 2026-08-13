"""Original plus-agent rules, from our H2H logs — not competitor code.

On the real table (96 games) we went bankrupt 60 times. Fast wins were
wipeouts after we actually built. Field 1 vs Underdog: mean book $4.8k vs
$14.9k, 19/24 wipeouts. These helpers are ours:

- wander cash floor (live hotel + next-roll); spend floor is next-roll only
- ASU-delta auction (counterfactual our scorer, not their deed formulas)
- cheapest legal raise, not the largest that still fits (we were overpaying)
- build-first ranked by rent-per-dollar, spend-floor (not distant hotels)
- race-buy the first-monopoly colours (brown / lightblue / darkblue) on sight
- lethal-jail: sit only after 3-houses/hotels; rails are not expensive
- incoming-trade: never hand a completing deed, take one that completes us
- dead-colour mortgage: cash for the open board from sets we cannot finish
- thaw: unmortgage a set we already own so the even-build ladder can start
- scrap-buy near the cap: engine book is 2.5× list, cash is 1.0×
"""

from __future__ import annotations

from monopoly_game_engine.actions import (
    AUCTION_ACTION_TO_INCREMENT,
    OFFSETS,
    ActionType,
    AuctionAction,
)
from monopoly_game_engine.constants import (
    COLOR_GROUPS,
    JAIL_BAIL,
    PROPERTIES,
    PROPERTY_IDS,
    REAL_ESTATE_IDS,
)
from monopoly_game_engine.env import PHASE_AUCTION, MonopolyEnv

AUCTION_KINDS = ("inncenta", "asu_delta")
FIRST_RACE_COLOURS = ("brown", "lightblue", "darkblue")
HOUSE_LO = OFFSETS["improve_house"]
HOTEL_LO = OFFSETS["improve_hotel"]
SELL_HOUSE_LO = OFFSETS["sell_house"]
SELL_PROP_LO = OFFSETS["sell_prop"]
MORTGAGE_LO = OFFSETS["mortgage"]
UNMORTGAGE_LO = OFFSETS["unmortgage"]
TRADE_LO = OFFSETS["buy_trade"]
AUCTION_LO = OFFSETS["auction"]
REAL_COLOURS = tuple(
    color for color in COLOR_GROUPS if color not in ("railroad", "utility")
)


def has_trade_action(legal: list[int]) -> bool:
    accept = int(ActionType.ACCEPT_TRADE)
    decline = int(ActionType.DECLINE_TRADE)
    return any(a in (accept, decline) or TRADE_LO <= a < AUCTION_LO for a in legal)


def max_live_rent(env: MonopolyEnv, pid: int) -> float:
    """Highest rent an opponent can charge on the current board."""

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
    """Worst published rent we can hit on a 2–12 walk from here."""

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


def cash_floor(env: MonopolyEnv, pid: int) -> float:
    """Wander reserve: max(bail, live hotel, next-roll landing)."""

    return max(float(JAIL_BAIL), max_live_rent(env, pid), next_roll_threat(env, pid))


def spend_floor(env: MonopolyEnv, pid: int) -> float:
    """Cash to keep when building. Distant hotels do not count."""

    return max(float(JAIL_BAIL), next_roll_threat(env, pid))


def complete_floor(env: MonopolyEnv, pid: int) -> float:
    """Cash to keep when finishing the first set. Bail is not worth missing it."""

    return float(next_roll_threat(env, pid))


def opponent_is_developed(env: MonopolyEnv, pid: int) -> bool:
    """True once an opponent has 3-houses or a hotel. Rails do not count."""

    for opp in env.players:
        if opp.player_id == pid or opp.bankrupt:
            continue
        for prop in opp.properties:
            if int(getattr(prop, "houses", 0) or 0) >= 3:
                return True
    return False


def unowned_count(env: MonopolyEnv) -> int:
    return sum(1 for prop in env.properties.values() if prop.owner is None)


def full_sets(env: MonopolyEnv, pid: int) -> int:
    n = 0
    for color in REAL_COLOURS:
        group = COLOR_GROUPS[color]
        if all(env.properties[sq].owner == pid for sq in group):
            n += 1
    return n


def cap_weight(env: MonopolyEnv) -> float:
    """0 until the midpoint, 1 at ``max_rounds``. Cap ranking is net worth."""

    cap = float(getattr(env, "max_rounds", 200) or 200)
    mid = 0.5 * max(cap, 1.0)
    return min(1.0, max(0.0, (float(env.round) - mid) / mid))


def would_complete(env: MonopolyEnv, pid: int, square: int) -> bool:
    data = PROPERTIES.get(square)
    if data is None:
        return False
    group = COLOR_GROUPS.get(data["color"]) or ()
    if not group:
        return False
    return all(env.properties[sq].owner == pid or sq == square for sq in group)


def is_race_square(env: MonopolyEnv, pid: int, square: int) -> bool:
    """Deed that decides the first monopoly: pair colours, one-away, or contested."""

    data = PROPERTIES.get(square)
    if data is None:
        return False
    color = data["color"]
    if color in ("railroad", "utility"):
        return False
    if color in FIRST_RACE_COLOURS:
        return True
    group = COLOR_GROUPS.get(color) or ()
    ours = sum(1 for sq in group if env.properties[sq].owner == pid)
    if ours >= 1:
        return True
    return any(
        env.properties[sq].owner not in (pid, None) for sq in group
    )


def lethal_jail_action(env: MonopolyEnv, pid: int, legal: list[int]) -> int | None:
    """Sit only after 3-houses/hotels exist. Railroad $50 is not expensive."""

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


def build_first_action(env: MonopolyEnv, pid: int, legal: list[int]) -> int | None:
    floor = spend_floor(env, pid)
    cash = float(env.players[pid].cash)
    best: tuple[float, float, int] | None = None
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
        rents = PROPERTIES[square].get("rent") or [0]
        current = float(prop.get_rent())
        nxt = float(rents[min(prop.houses + 1, len(rents) - 1)])
        efficiency = (nxt - current) / max(cost, 1.0)
        if best is None or (efficiency, -cost) > (best[0], best[1]):
            best = (efficiency, -cost, int(action))
    return None if best is None else best[2]


def race_buy_action(env: MonopolyEnv, pid: int, legal: list[int]) -> int | None:
    if int(ActionType.BUY_PROPERTY) not in legal:
        return None
    square = int(env.players[pid].position)
    prop = env.properties.get(square)
    if prop is None or prop.owner is not None:
        return None
    if not is_race_square(env, pid, square):
        return None
    price = float(prop.price)
    floor = complete_floor(env, pid)
    if prop.color not in FIRST_RACE_COLOURS and not would_complete(env, pid, square):
        floor = spend_floor(env, pid)
    if float(env.players[pid].cash) - price < floor:
        return None
    return int(ActionType.BUY_PROPERTY)


def asu_delta_auction(env: MonopolyEnv, pid: int, legal: list[int]) -> int | None:
    """Bid from our ASU swing if we held the deed, capped by live-rent cash."""

    if env.phase != PHASE_AUCTION:
        return None
    bids = [a for a in legal if a != int(AuctionAction.PASS)]
    if not bids:
        return None
    square = getattr(env, "auction_property_id", None)
    if square is None:
        return None
    prop = env.properties.get(int(square))
    if prop is None:
        return None
    from ASU_FROZEN_TEACHER import evaluate_value

    before = float(evaluate_value(env, pid).total)
    saved = prop.owner
    prop.owner = pid
    try:
        after = float(evaluate_value(env, pid).total)
    finally:
        prop.owner = saved
    delta = after - before
    high = float(getattr(env, "auction_high_bid", 0) or 0)
    cash = float(env.players[pid].cash)
    headroom = min(delta, cash - cash_floor(env, pid)) - high
    if headroom <= 0:
        return int(AuctionAction.PASS) if int(AuctionAction.PASS) in legal else None
    for action, step in sorted(AUCTION_ACTION_TO_INCREMENT.items(), key=lambda kv: kv[1]):
        if int(action) in bids and step <= headroom:
            return int(action)
    return int(AuctionAction.PASS) if int(AuctionAction.PASS) in legal else None


def incoming_trade_action(env: MonopolyEnv, pid: int, legal: list[int]) -> int | None:
    """Accept a completing deed; refuse one that finishes or breaks a set."""

    accept = int(ActionType.ACCEPT_TRADE)
    decline = int(ActionType.DECLINE_TRADE)
    if accept not in legal and decline not in legal:
        return None
    offer = env._incoming_trade(pid)
    if offer is None:
        return None
    requested = offer.requested_prop
    offered = offer.offered_prop
    if requested is not None:
        square = int(requested.square_id)
        if would_complete(env, offer.from_player, square):
            return decline if decline in legal else None
        group = COLOR_GROUPS.get(requested.color) or ()
        if group and all(env.properties[sq].owner == pid for sq in group):
            return decline if decline in legal else None
    if offered is not None and would_complete(env, pid, int(offered.square_id)):
        return accept if accept in legal else None
    return None


def scrap_buy_action(env: MonopolyEnv, pid: int, legal: list[int]) -> int | None:
    """Near the cap a deed books at 2.5× list and cash at 1.0×."""

    if int(ActionType.BUY_PROPERTY) not in legal:
        return None
    if cap_weight(env) < 0.5 and unowned_count(env) > 2:
        return None
    square = int(env.players[pid].position)
    prop = env.properties.get(square)
    if prop is None or prop.owner is not None:
        return None
    price = float(prop.price)
    if float(env.players[pid].cash) - price < cash_floor(env, pid):
        return None
    return int(ActionType.BUY_PROPERTY)


def thaw_unmortgage_action(env: MonopolyEnv, pid: int, legal: list[int]) -> int | None:
    """A mortgaged monopoly cannot build. Unfreeze the cheapest deed we fully own."""

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


def dead_mortgage_action(
    env: MonopolyEnv, pid: int, legal: list[int], *, force: bool = False
) -> int | None:
    """Mortgage a colour we can never finish, only while the board is still open."""

    if unowned_count(env) == 0:
        return None
    if getattr(env, "player_needs_funds", None) == pid:
        return None
    cash = float(env.players[pid].cash)
    hungry = force or cash < max(200.0, 1.5 * spend_floor(env, pid))
    if int(ActionType.BUY_PROPERTY) in legal:
        square = int(env.players[pid].position)
        prop = env.properties.get(square)
        if (
            prop is not None
            and prop.owner is None
            and is_race_square(env, pid, square)
            and cash - float(prop.price) < complete_floor(env, pid)
        ):
            hungry = True
    if not hungry:
        return None
    best: tuple[float, int] | None = None
    for action in legal:
        if not (MORTGAGE_LO <= action < UNMORTGAGE_LO):
            continue
        square = PROPERTY_IDS[action - MORTGAGE_LO]
        prop = env.properties[square]
        color = prop.color
        if color not in REAL_COLOURS or prop.houses > 0:
            continue
        group = COLOR_GROUPS[color]
        if all(env.properties[sq].owner == pid for sq in group):
            continue
        blocked = any(
            env.properties[sq].owner not in (pid, None) for sq in group
        )
        if not blocked:
            continue
        value = float(prop.mortgage_v)
        if best is None or value > best[0]:
            best = (value, int(action))
    return None if best is None else best[1]


def is_sell_prop(action: int) -> bool:
    return SELL_PROP_LO <= action < TRADE_LO


def is_dominated_cash_action(action: int, legal: list[int]) -> bool:
    """Bank-sell pays mortgage value and loses the deed. Mortgage keeps it."""

    if action == int(ActionType.DECLARE_BANKRUPT):
        return any(a != action for a in legal)
    if not is_sell_prop(action):
        return False
    square = PROPERTY_IDS[action - SELL_PROP_LO]
    idx = PROPERTY_IDS.index(square)
    return (MORTGAGE_LO + idx) in legal
