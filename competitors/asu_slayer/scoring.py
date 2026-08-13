"""Net-worth-exact valuation for ``ppo-plus-v2``.

The engine decides every capped game by ``Player.net_worth()``, and that
function does not price a deed at its list price: an unmortgaged deed is worth
``2.5 x price``, or ``5.0 x price`` once its color group is complete, and a
development level is worth ``houses * house_price * (1 + 0.5 * houses)``. Every
number below is derived from that scoring rule, so the agent optimizes the
quantity the simulator actually ranks players by.
"""

from __future__ import annotations

from monopoly_game_engine.actions import OFFSETS, PROPERTY_IDS
from monopoly_game_engine.constants import MAX_HOUSES, PROPERTIES, REAL_ESTATE_IDS

from .board import exposure, group_of, income, is_real_estate, owned_in_group


SOLO_MULTIPLIER = 2.5
MONOPOLY_MULTIPLIER = 5.0
# A colour group an opponent has entered can never be built on by us
# through purchase, so its deeds are kept only for blocking and trade.
BLOCKED_GROUP_WEIGHT = 0.25
_PROPERTY_INDEX = {square: index for index, square in enumerate(PROPERTY_IDS)}
_REAL_ESTATE_INDEX = {square: index for index, square in enumerate(REAL_ESTATE_IDS)}


def deed_worth(square: int, houses: int, mortgaged: bool, monopoly: bool) -> float:
    """``Property.calculate_net_worth`` evaluated on hypothetical attributes."""

    data = PROPERTIES[square]
    price = data["price"]
    mortgage_value = data["mortgage"] if mortgaged else 0
    base = (price - mortgage_value) * (
        MONOPOLY_MULTIPLIER if monopoly else SOLO_MULTIPLIER
    )
    if not is_real_estate(square) or houses <= 0:
        return base
    house_price = data["house_price"]
    multiplier = 1.0 + houses * 0.5
    count = 5 if houses == 5 else houses
    return base + count * house_price * multiplier


def group_worth(env, owner: int, squares: tuple[int, ...]) -> float:
    """Total net worth ``owner`` currently draws from one color group."""

    monopoly = all(env.properties[item].owner == owner for item in squares)
    total = 0.0
    for item in squares:
        prop = env.properties[item]
        if prop.owner != owner:
            continue
        total += deed_worth(item, prop.houses, prop.mortgaged, monopoly)
    return total


def acquisition_gain(
    env, player_id: int, square: int, include_denial: bool = True
) -> float:
    """Net worth gained by acquiring ``square``, ignoring the price paid.

    Completing a color group re-prices every deed already held in it, so this
    is far larger than the deed's own worth at the moment a group closes.

    ``include_denial`` adds what the current owner loses, which is what a
    competitive decision cares about. Pass ``False`` to compare two players'
    gains from the same trade, where counting one side's loss as the other
    side's gain would double it.
    """

    squares = group_of(square)
    before = group_worth(env, player_id, squares)
    owner = env.properties[square].owner
    completes = all(
        item == square or env.properties[item].owner == player_id for item in squares
    )
    after = 0.0
    for item in squares:
        prop = env.properties[item]
        if item != square and prop.owner != player_id:
            continue
        after += deed_worth(item, prop.houses, prop.mortgaged, completes)
    gain = after - before
    if include_denial and owner is not None and owner != player_id:
        # Taking a deed from a live opponent also removes it from their score.
        gain += group_worth(env, owner, squares) - _group_worth_without(
            env, owner, squares, square
        )
    return gain


def _group_worth_without(
    env, owner: int, squares: tuple[int, ...], removed: int
) -> float:
    # Every color group has at least two deeds, so losing one always breaks it.
    monopoly = False
    total = 0.0
    for item in squares:
        if item == removed:
            continue
        prop = env.properties[item]
        if prop.owner != owner:
            continue
        total += deed_worth(item, prop.houses, prop.mortgaged, monopoly)
    return total


def development_outlook(env, player_id: int, square: int) -> float:
    """How much of a deed's worth is still strategically live for this player.

    Houses need the complete colour group — the engine refuses to build
    otherwise — and rent without houses is trivial: New York Avenue pays $16
    alone, $32 as a monopoly, and $1,000 with a hotel. So a deed in a group an
    opponent already holds part of can never become a weapon for us, however
    much net worth it books.

    ``net_worth`` prices both kinds of deed identically at 2.5x list, and
    following it spread the policy across 4.5 colour groups while ASU
    concentrated on 3.2, which is the measured difference between them.
    Railroads and utilities are exempt: their rent scales with how many you
    hold, so partial ownership still earns.
    """

    if not is_real_estate(square):
        return 1.0
    blocked = any(
        item != square
        and env.properties[item].owner is not None
        and env.properties[item].owner != player_id
        for item in group_of(square)
    )
    return BLOCKED_GROUP_WEIGHT if blocked else 1.0


def disposal_loss(env, player_id: int, square: int) -> float:
    """Net worth lost by giving ``square`` away, including a broken group."""

    squares = group_of(square)
    return group_worth(env, player_id, squares) - _group_worth_without(
        env, player_id, squares, square
    )


def improvement_gain(env, square: int, to_hotel: bool) -> float:
    """Net worth added by one build step, excluding the cash spent."""

    prop = env.properties[square]
    current = prop.houses
    following = 5 if to_hotel else current + 1
    monopoly = owned_in_group(env, prop.owner, square) == len(group_of(square))
    return deed_worth(square, following, prop.mortgaged, monopoly) - deed_worth(
        square, current, prop.mortgaged, monopoly
    )


def mortgage_loss(env, square: int) -> float:
    """Net worth destroyed by mortgaging ``square``."""

    prop = env.properties[square]
    monopoly = owned_in_group(env, prop.owner, square) == len(group_of(square))
    return deed_worth(square, prop.houses, False, monopoly) - deed_worth(
        square, prop.houses, True, monopoly
    )


def net_worth(env, player_id: int) -> float:
    return float(env.players[player_id].net_worth())


def liquidation_options(env, player_id: int, legal: set[int]) -> list[tuple]:
    """Legal cash-raising actions ranked by net worth destroyed per dollar.

    Returns ``(ratio, cash, loss, action)`` tuples. Mortgaging an ordinary deed
    costs 2.5 net worth per dollar raised; selling it to the bank costs 5.0 and
    breaking a hotel costs 11.0, so the ordering here is a real strategic edge
    over liquidating in action-id order.
    """

    options: list[tuple[float, float, float, int]] = []
    for square in PROPERTY_IDS:
        prop = env.properties[square]
        if prop.owner != player_id:
            continue
        index = _PROPERTY_INDEX[square]
        data = PROPERTIES[square]
        monopoly = owned_in_group(env, player_id, square) == len(group_of(square))

        action = OFFSETS["mortgage"] + index
        if action in legal:
            cash = float(prop.mortgage_v)
            loss = mortgage_loss(env, square)
            options.append((loss / max(cash, 1.0), cash, loss, action))

        action = OFFSETS["sell_prop"] + index
        if action in legal:
            cash = float(prop.mortgage_v)
            loss = deed_worth(square, prop.houses, prop.mortgaged, monopoly)
            options.append((loss / max(cash, 1.0), cash, loss, action))

        if not is_real_estate(square) or prop.houses <= 0:
            continue
        real_index = _REAL_ESTATE_INDEX[square]
        house_price = data["house_price"]
        cash = float(house_price // 2)
        if prop.houses == 5:
            action = OFFSETS["sell_hotel"] + real_index
            following = MAX_HOUSES
        else:
            action = OFFSETS["sell_house"] + real_index
            following = prop.houses - 1
        if action in legal:
            loss = deed_worth(square, prop.houses, prop.mortgaged, monopoly) - deed_worth(
                square, following, prop.mortgaged, monopoly
            )
            options.append((loss / max(cash, 1.0), cash, loss, action))

    options.sort(key=lambda item: (item[0], -item[1], item[3]))
    return options


def liquidatable_cash(env, player_id: int) -> float:
    """Cash reachable by mortgaging every unmortgaged, undeveloped deed."""

    total = 0.0
    for square in PROPERTY_IDS:
        prop = env.properties[square]
        if prop.owner != player_id or prop.mortgaged:
            continue
        if prop.houses:
            total += prop.houses * (PROPERTIES[square]["house_price"] // 2)
        else:
            total += prop.mortgage_v
    return total


def strength(env, player_id: int, rent_horizon: float) -> float:
    """Net worth plus the earning power that will convert into net worth."""

    player = env.players[player_id]
    if player.bankrupt:
        return 0.0
    expected_out, _worst = exposure(env, player_id, 1)
    return (
        net_worth(env, player_id)
        + rent_horizon * (income(env, player_id, 1) - expected_out)
    )


def equity(env, player_id: int, rent_horizon: float, survival_bonus: float) -> float:
    """Advantage over the strongest live opponent, the quantity that wins games."""

    player = env.players[player_id]
    if player.bankrupt:
        return -1e9
    live = [other for other in env.players if not other.bankrupt]
    if len(live) == 1:
        return 1e9
    own = strength(env, player_id, rent_horizon)
    best_other = max(
        strength(env, other.player_id, rent_horizon)
        for other in live
        if other.player_id != player_id
    )
    eliminated = sum(1 for other in env.players if other.bankrupt)
    return own - best_other + survival_bonus * eliminated


__all__ = [
    "acquisition_gain",
    "deed_worth",
    "development_outlook",
    "equity",
    "group_worth",
    "improvement_gain",
    "liquidatable_cash",
    "liquidation_options",
    "mortgage_loss",
    "net_worth",
    "strength",
]
