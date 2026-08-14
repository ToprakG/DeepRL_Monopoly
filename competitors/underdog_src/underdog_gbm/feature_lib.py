"""Board valuation mathematics and feature context (engine-derived tables).
Landing frequencies are an exact Markov chain over the engine, validated against
460 simulated games / 198,296 landings."""
from __future__ import annotations

import random
from typing import Any, Iterable, Optional, Sequence

import engine_shim  # noqa: F401  (binds the engine package)
from engine.actions import (
    AUCTION_ACTION_TO_INCREMENT,
    OFFSETS,
    ActionType,
    AuctionAction,
)
from engine.constants import (
    COLOR_GROUPS,
    JAIL_BAIL,
    MAX_HOUSES,
    NUM_PLAYERS,
    PROPERTIES,
    PROPERTY_IDS,
    RAILROAD_IDS,
    REAL_ESTATE_IDS,
    TRADE_CASH_LEVELS,
    UTILITY_IDS,
)
from engine.env import (
    PHASE_AUCTION,
    PHASE_OUT_OF_TURN,
    PHASE_POST_ROLL,
    PHASE_PRE_ROLL,
)


__all__ = [
    "LANDINGS_PER_OPP_TURN",
    "GROUP_ORDER",
    "GROUP_RENT_PER_ROUND",
    "GROUP_CAPITAL",
    "GROUP_STEP_EFFICIENCY",
    "GROUP_PRIORITY",
    "RAIL_INCOME_PER_ROUND",
    "UTIL_INCOME_PER_ROUND",
    "ODD_HOUSE_ORDER",
    "object",
    "SpineFixedAgent",
    "variant",
]


# ==========================================================================
# Board mathematics -- measured, not assumed
# ==========================================================================

# E[landings on square s per player-turn], jail policy "never pay bail early".
# Verbatim from research/r4_monopoly_math/LANDING_TABLE_ppo_plus_v2.json, key
# "E_landings_per_player_turn".  Exact Markov chain over the ppo-plus-v2 engine
# (Chance/CC dead), validated against 460 engine games / 198,296 landings.
LANDINGS_PER_OPP_TURN: tuple[float, ...] = (
    0.02551342, 0.02566262, 0.02590089, 0.02613940, 0.02573600,
    0.02548140, 0.02541429, 0.02538803, 0.02546422, 0.02550930,
    0.02558164, 0.02554670, 0.02792083, 0.02678262, 0.02922059,
    0.02820570, 0.03080079, 0.03000032, 0.03151909, 0.02967847,
    0.03134648, 0.02958031, 0.03124225, 0.02939584, 0.02980851,
    0.03008245, 0.03020564, 0.03026375, 0.03022585, 0.03017458,
    0.03007628, 0.03002314, 0.02920140, 0.02833591, 0.02750787,
    0.02657076, 0.02562648, 0.02451689, 0.02499518, 0.02526994,
)

GROUP_ORDER: tuple[str, ...] = (
    "brown", "lightblue", "pink", "orange",
    "red", "yellow", "green", "darkblue",
)
_REAL_GROUPS = frozenset(GROUP_ORDER)
_COLOR_OF = {sq: PROPERTIES[sq]["color"] for sq in PROPERTY_IDS}
_PRICE_OF = {sq: PROPERTIES[sq]["price"] for sq in PROPERTY_IDS}
_MORTGAGE_OF = {sq: PROPERTIES[sq]["mortgage"] for sq in PROPERTY_IDS}
_HOUSE_PRICE = {sq: PROPERTIES[sq]["house_price"] for sq in REAL_ESTATE_IDS}
_PROP_INDEX = {sq: i for i, sq in enumerate(PROPERTY_IDS)}
_REAL_INDEX = {sq: i for i, sq in enumerate(REAL_ESTATE_IDS)}
_GROUP_SQUARES = {g: tuple(sorted(COLOR_GROUPS[g])) for g in GROUP_ORDER}
_N_CASH_LEVELS = len(TRADE_CASH_LEVELS)
_N_PROPS = len(PROPERTY_IDS)
_EXCH_STRIDE = _N_PROPS * (_N_PROPS - 1)
_TRADE_STRIDE = _N_PROPS * _N_CASH_LEVELS
_MAX_OPPONENTS = NUM_PLAYERS - 1


def deed_rent(square: int, level: int, monopoly: bool = True) -> int:
    """Rent a real-estate deed charges at `level` houses (5 = hotel)."""
    rents = PROPERTIES[square]["rent"]
    if level <= 0:
        return rents[0] * 2 if monopoly else rents[0]
    return rents[min(level, 5)]


def _group_rent_per_round(group: str, level: int, opponents: int = 3) -> float:
    """Expected rent collected per ROUND with the whole group at `level`."""
    return opponents * sum(
        LANDINGS_PER_OPP_TURN[sq] * deed_rent(sq, level) for sq in _GROUP_SQUARES[group]
    )


# GROUP_RENT_PER_ROUND[g][L] : $/round from a full group at even level L (3 opponents)
GROUP_RENT_PER_ROUND: dict[str, tuple[float, ...]] = {
    g: tuple(_group_rent_per_round(g, L) for L in range(6)) for g in GROUP_ORDER
}
# GROUP_CAPITAL[g][L] : total $ sunk -- deeds plus every house -- at even level L
GROUP_CAPITAL: dict[str, tuple[int, ...]] = {
    g: tuple(
        sum(_PRICE_OF[sq] for sq in _GROUP_SQUARES[g])
        + L * len(_GROUP_SQUARES[g]) * _HOUSE_PRICE[_GROUP_SQUARES[g][0]]
        for L in range(6)
    )
    for g in GROUP_ORDER
}
# GROUP_STEP_EFFICIENCY[g][L] : $/round gained per $1000 spent taking the WHOLE
# group from level L-1 to L.  This is the even-build unit of decision.
GROUP_STEP_EFFICIENCY: dict[str, tuple[float, ...]] = {
    g: (0.0,)
    + tuple(
        (GROUP_RENT_PER_ROUND[g][L] - GROUP_RENT_PER_ROUND[g][L - 1])
        / (len(_GROUP_SQUARES[g]) * _HOUSE_PRICE[_GROUP_SQUARES[g][0]])
        * 1000.0
        for L in range(1, 6)
    )
    for g in GROUP_ORDER
}
# GROUP_PRIORITY[g] : which colour to chase, best first.  Derived, not asserted.
# Score = rent/round at L3 divided by total capital to get there, i.e. the
# reciprocal of full-acquisition payback.  Computed order:
#   orange(9.3) > yellow(9.7) > darkblue(10.4) > red(10.6) > green(11.2)
#   > pink(11.7) > lightblue(12.0) > brown(20.0)      [rounds to pay back]
# That reproduces the received wisdom exactly except for lightblue, which the
# wisdom places 4th and the board mathematics place 7th: lightblue's payback is
# good only because it is cheap, and $64/round at L3 does not bankrupt anybody.
# Build ranking uses by marginal step efficiency (which is level- and
# state-dependent) and breaks ties on GROUP_POWER, not on this static table.
GROUP_PRIORITY: dict[str, float] = {
    g: GROUP_RENT_PER_ROUND[g][3] / GROUP_CAPITAL[g][3] for g in GROUP_ORDER
}
# Absolute earning power at L3 -- used to break ties toward groups that can
# actually bankrupt somebody.  brown at L3 earns $50/round; that does not win.
GROUP_POWER: dict[str, float] = {g: GROUP_RENT_PER_ROUND[g][3] for g in GROUP_ORDER}

# ODD_HOUSE_ORDER[g][L] : within a group, which deed should receive the next
# house when going to level L.  Ranked by P(land) x MARGINAL rent, which is NOT
# the same as landing probability alone -- it reorders orange and yellow at L4+.
ODD_HOUSE_ORDER: dict[str, dict[int, tuple[int, ...]]] = {
    g: {
        L: tuple(
            sorted(
                _GROUP_SQUARES[g],
                key=lambda sq: -(
                    LANDINGS_PER_OPP_TURN[sq]
                    * (deed_rent(sq, L) - deed_rent(sq, L - 1))
                ),
            )
        )
        for L in range(1, 6)
    }
    for g in GROUP_ORDER
}

_RAIL_RENT = (0, 25, 50, 100, 200)
_RAIL_TRAFFIC = sum(LANDINGS_PER_OPP_TURN[sq] for sq in RAILROAD_IDS) / len(RAILROAD_IDS)
# RAIL_INCOME_PER_ROUND[n] : $/round from owning n railroads (3 opponents).
RAIL_INCOME_PER_ROUND: tuple[float, ...] = tuple(
    3.0 * _RAIL_TRAFFIC * n * _RAIL_RENT[n] for n in range(5)
)
_UTIL_TRAFFIC = sum(LANDINGS_PER_OPP_TURN[sq] for sq in UTILITY_IDS) / len(UTILITY_IDS)
# UTIL_INCOME_PER_ROUND[n] : $/round from owning n utilities, E[dice] = 7.
UTIL_INCOME_PER_ROUND: tuple[float, ...] = (
    0.0,
    3.0 * _UTIL_TRAFFIC * 1 * 4 * 7,
    3.0 * _UTIL_TRAFFIC * 2 * 10 * 7,
)

_END_TURN = int(ActionType.END_TURN)
_ROLL_DICE = int(ActionType.ROLL_DICE)
_BUY_PROPERTY = int(ActionType.BUY_PROPERTY)
_PAY_BAIL = int(ActionType.PAY_BAIL)
_ACCEPT_TRADE = int(ActionType.ACCEPT_TRADE)
_DECLINE_TRADE = int(ActionType.DECLINE_TRADE)
_DECLARE_BANKRUPT = int(ActionType.DECLARE_BANKRUPT)
_USE_GOOJ = int(ActionType.USE_GOOJ_CARD)
_AUCTION_PASS = int(AuctionAction.PASS)


# ==========================================================================
# The agent
# ==========================================================================



class _Context:
    """Per-position feature context: ownership, income and valuation tables."""

    __slots__ = (
        "cfg", "env", "pid", "opponents", "opp_scale", "targets",
        "other_index", "owned", "reserve", "danger_per_turn",
        "unowned_deeds", "_deed_cache",
    )

    def __init__(self, cfg: object, env, pid: int) -> None:
        self.cfg = cfg
        self.env = env
        self.pid = pid
        self._deed_cache: dict[tuple[int, int, bool], float] = {}

        self.opponents = [
            i for i in range(NUM_PLAYERS) if i != pid and not env.players[i].bankrupt
        ]
        self.opp_scale = max(len(self.opponents), 1) / _MAX_OPPONENTS
        others = [i for i in range(NUM_PLAYERS) if i != pid]
        self.other_index = {p: i for i, p in enumerate(others)}
        self.targets = [p for p in others if not env.players[p].bankrupt]

        # owned[player][colour] -> count of unmortgaged-or-not deeds held
        self.owned: dict[int, dict[str, int]] = {
            p: {g: 0 for g in GROUP_ORDER} for p in range(NUM_PLAYERS)
        }
        worst_rent = 0
        exposure = 0.0
        unowned = 0
        for square in PROPERTY_IDS:
            prop = env.properties[square]
            owner = prop.owner
            if owner is None:
                unowned += 1
                continue
            colour = _COLOR_OF[square]
            if colour in _REAL_GROUPS:
                self.owned[owner][colour] += 1
            if owner == pid or prop.mortgaged:
                continue
            rent = prop.get_rent(
                7, env.players[owner].railroads_owned(), env.players[owner].utilities_owned()
            )
            worst_rent = max(worst_rent, rent)
            exposure += LANDINGS_PER_OPP_TURN[square] * rent
        self.unowned_deeds = unowned
        self.danger_per_turn = exposure
        self.reserve = float(
            min(
                cfg.RESERVE_CAP,
                cfg.RESERVE_BASE + cfg.RESERVE_RENT_K * worst_rent,
            )
        )

    # --- ownership questions ------------------------------------------

    def completes_for(self, square: int, player: int) -> bool:
        """True if giving `square` to `player` hands them the whole colour."""
        colour = _COLOR_OF[square]
        if colour not in _REAL_GROUPS:
            return False
        squares = _GROUP_SQUARES[colour]
        held = sum(
            1
            for sq in squares
            if sq != square and self.env.properties[sq].owner == player
        )
        return held == len(squares) - 1

    def is_development_group(self, square: int) -> bool:
        """True if `square` sits in a monopoly of ours -- never mortgage these."""
        colour = _COLOR_OF[square]
        if colour not in _REAL_GROUPS:
            return False
        return self.owned[self.pid][colour] == len(_GROUP_SQUARES[colour])

    def hotels_ok(self, env) -> bool:
        """Hotelling hands 4 houses per deed back to the bank.  Under a housing
        shortage that is a gift to any opponent sitting on a bare monopoly."""
        if env.houses_available >= self.cfg.HOTEL_HOLD_HOUSES:
            return True
        for opponent in self.opponents:
            for group in GROUP_ORDER:
                if self.owned[opponent][group] != len(_GROUP_SQUARES[group]):
                    continue
                if all(env.properties[sq].houses == 0 for sq in _GROUP_SQUARES[group]):
                    return False
        return True

    # --- income --------------------------------------------------------

    def current_group_rent(self, group: str) -> float:
        """$/round the group currently earns its owner, at its current layout."""
        env = self.env
        total = 0.0
        for square in _GROUP_SQUARES[group]:
            prop = env.properties[square]
            if prop.mortgaged:
                continue
            total += LANDINGS_PER_OPP_TURN[square] * deed_rent(
                square, prop.houses, prop.is_monopoly
            )
        return 3.0 * total

    def square_income(self, square: int, assume_unmortgaged: bool = False) -> float:
        """$/round this one square earns its current owner."""
        env = self.env
        prop = env.properties[square]
        if prop.mortgaged and not assume_unmortgaged:
            return 0.0
        colour = _COLOR_OF[square]
        traffic = 3.0 * self.opp_scale * LANDINGS_PER_OPP_TURN[square]
        if colour == "railroad":
            owner = prop.owner
            n = env.players[owner].railroads_owned() if owner is not None else 1
            return traffic * _RAIL_RENT[max(1, min(n, 4))]
        if colour == "utility":
            owner = prop.owner
            n = env.players[owner].utilities_owned() if owner is not None else 1
            return traffic * (4 * 7 if n <= 1 else 10 * 7)
        return traffic * deed_rent(square, prop.houses, prop.is_monopoly)

    def block_income(self, square: int) -> float:
        """$/round an opponent would gain if we released `square` to them."""
        colour = _COLOR_OF[square]
        if colour not in _REAL_GROUPS:
            return 0.0
        for opponent in self.opponents:
            if self.completes_for(square, opponent):
                target = self.cfg.VALUE_LEVEL
                return (
                    GROUP_RENT_PER_ROUND[colour][target] * self.opp_scale
                    - self.current_group_rent(colour)
                )
        return 0.0

    # --- deed valuation, in dollars ------------------------------------

    def deed_value(self, square: int, player: int, owned: bool = False) -> float:
        """What owning `square` is worth to `player`, in dollars.

        Rent income over VALUE_HORIZON rounds, plus the value of completing a
        colour group at VALUE_LEVEL, plus BLOCK_WEIGHT of whatever an opponent
        would have gained had they got it instead.
        """
        key = (square, player, owned)
        cached = self._deed_cache.get(key)
        if cached is not None:
            return cached
        value = self._deed_value(square, player)
        value = min(value, self.cfg.VALUE_CAP_MULT * _PRICE_OF[square])
        self._deed_cache[key] = value
        return value

    def _deed_value(self, square: int, player: int) -> float:
        cfg, env = self.cfg, self.env
        horizon = cfg.VALUE_HORIZON
        colour = _COLOR_OF[square]
        traffic = 3.0 * self.opp_scale * LANDINGS_PER_OPP_TURN[square]

        if colour == "railroad":
            held = sum(
                1 for sq in RAILROAD_IDS if env.properties[sq].owner == player
            )
            if env.properties[square].owner == player:
                held -= 1
            after = min(held + 1, 4)
            # Marginal: the whole set's income at `after` minus at `held`.  The
            # 4th railroad is worth far more than the 1st (payback 12.1 rounds
            # vs 96.7), and a flat per-deed value would miss that entirely.
            gain = (RAIL_INCOME_PER_ROUND[after] - RAIL_INCOME_PER_ROUND[held]) * self.opp_scale
            return horizon * max(gain, 0.0)
        if colour == "utility":
            held = sum(1 for sq in UTILITY_IDS if env.properties[sq].owner == player)
            if env.properties[square].owner == player:
                held -= 1
            after = min(held + 1, 2)
            gain = (UTIL_INCOME_PER_ROUND[after] - UTIL_INCOME_PER_ROUND[held]) * self.opp_scale
            return horizon * max(gain, 0.0)

        squares = _GROUP_SQUARES[colour]
        n = len(squares)
        mine = sum(
            1
            for sq in squares
            if sq != square and env.properties[sq].owner == player
        )
        solo = horizon * traffic * PROPERTIES[square]["rent"][0]

        # 1. Completion.
        if mine == n - 1:
            target = min(cfg.VALUE_LEVEL, 5)
            gain = (
                GROUP_RENT_PER_ROUND[colour][target] * self.opp_scale
                - self.current_group_rent(colour)
            )
            return solo + horizon * max(gain, 0.0)

        # 2. Progress toward a group that is still gettable.
        blocked_by = {
            env.properties[sq].owner
            for sq in squares
            if sq != square and env.properties[sq].owner not in (None, player)
        }
        value = solo
        if not blocked_by:
            target = min(cfg.VALUE_LEVEL, 5)
            full = GROUP_RENT_PER_ROUND[colour][target] * self.opp_scale
            value += (
                cfg.PARTIAL_WEIGHT
                * horizon
                * full
                * ((mine + 1) / n) ** 2
            )

        # 3. Denial: what an opponent loses by not getting it.
        for opponent in range(NUM_PLAYERS):
            if opponent == player or env.players[opponent].bankrupt:
                continue
            held = sum(
                1
                for sq in squares
                if sq != square and env.properties[sq].owner == opponent
            )
            if held == n - 1:
                target = min(cfg.VALUE_LEVEL, 5)
                gain = (
                    GROUP_RENT_PER_ROUND[colour][target] * self.opp_scale
                    - self.current_group_rent(colour)
                )
                value += cfg.BLOCK_WEIGHT * horizon * max(gain, 0.0)
                break
        return value



def unwrap(game):
    return getattr(game, "env", game)


class _Cfg:
    pass


CFG = _Cfg()
for _k, _v in {'AUCTION_MAX_MULT': 2.5, 'AUCTION_MIN_RAISE': True, 'AUCTION_PLAIN_MULT': 0.85, 'AUCTION_RESERVE': 80, 'AUC_DECIDE_OPP_ONLY': True, 'BLOCK_WEIGHT': 0.55, 'BUILD_ESCALATE': 5, 'BUILD_RESERVE_RELIEF': 0.0, 'BUILD_TARGET': 3, 'BUY_RESERVE_RELIEF': 0.5, 'CHEST_CLEARING_MULT': 1.0, 'DENY_DARKBLUE': True, 'ENDGAME_BUY_MULT': 1.0, 'ENDGAME_SPAN': 100.0, 'EPSILON': 0.0, 'ESCALATE_LUMPS': 2.0, 'GUARD_SALE_COVERS': False, 'HOTEL_HOLD_HOUSES': 6, 'JAIL_CAPSIT_ROUNDS_LEFT': 4, 'JAIL_EARLY_ROUNDS': 25, 'JAIL_SIT': True, 'JUNK_SELL_MULT': 1.25, 'MORTGAGE_TO_BUILD': True, 'PARTIAL_WEIGHT': 0.3, 'R8_REPEAT_LIMIT': 2, 'RESERVE_BASE': 150, 'RESERVE_CAP': 900, 'RESERVE_RENT_K': 0.6, 'TRADE_ACCEPT_MARGIN': 1.0, 'TRADE_ENABLED': True, 'TRADE_OFFER_MARGIN': 60.0, 'TRADE_PEER_MARGIN': 1.0, 'UNMORTGAGE_BUFFER': 250, 'VALUE_CAP_MULT': 6.0, 'VALUE_HORIZON': 100.0, 'VALUE_LEVEL': 3}.items():
    setattr(CFG, _k, _v)


def dead_deeds(ctx) -> list:
    env, pid = ctx.env, ctx.pid
    out = []
    for square in PROPERTY_IDS:
        prop = env.properties[square]
        if prop.owner != pid or prop.houses:
            continue
        colour = _COLOR_OF[square]
        if colour not in _REAL_GROUPS:
            continue          # a railroad is never dead; each one raises the set
        for sq in _GROUP_SQUARES[colour]:
            other = env.properties[sq].owner
            if other is not None and other != pid:
                out.append(square)
                break
    return out


def chest_target(ctx) -> float:
    env = ctx.env
    biggest = 0
    for square in PROPERTY_IDS:
        if env.properties[square].owner is None:
            price = _PRICE_OF[square]
            if price > biggest:
                biggest = price
    if biggest <= 0:
        return 0.0
    return float(CFG.AUCTION_RESERVE + CFG.CHEST_CLEARING_MULT * biggest)


def make_context(env, pid: int) -> "_Context":
    return _Context(CFG, env, int(pid))
