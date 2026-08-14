"""SPINE-v2 -- the strong hand-written heuristic for ppo-plus-v2.

WHAT THIS IS FOR
----------------
SPINE can never be the submitted entry.  The 95% rule says the submitted MODEL
must decide 95 of every 100 non-forced actions, and trading alone is 37-65% of
non-forced decisions, so a rules-based policy is an illegal entry no matter how
good it is.  SPINE exists for three other jobs, all of which need it to be
*strong*:

  1. BEHAVIOUR-CLONING TEACHER.  Its strength transfers to the student network
     that we do submit.  Every point of win rate here is a point the student
     starts from.
  2. POTENTIAL FUNCTION for reward shaping.  `Spine.position_value` is a
     dollar-denominated evaluation of a seat that is not the engine's own
     net-worth proxy (which is measurably anti-correlated with winning).
  3. SPARRING OPPONENT.  Every experiment is measured against it.

PORTED FROM (this is not written from scratch)
----------------------------------------------
  research/r2_sweep/theory_agent.py   -- the 63.0% +/- 3.4 r2 heuristic.  Ported:
      the layered choose_action skeleton, the CORRECTED trade-acceptance sign
      (upstream agents_fixed.py has it inverted), the "never hand over a deed
      that completes someone else's group / never break my own monopoly" vetoes,
      the "smallest legal raise wins the auction cheap" trick, the
      danger-vs-bail jail comparison, and mortgage-ranked-by-rent-lost-per-dollar.
      NOT ported: its `UNEVEN` build allocator and `marg_rent(sq, lvl)` single-deed
      ladder -- both are v1 artefacts.  v2 enforces even building, so the unit of
      decision is a whole-group level step, not one deed.
  research/r3_anti_asu/agents_x.py    -- ported: the explicit phase switch
      (PHASE_AUCTION / OUT_OF_TURN / PRE_ROLL / POST_ROLL) instead of relying on
      the base class ordering, the `_rescue` debt ladder shape, and the
      value-capped auction with a cash reserve term (the measured hole in ASU:
      no cash term, auction ceiling averages 4.26x list).
  research/r3_even/econ2.py, econ3.py, econ4.py -- ported: the even-build ladder
      formulation (whole-group level steps, cash lump = m x house_price), the
      "jump to level L" bundle ranking, and the odd-house rule
      "P(land) x MARGINAL rent", not landing probability alone.
  research/r4_monopoly_math/LANDING_TABLE_ppo_plus_v2.json -- ported wholesale as
      `LANDINGS_PER_OPP_TURN` below.  Exact Markov chain over the v2 engine
      (Fraction arithmetic), validated against 460 real games / 198,296 landings,
      max |z| = 3.72.  This is the single most valuable thing in research/.

RULESET FACTS THIS AGENT RELIES ON (all read out of engine/env.py, not assumed)
------------------------------------------------------------------------------
  * Chance and Community Chest DRAW NOTHING.  env._handle_landing returns early
    for any square not in env.properties.  Get-out-of-jail cards are therefore
    unobtainable, and USE_GOOJ_CARD is dead code.
  * Sitting in jail is FREE.  A failed jail roll returns before the move, so you
    pay no rent, keep collecting it, and may still build and trade in pre-roll.
  * Declining to buy (END_TURN in post-roll on an unowned deed) sends the deed to
    AUCTION, and the declining player bids FIRST.  Declining is a real option.
  * Building is legal in PRE_ROLL only; the normal post-roll menu has no improve
    actions.
  * Even building is enforced by env._is_least_developed.
  * Exchange (deed-for-deed) offers are legal: OFFSETS["exch_trade"], 2268 of them.
  * One pending trade offer per player per turn; cleared on _next_player.
  * sell_prop (deed back to the bank at mortgage value) is STRICTLY DOMINATED by
    mortgage: same cash, but you lose the deed and its blocking value forever.
    SPINE never uses it except as the last rung before bankruptcy.

CONTRACT
--------
    from mono.spine import Spine
    agent = Spine()
    action = agent.choose_action(game, player_id, decision_seed)

`choose_action` is `mono.types.Agent`.  The agent is stateless and seat-agnostic:
`player_id` is read from the argument every call, never cached, so one instance
can legally play all four seats at once.
"""

from __future__ import annotations

import random
from typing import Any, Iterable, Optional, Sequence

from . import _bind  # noqa: F401  -- binds engine/ as monopoly_game_engine; must come first
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
from ._bind import unwrap

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
    "Spine",
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
# SPINE therefore ranks builds by marginal step efficiency (which is level- and
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


class Spine:
    """SPINE-v2.  Every knob is a class attribute so `variant()` can sweep it."""

    name = "spine_v2"

    # --- cash discipline ---------------------------------------------------
    # Reserve is the cash floor we refuse to spend below.  r2 measured reserve
    # size as one of the largest single levers, so it is dynamic: it tracks the
    # biggest single rent bill currently on the board.
    RESERVE_BASE = 150
    RESERVE_RENT_K = 0.60          # fraction of the worst single rent hit to hold
    RESERVE_CAP = 900
    AUCTION_RESERVE = 80           # a thinner floor: an auction win is an asset
    BUY_RESERVE_RELIEF = 0.50      # completing/blocking deeds may eat this much
                                   # of the reserve

    # --- valuation ---------------------------------------------------------
    VALUE_HORIZON = 12.0           # rounds of rent a deed is valued over
    VALUE_LEVEL = 3                # development level assumed when valuing a group
    PARTIAL_WEIGHT = 0.30          # credit for progress toward an open group
    BLOCK_WEIGHT = 0.55            # denial is worth this fraction of the gain
    VALUE_CAP_MULT = 6.0           # hard cap on deed value, in multiples of list

    # --- building ----------------------------------------------------------
    BUILD_TARGET = 3               # plan every monopoly to level 3 first
    BUILD_ESCALATE = 5             # then to 4/hotel once cash is deep
    ESCALATE_LUMPS = 2.0           # "deep" = reserve + this many full lumps spare
    HOTEL_HOLD_HOUSES = 6          # do not free houses to the bank below this
                                   # while an opponent has an undeveloped monopoly

    # --- auctions ----------------------------------------------------------
    # The measured hole in ASU: no cash term, ceiling averages 4.26x list.  A
    # disciplined ceiling beats that for free.
    AUCTION_MAX_MULT = 2.50        # never bid above this multiple of list
    AUCTION_PLAIN_MULT = 0.85      # a deed we merely like is only worth a bargain
    AUCTION_MIN_RAISE = True       # win cheap: take the smallest legal increment

    # --- trading -----------------------------------------------------------
    TRADE_ENABLED = True
    TRADE_ACCEPT_MARGIN = 1.0      # $ of edge required to accept
    TRADE_OFFER_MARGIN = 60.0      # $ of edge required to bother offering
    TRADE_PEER_MARGIN = 1.0        # $ the counterparty must gain for us to bother
    JUNK_SELL_MULT = 1.25          # sell dead deeds at the top cash level

    # --- jail --------------------------------------------------------------
    JAIL_EARLY_ROUNDS = 25         # before this, circulating is worth $50
    JAIL_SIT = True

    # --- mortgage ----------------------------------------------------------
    MORTGAGE_TO_BUILD = True
    UNMORTGAGE_BUFFER = 250        # spare cash above reserve before redeeming

    # --- exploration (0.0 = deterministic; raise it to diversify BC data) ---
    EPSILON = 0.0

    def __init__(self, name: Optional[str] = None, **overrides: Any) -> None:
        if name is not None:
            self.name = name
        for key, value in overrides.items():
            if not hasattr(type(self), key):
                raise AttributeError(f"Spine has no tunable named {key!r}")
            setattr(self, key, value)

    # ------------------------------------------------------------------
    # mono.types.Agent
    # ------------------------------------------------------------------

    def choose_action(self, game: Any, player_id: int, decision_seed: int) -> int:
        env = unwrap(game)
        pid = int(player_id)
        allowed = tuple(int(a) for a in env.get_allowed_actions(pid))
        if len(allowed) == 1:
            return allowed[0]

        if self.EPSILON > 0.0:
            rng = random.Random(decision_seed)
            if rng.random() < self.EPSILON:
                return rng.choice(allowed)

        action = self._decide(env, pid, allowed, set(allowed))
        # Fail-safe.  The arena scores an illegal action as a CRASH, not a loss,
        # so SPINE never trusts its own arithmetic on the way out.
        if action not in allowed:
            return _END_TURN if _END_TURN in allowed else allowed[0]
        return int(action)

    # ------------------------------------------------------------------
    # Layered policy
    # ------------------------------------------------------------------

    def _decide(self, env, pid: int, allowed: Sequence[int], aset: set) -> int:
        ctx = _Context(self, env, pid)

        if env.phase == PHASE_AUCTION:
            return self._auction(ctx, allowed, aset)

        # A pending offer aimed at us outranks everything: it is the only
        # decision that expires this step.
        if _ACCEPT_TRADE in aset:
            return _ACCEPT_TRADE if self._accept_offer(ctx) else _DECLINE_TRADE

        # Rescue: unpaid rent must be settled before the turn can continue.
        if env.debt_player == pid:
            rescue = self._rescue(ctx, aset)
            if rescue is not None:
                return rescue
            return _DECLARE_BANKRUPT if _DECLARE_BANKRUPT in aset else allowed[0]

        if env.phase == PHASE_POST_ROLL:
            return self._post_roll(ctx, allowed, aset)

        if env.phase == PHASE_OUT_OF_TURN:
            offer = self._make_offer(ctx, aset)
            if offer is not None:
                return offer
            return _END_TURN if _END_TURN in aset else allowed[0]

        if env.phase == PHASE_PRE_ROLL:
            return self._pre_roll(ctx, allowed, aset)

        return _END_TURN if _END_TURN in aset else allowed[0]

    # ------------------------------------------------------------------
    # POST-ROLL: roll, then buy or push to auction
    # ------------------------------------------------------------------

    def _post_roll(self, ctx, allowed: Sequence[int], aset: set) -> int:
        env, pid = ctx.env, ctx.pid
        if not env.has_rolled:
            if env.players[pid].in_jail:
                jail = self._jail(ctx, aset)
                if jail is not None:
                    return jail
            return _ROLL_DICE if _ROLL_DICE in aset else allowed[0]

        if _BUY_PROPERTY in aset:
            square = env.players[pid].position
            if self._should_buy(ctx, square):
                return _BUY_PROPERTY
            # Otherwise END_TURN, which sends the deed to auction with US bidding
            # first (env._handle_end_turn -> _start_auction).  Declining is how a
            # disciplined bidder buys below list.

        return _END_TURN if _END_TURN in aset else allowed[0]

    def _should_buy(self, ctx, square: int) -> bool:
        env, pid = ctx.env, ctx.pid
        prop = env.properties.get(square)
        if prop is None or prop.owner is not None:
            return False
        price = _PRICE_OF[square]
        cash = env.players[pid].cash
        value = ctx.deed_value(square, pid)
        critical = value >= 1.5 * price
        floor = ctx.reserve * (self.BUY_RESERVE_RELIEF if critical else 1.0)
        if cash - price < floor:
            return False
        return value >= price

    # ------------------------------------------------------------------
    # PRE-ROLL: build, redeem, trade, mortgage-to-build, jail, then roll
    # ------------------------------------------------------------------

    def _pre_roll(self, ctx, allowed: Sequence[int], aset: set) -> int:
        build = self._build(ctx, aset)
        if build is not None:
            return build

        redeem = self._unmortgage(ctx, aset)
        if redeem is not None:
            return redeem

        offer = self._make_offer(ctx, aset)
        if offer is not None:
            return offer

        raise_cash = self._mortgage_to_build(ctx, aset)
        if raise_cash is not None:
            return raise_cash

        if ctx.env.players[ctx.pid].in_jail:
            jail = self._jail(ctx, aset)
            if jail is not None:
                return jail

        return _END_TURN if _END_TURN in aset else allowed[0]

    # ------------------------------------------------------------------
    # Even-build ladder
    # ------------------------------------------------------------------

    def _build(self, ctx, aset: set) -> Optional[int]:
        """Pick the next house, planning by WHOLE-GROUP level steps.

        v2 enforces even building, so a house is never an independent purchase:
        the real decision is "take this group from level L to L+1", which costs
        `k * house_price` where k is the number of deeds still at L.  Buying one
        house of a step we cannot finish is the classic v1 mistake -- it spends
        cash for a fraction of the rent and leaves the group stranded.  So we
        rank groups by the efficiency of the whole PLAN to BUILD_TARGET and
        refuse to start a step we cannot pay for in full.
        """
        env, pid = ctx.env, ctx.pid
        cash = env.players[pid].cash
        best = None

        for group in GROUP_ORDER:
            squares = _GROUP_SQUARES[group]
            props = [env.properties[sq] for sq in squares]
            if any(p.owner != pid or p.mortgaged for p in props):
                continue
            if not props[0].is_monopoly:
                continue
            levels = [p.houses for p in props]
            floor_level = min(levels)
            if floor_level >= 5:
                continue

            target = self.BUILD_TARGET
            if floor_level >= target:
                # Escalate only once the group is safe and cash is deep.
                target = self.BUILD_ESCALATE
                if target <= floor_level:
                    continue
            step_level = floor_level + 1
            house_price = _HOUSE_PRICE[squares[0]]
            k = sum(1 for lv in levels if lv == floor_level)
            lump = k * house_price          # cash to FINISH the current step
            if cash - lump < ctx.reserve:
                continue
            if step_level > self.BUILD_TARGET:
                deep = ctx.reserve + self.ESCALATE_LUMPS * len(squares) * house_price
                if cash - lump < deep:
                    continue
            if step_level == 5 and not ctx.hotels_ok(env):
                continue

            # Plan efficiency: rent gained per dollar of the whole remaining plan.
            plan_cost = sum(
                (min(target, 5) - lv) * house_price for lv in levels if lv < target
            )
            if plan_cost <= 0:
                continue
            gain = (
                GROUP_RENT_PER_ROUND[group][min(target, 5)]
                - ctx.current_group_rent(group)
            ) * ctx.opp_scale
            score = gain / plan_cost

            action = self._pick_house_action(ctx, group, step_level, floor_level, aset)
            if action is None:
                continue
            key = (score, GROUP_POWER[group])
            if best is None or key > best[0]:
                best = (key, action)

        return None if best is None else best[1]

    def _pick_house_action(
        self, ctx, group: str, step_level: int, floor_level: int, aset: set
    ) -> Optional[int]:
        """Which deed of the group gets this house: P(land) x MARGINAL rent."""
        env, pid = ctx.env, ctx.pid
        offset = OFFSETS["improve_hotel"] if step_level == 5 else OFFSETS["improve_house"]
        for square in ODD_HOUSE_ORDER[group][step_level]:
            if env.properties[square].houses != floor_level:
                continue
            action = offset + _REAL_INDEX[square]
            if action in aset:
                return action
        return None

    # ------------------------------------------------------------------
    # Mortgage / unmortgage
    # ------------------------------------------------------------------

    def _mortgage_to_build(self, ctx, aset: set) -> Optional[int]:
        """Raise cash for a build step, but never from a group we are developing.

        Mortgaging a deed inside a monopoly zeroes that square's rent AND makes
        it unimprovable (env._improve_actions requires `not prop.mortgaged`), so
        it cannibalises exactly the thing we are trying to fund.
        """
        if not self.MORTGAGE_TO_BUILD:
            return None
        env, pid = ctx.env, ctx.pid
        shortfall = self._build_shortfall(ctx)
        if shortfall <= 0:
            return None

        best = None
        for square in PROPERTY_IDS:
            prop = env.properties[square]
            if prop.owner != pid or prop.mortgaged or prop.houses:
                continue
            if ctx.is_development_group(square):
                continue
            action = OFFSETS["mortgage"] + _PROP_INDEX[square]
            if action not in aset:
                continue
            raised = _MORTGAGE_OF[square]
            lost = ctx.square_income(square)
            key = lost / max(raised, 1)     # rent lost per dollar raised, minimise
            if best is None or key < best[0]:
                best = (key, action)
        return None if best is None else best[1]

    def _build_shortfall(self, ctx) -> int:
        """Cash still needed to finish the best affordable-if-funded build step."""
        env, pid = ctx.env, ctx.pid
        cash = env.players[pid].cash
        worst = 0
        for group in GROUP_ORDER:
            squares = _GROUP_SQUARES[group]
            props = [env.properties[sq] for sq in squares]
            if any(p.owner != pid for p in props) or not props[0].is_monopoly:
                continue
            levels = [p.houses for p in props]
            floor_level = min(levels)
            if floor_level >= self.BUILD_TARGET:
                continue
            house_price = _HOUSE_PRICE[squares[0]]
            lump = sum(1 for lv in levels if lv == floor_level) * house_price
            need = int(lump + ctx.reserve - cash)
            if need > 0:
                worst = max(worst, need)
        return worst

    def _unmortgage(self, ctx, aset: set) -> Optional[int]:
        env, pid = ctx.env, ctx.pid
        cash = env.players[pid].cash
        best = None
        for square in PROPERTY_IDS:
            prop = env.properties[square]
            if prop.owner != pid or not prop.mortgaged:
                continue
            action = OFFSETS["unmortgage"] + _PROP_INDEX[square]
            if action not in aset:
                continue
            cost = int(_MORTGAGE_OF[square] * 1.1)
            if cash - cost < ctx.reserve + self.UNMORTGAGE_BUFFER:
                continue
            gain = ctx.square_income(square, assume_unmortgaged=True)
            if prop.is_monopoly:
                gain *= 3.0            # unblocks the whole group's build ladder
            key = gain / max(cost, 1)
            if best is None or key > best[0]:
                best = (key, action)
        return None if best is None else best[1]

    # ------------------------------------------------------------------
    # Debt rescue
    # ------------------------------------------------------------------

    def _rescue(self, ctx, aset: set) -> Optional[int]:
        """Liquidate for maximum cash per dollar of income destroyed.

        Ordering, cheapest damage first:
          1. mortgage a deed outside every monopoly we own
          2. sell houses off the least efficient developed group
          3. mortgage inside a monopoly
          4. sell a deed back to the bank (dominated by mortgage; last resort)
        """
        env, pid = ctx.env, ctx.pid
        tiers: dict[int, list] = {1: [], 2: [], 3: [], 4: []}

        for square in PROPERTY_IDS:
            prop = env.properties[square]
            if prop.owner != pid:
                continue
            action = OFFSETS["mortgage"] + _PROP_INDEX[square]
            if action in aset:
                raised = _MORTGAGE_OF[square]
                lost = ctx.square_income(square)
                tier = 3 if ctx.is_development_group(square) else 1
                tiers[tier].append((lost / max(raised, 1), -raised, action))
            action = OFFSETS["sell_prop"] + _PROP_INDEX[square]
            if action in aset:
                raised = _MORTGAGE_OF[square]
                lost = ctx.square_income(square) + ctx.block_income(square)
                tiers[4].append((lost / max(raised, 1), -raised, action))

        for square in REAL_ESTATE_IDS:
            prop = env.properties[square]
            if prop.owner != pid or prop.houses <= 0:
                continue
            house_price = _HOUSE_PRICE[square]
            if prop.houses == 5:
                action = OFFSETS["sell_hotel"] + _REAL_INDEX[square]
                new_level = MAX_HOUSES
            else:
                action = OFFSETS["sell_house"] + _REAL_INDEX[square]
                new_level = prop.houses - 1
            if action not in aset:
                continue
            raised = house_price // 2
            lost = (
                3.0
                * ctx.opp_scale
                * LANDINGS_PER_OPP_TURN[square]
                * (deed_rent(square, prop.houses) - deed_rent(square, new_level))
            )
            tiers[2].append((lost / max(raised, 1), -raised, action))

        for tier in (1, 2, 3, 4):
            if tiers[tier]:
                tiers[tier].sort()
                return tiers[tier][0][2]
        return None

    # ------------------------------------------------------------------
    # Auctions
    # ------------------------------------------------------------------

    def _auction(self, ctx, allowed: Sequence[int], aset: set) -> int:
        env, pid = ctx.env, ctx.pid
        square = env.auction_property_id
        if square is None or pid != env.auction_current_pid:
            return _AUCTION_PASS if _AUCTION_PASS in aset else allowed[0]

        price = _PRICE_OF[square]
        value = ctx.deed_value(square, pid)
        if value < price:
            # A deed we merely like is only worth having at a discount.
            ceiling = min(value, self.AUCTION_PLAIN_MULT * price)
        else:
            ceiling = min(value, self.AUCTION_MAX_MULT * price)
        ceiling = min(ceiling, env.players[pid].cash - self.AUCTION_RESERVE)
        if ceiling <= 0:
            return _AUCTION_PASS

        options = []
        for action, increment in AUCTION_ACTION_TO_INCREMENT.items():
            code = int(action)
            if code not in aset:
                continue
            total = env.auction_high_bid + increment
            if total <= ceiling:
                options.append((increment, code))
        if not options:
            return _AUCTION_PASS
        options.sort()
        # Smallest legal raise: win cheap.  The shipped agents always take the
        # largest increment available, which is how they end up overpaying.
        return options[0][1] if self.AUCTION_MIN_RAISE else options[-1][1]

    # ------------------------------------------------------------------
    # Jail
    # ------------------------------------------------------------------

    def _jail(self, ctx, aset: set) -> Optional[int]:
        """Sitting in jail is free under this ruleset -- a failed jail roll
        returns before the move, so no rent is paid, rent is still collected,
        and pre-roll building and trading still happen.  Pay only while the
        board is still worth circulating on."""
        if _USE_GOOJ in aset:
            return _USE_GOOJ           # dead code in v2: cards draw nothing
        if _PAY_BAIL not in aset:
            return None
        env, pid = ctx.env, ctx.pid
        cash = env.players[pid].cash
        if cash - JAIL_BAIL < ctx.reserve:
            return None
        early = env.round < self.JAIL_EARLY_ROUNDS and ctx.unowned_deeds > 0
        if early:
            return _PAY_BAIL
        if not self.JAIL_SIT:
            return _PAY_BAIL
        # Late game: pay only when the board is not dangerous.
        return _PAY_BAIL if ctx.danger_per_turn < JAIL_BAIL else None

    # ------------------------------------------------------------------
    # Trading -- 37-65% of all non-forced decisions
    # ------------------------------------------------------------------

    def _accept_offer(self, ctx) -> bool:
        env, pid = ctx.env, ctx.pid
        offer = env._incoming_trade(pid)
        if offer is None:
            return False

        got = offer.offered_prop
        gave = offer.requested_prop

        # Hard vetoes.  These override any arithmetic.
        if gave is not None:
            if ctx.completes_for(gave.square_id, offer.from_player):
                return False           # never hand anyone their monopoly
            if gave.is_monopoly:
                return False           # never break our own
        # The engine silently validates affordability, but an unaffordable leg
        # just makes the accept a no-op, so check it ourselves.
        sender = env.players[offer.from_player]
        if offer.cash_offered and sender.cash < offer.cash_offered:
            return False
        me = env.players[pid]
        if me.cash - offer.cash_requested < ctx.reserve * self.BUY_RESERVE_RELIEF:
            return False

        edge = float(offer.cash_offered) - float(offer.cash_requested)
        if got is not None:
            edge += ctx.deed_value(got.square_id, pid)
        if gave is not None:
            edge -= ctx.deed_value(gave.square_id, pid)
        return edge > self.TRADE_ACCEPT_MARGIN

    def _make_offer(self, ctx, aset: set) -> Optional[int]:
        """One offer per turn; pick the single best of three families.

        Exchange offers (deed for deed) are legal in v2 and are the only way to
        turn two half-groups into one whole one, so they are tried first.
        """
        if not self.TRADE_ENABLED:
            return None
        env, pid = ctx.env, ctx.pid
        if pid in env.pending_trades:
            return None

        best = None
        for target in ctx.targets:
            for cand in self._exchange_candidates(ctx, target, aset):
                if best is None or cand[0] > best[0]:
                    best = cand
            for cand in self._cash_candidates(ctx, target, aset):
                if best is None or cand[0] > best[0]:
                    best = cand
        if best is None or best[0] < self.TRADE_OFFER_MARGIN:
            return None
        return best[1]

    def _exchange_candidates(self, ctx, target: int, aset: set) -> Iterable[tuple]:
        env, pid = ctx.env, ctx.pid
        t_idx = ctx.other_index[target]
        base = OFFSETS["exch_trade"] + t_idx * _EXCH_STRIDE
        mine = [
            sq
            for sq in PROPERTY_IDS
            if env.properties[sq].owner == pid and env.properties[sq].houses == 0
        ]
        theirs = [
            sq
            for sq in PROPERTY_IDS
            if env.properties[sq].owner == target and env.properties[sq].houses == 0
        ]
        for want in theirs:
            my_gain = ctx.deed_value(want, pid)
            their_loss = ctx.deed_value(want, target)
            for give in mine:
                if ctx.completes_for(give, target):
                    continue           # never hand over their monopoly
                if env.properties[give].is_monopoly:
                    continue
                offer_idx = _PROP_INDEX[give]
                req_idx = _PROP_INDEX[want]
                req_raw = req_idx if req_idx < offer_idx else req_idx - 1
                action = base + offer_idx * (_N_PROPS - 1) + req_raw
                if action not in aset:
                    continue
                edge = my_gain - ctx.deed_value(give, pid)
                if edge <= self.TRADE_OFFER_MARGIN:
                    continue
                peer = ctx.deed_value(give, target) - their_loss
                if peer <= self.TRADE_PEER_MARGIN:
                    continue           # they would never take it
                yield (edge + 0.25 * peer, action)

    def _cash_candidates(self, ctx, target: int, aset: set) -> Iterable[tuple]:
        env, pid = ctx.env, ctx.pid
        t_idx = ctx.other_index[target]
        buy_base = OFFSETS["buy_trade"] + t_idx * _TRADE_STRIDE
        sell_base = OFFSETS["sell_trade"] + t_idx * _TRADE_STRIDE

        for square in PROPERTY_IDS:
            prop = env.properties[square]
            if prop.houses:
                continue
            idx = _PROP_INDEX[square]

            if prop.owner == target:
                value = ctx.deed_value(square, pid)
                their_value = ctx.deed_value(square, target)
                for level, multiplier in enumerate(TRADE_CASH_LEVELS):
                    action = buy_base + idx * _N_CASH_LEVELS + level
                    if action not in aset:
                        continue
                    cost = int(_PRICE_OF[square] * multiplier)
                    edge = value - cost
                    if edge <= self.TRADE_OFFER_MARGIN:
                        continue
                    if cost - their_value <= self.TRADE_PEER_MARGIN:
                        continue       # they lose by selling; do not waste the turn
                    # Prefer the cheapest level that still clears both bars.
                    yield (edge, action)

            elif prop.owner == pid:
                if prop.is_monopoly:
                    continue
                if ctx.completes_for(square, target):
                    continue
                value = ctx.deed_value(square, pid)
                their_value = ctx.deed_value(square, target)
                for level, multiplier in enumerate(TRADE_CASH_LEVELS):
                    if multiplier < self.JUNK_SELL_MULT:
                        continue
                    action = sell_base + idx * _N_CASH_LEVELS + level
                    if action not in aset:
                        continue
                    proceeds = int(_PRICE_OF[square] * multiplier)
                    edge = proceeds - value
                    if edge <= self.TRADE_OFFER_MARGIN:
                        continue
                    if their_value - proceeds <= self.TRADE_PEER_MARGIN:
                        continue
                    yield (edge, action)

    # ------------------------------------------------------------------
    # Evaluation, exported for reward shaping
    # ------------------------------------------------------------------

    def position_value(self, game: Any, player_id: int) -> float:
        """Dollar-denominated worth of a seat: cash + deed values + income.

        Offered as a POTENTIAL FUNCTION for shaping.  It is deliberately not the
        engine's `Player.net_worth`, whose shaping term was measured to be
        anti-correlated with winning (per-game total_reward +3.65..+4.48 while
        losing 40/40).
        """
        env = unwrap(game)
        pid = int(player_id)
        if env.players[pid].bankrupt:
            return 0.0
        ctx = _Context(self, env, pid)
        total = float(env.players[pid].cash)
        for square in PROPERTY_IDS:
            prop = env.properties[square]
            if prop.owner != pid:
                continue
            total += ctx.deed_value(square, pid, owned=True)
            if prop.mortgaged:
                total -= _MORTGAGE_OF[square] * 1.1
            if prop.is_real_estate and prop.houses:
                houses = 5 if prop.houses == 5 else prop.houses
                total += houses * _HOUSE_PRICE[square] * 0.5
                total += (
                    self.VALUE_HORIZON
                    * 3.0
                    * ctx.opp_scale
                    * LANDINGS_PER_OPP_TURN[square]
                    * (deed_rent(square, prop.houses) - deed_rent(square, 0))
                )
        return total


# ==========================================================================
# Per-decision derived state.  Built once per choose_action call.
# ==========================================================================


class _Context:
    """Everything the policy layers need, computed once and shared."""

    __slots__ = (
        "spine", "env", "pid", "opponents", "opp_scale", "targets",
        "other_index", "owned", "reserve", "danger_per_turn",
        "unowned_deeds", "_deed_cache",
    )

    def __init__(self, spine: Spine, env, pid: int) -> None:
        self.spine = spine
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
                spine.RESERVE_CAP,
                spine.RESERVE_BASE + spine.RESERVE_RENT_K * worst_rent,
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
        if env.houses_available >= self.spine.HOTEL_HOLD_HOUSES:
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
                target = self.spine.VALUE_LEVEL
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
        value = min(value, self.spine.VALUE_CAP_MULT * _PRICE_OF[square])
        self._deed_cache[key] = value
        return value

    def _deed_value(self, square: int, player: int) -> float:
        spine, env = self.spine, self.env
        horizon = spine.VALUE_HORIZON
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
            target = min(spine.VALUE_LEVEL, 5)
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
            target = min(spine.VALUE_LEVEL, 5)
            full = GROUP_RENT_PER_ROUND[colour][target] * self.opp_scale
            value += (
                spine.PARTIAL_WEIGHT
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
                target = min(spine.VALUE_LEVEL, 5)
                gain = (
                    GROUP_RENT_PER_ROUND[colour][target] * self.opp_scale
                    - self.current_group_rent(colour)
                )
                value += spine.BLOCK_WEIGHT * horizon * max(gain, 0.0)
                break
        return value


# ==========================================================================
# Compatibility shims and variants
# ==========================================================================


class SpineFixedAgent:
    """`FixedPolicyAgent`-shaped wrapper: `SpineFixedAgent(pid).choose_action(env)`.

    Upstream `train.py` and the six shipped personalities use this older shape.
    Provided so SPINE can be dropped in as a sparring opponent there without
    touching anything under reference/.
    """

    SPINE_CLASS = Spine

    def __init__(self, player_id: int, spine: Optional[Spine] = None) -> None:
        self.player_id = int(player_id)
        self.spine = spine if spine is not None else self.SPINE_CLASS()
        self.name = self.spine.name

    def choose_action(self, env, decision_seed: int = 0) -> int:
        return self.spine.choose_action(env, self.player_id, decision_seed)


def variant(name: str, **overrides: Any) -> type:
    """Build a Spine subclass with tunables overridden, for ablation sweeps.

        Cheap = variant("spine_cheap", AUCTION_MAX_MULT=1.0, RESERVE_BASE=400)
    """
    for key in overrides:
        if not hasattr(Spine, key):
            raise AttributeError(f"Spine has no tunable named {key!r}")
    return type(name, (Spine,), {"name": name, **overrides})
