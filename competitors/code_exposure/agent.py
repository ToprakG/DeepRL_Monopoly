"""Submission entrypoint -- NEMESIS.

A single, self-contained policy file. It imports nothing from this repository
and nothing from ``ASU_FROZEN_TEACHER``: every number it uses is derived here
from the simulator's own scoring rule. Copy this one file next to the engine
and it runs.

Why it is written this way
--------------------------
The engine ends a game two ways (``MonopolyEnv._check_game_over`` /
``MonopolyEnv.winner``): one player left standing, or the round cap is reached
and the greatest ``Player.net_worth()`` wins. That function is not list price
(``state.py:43``):

    unmortgaged deed          (price - 0)         * 2.5
    deed in a complete group  (price - 0)         * 5.0
    mortgaged deed            (price - mortgage)  * same multiplier
    h houses                  h * house_price * (1 + 0.5h)
    hotel (h = 5)             5 * house_price * 3.5
    cash                      face value

Five consequences drive every rule below, and each is stated where it is used:

1. **Every acquisition is accretive.** A dollar of cash spent on a deed at
   list price becomes $2.50 of score, and closing a colour group re-prices
   every deed already in it from 2.5x to 5.0x.
2. **Development is the best conversion on the board.** The n-th house costs
   ``house_price`` and adds ``(n - 0.5) * house_price`` of score: +0.5x for
   the first, +4.5x for the hotel. Nothing else pays 4.5:1.
3. **Every liquidation is expensive and unequally so.** Mortgaging an
   ordinary deed destroys 2.5 of score per dollar raised, selling it to the
   bank 5.0, breaking a hotel 11.0. Liquidating in action-id order is a large
   silent loss, so we liquidate in ratio order and only under duress.
4. **The house bank is finite and shared** (``houses_available = 32``).
   Houses bought are houses no opponent can buy, and upgrading to a hotel
   *returns four houses to the bank* (``env.py:411``). Holding four houses on
   cheap groups is simultaneously the best score-per-dollar and a hard denial.
5. **Jail is shelter, not a penalty.** A jailed player still gets the full
   pre-roll economy -- build, trade, mortgage -- and cannot be charged rent.
   The engine releases you after three turns anyway, so declining bail is a
   free option, not a sacrifice.

Speed
-----
Purely analytic: no environment copies, no rollouts, no tensors. Every
decision is O(number of deeds). Measured well under a millisecond, which
matters twice over -- the tournament disqualifies the slowest player in any
game that runs past ten minutes, and several opponents in this field spend
0.3-4 seconds per decision.

Signature compatibility
-----------------------
The reference agents disagree on the call shape:

    FixedPolicyAgent.choose_action(env)
    DDQNAgent.choose_action(state, env, allowed_actions)
    PPOAgent.choose_action(state, env, allowed_actions)

and an arena may call ``choose_action(game, player_id, seed)``.
``Agent.choose_action`` accepts all of those plus the two-argument
``choose_action(state, allowed_actions)`` form, locating the environment and
the legal-action list by structure rather than by position. If no environment
is reachable at all, it decodes what it can from the 300-value observation and
still returns a legal action -- the harness treats a raised exception as a
forfeit, so this path never raises.
"""

from __future__ import annotations

import json
import os
from typing import Any, Sequence

__all__ = ["Agent", "Nemesis", "Tuning", "choose_action", "make_agent"]


# ══════════════════════════════════════════════════════════════════════════
# Board tables
#
# Embedded rather than imported so this file has no dependency on the engine
# package name. `_verify_against_engine()` below cross-checks them when the
# engine happens to be importable, so a rules change is caught loudly instead
# of being played against silently.
# ══════════════════════════════════════════════════════════════════════════

# square: (price, mortgage, colour, house_price, rent levels)
_DEEDS: dict[int, tuple[int, int, str, int, tuple[int, ...]]] = {
    1:  (60,  30,  "brown",     50,  (2, 10, 30, 90, 160, 250)),
    3:  (60,  30,  "brown",     50,  (4, 20, 60, 180, 320, 450)),
    5:  (200, 100, "railroad",  0,   (25, 50, 100, 200)),
    6:  (100, 50,  "lightblue", 50,  (6, 30, 90, 270, 400, 550)),
    8:  (100, 50,  "lightblue", 50,  (6, 30, 90, 270, 400, 550)),
    9:  (120, 60,  "lightblue", 50,  (8, 40, 100, 300, 450, 600)),
    11: (140, 70,  "pink",      100, (10, 50, 150, 450, 625, 750)),
    12: (150, 75,  "utility",   0,   (4, 10)),
    13: (140, 70,  "pink",      100, (10, 50, 150, 450, 625, 750)),
    14: (160, 80,  "pink",      100, (12, 60, 180, 500, 700, 900)),
    15: (200, 100, "railroad",  0,   (25, 50, 100, 200)),
    16: (180, 90,  "orange",    100, (14, 70, 200, 550, 750, 950)),
    18: (180, 90,  "orange",    100, (14, 70, 200, 550, 750, 950)),
    19: (200, 100, "orange",    100, (16, 80, 220, 600, 800, 1000)),
    21: (220, 110, "red",       150, (18, 90, 250, 700, 875, 1050)),
    23: (220, 110, "red",       150, (18, 90, 250, 700, 875, 1050)),
    24: (240, 120, "red",       150, (20, 100, 300, 750, 925, 1100)),
    25: (200, 100, "railroad",  0,   (25, 50, 100, 200)),
    26: (260, 130, "yellow",    150, (22, 110, 330, 800, 975, 1150)),
    27: (260, 130, "yellow",    150, (22, 110, 330, 800, 975, 1150)),
    28: (150, 75,  "utility",   0,   (4, 10)),
    29: (280, 140, "yellow",    150, (24, 120, 360, 850, 1025, 1200)),
    31: (300, 150, "green",     200, (26, 130, 390, 900, 1100, 1275)),
    32: (300, 150, "green",     200, (26, 130, 390, 900, 1100, 1275)),
    34: (320, 160, "green",     200, (28, 150, 450, 1000, 1200, 1400)),
    35: (200, 100, "railroad",  0,   (25, 50, 100, 200)),
    37: (350, 175, "darkblue",  200, (35, 175, 500, 1100, 1300, 1500)),
    39: (400, 200, "darkblue",  200, (50, 200, 600, 1400, 1700, 2000)),
}

PROPERTY_IDS: tuple[int, ...] = tuple(sorted(_DEEDS))
REAL_ESTATE_IDS: tuple[int, ...] = tuple(
    s for s in PROPERTY_IDS if _DEEDS[s][2] not in ("railroad", "utility")
)

PRICE = {s: d[0] for s, d in _DEEDS.items()}
MORTGAGE = {s: d[1] for s, d in _DEEDS.items()}
COLOUR = {s: d[2] for s, d in _DEEDS.items()}
HOUSE_PRICE = {s: d[3] for s, d in _DEEDS.items()}
RENT = {s: d[4] for s, d in _DEEDS.items()}

GROUP_SQUARES: dict[str, tuple[int, ...]] = {}
for _s in PROPERTY_IDS:
    GROUP_SQUARES.setdefault(COLOUR[_s], []).append(_s)  # type: ignore[arg-type]
GROUP_SQUARES = {c: tuple(v) for c, v in GROUP_SQUARES.items()}
GROUP_OF = {s: GROUP_SQUARES[COLOUR[s]] for s in PROPERTY_IDS}
GROUP_SIZE = {s: len(GROUP_OF[s]) for s in PROPERTY_IDS}
IS_REAL_ESTATE = {s: COLOUR[s] not in ("railroad", "utility") for s in PROPERTY_IDS}

PROP_INDEX = {s: i for i, s in enumerate(PROPERTY_IDS)}
RE_INDEX = {s: i for i, s in enumerate(REAL_ESTATE_IDS)}

NUM_PLAYERS = 4
TRADE_CASH_LEVELS = (0.75, 1.0, 1.25)
AUCTION_BID_INCREMENTS = (1, 10, 50, 100)
MAX_HOUSES = 4
JAIL_BAIL = 50
JAIL_SQUARE = 10
GO_SALARY = 200

# Action-space offsets, computed exactly as `monopoly_game_engine.actions`
# computes them, from sizes that are fixed by the ruleset.
_N_PROP = len(PROPERTY_IDS)          # 28
_N_RE = len(REAL_ESTATE_IDS)         # 22
_N_CASH = len(TRADE_CASH_LEVELS)     # 3
_OTHERS = NUM_PLAYERS - 1            # 3

OFF: dict[str, int] = {}
_cur = 9  # nine binary actions occupy 0..8
OFF["binary"] = 0
for _name, _size in (
    ("mortgage", _N_PROP),
    ("unmortgage", _N_PROP),
    ("improve_house", _N_RE),
    ("improve_hotel", _N_RE),
    ("sell_house", _N_RE),
    ("sell_hotel", _N_RE),
    ("sell_prop", _N_PROP),
    ("buy_trade", _OTHERS * _N_PROP * _N_CASH),
    ("sell_trade", _OTHERS * _N_PROP * _N_CASH),
    ("exch_trade", _OTHERS * _N_PROP * (_N_PROP - 1)),
    ("auction", 1 + len(AUCTION_BID_INCREMENTS)),
):
    OFF[_name] = _cur
    _cur += _size
ACTION_SPACE_SIZE = _cur  # 2958

DO_NOTHING, END_TURN, ROLL_DICE, BUY_PROPERTY = 0, 1, 2, 3
USE_GOOJ_CARD, PAY_BAIL, DECLARE_BANKRUPT = 4, 5, 6
ACCEPT_TRADE, DECLINE_TRADE = 7, 8

AUCTION_PASS = OFF["auction"]
AUCTION_BIDS = tuple(
    (AUCTION_PASS + 1 + i, inc) for i, inc in enumerate(AUCTION_BID_INCREMENTS)
)

_TRADE_STRIDE = _N_PROP * _N_CASH          # 84
_EXCH_STRIDE = _N_PROP * (_N_PROP - 1)     # 756

#: P(sum of 2d6 == total), used for landing and rent expectations.
DICE_P = {t: (6 - abs(7 - t)) / 36.0 for t in range(2, 13)}

SOLO_MULT = 2.5
GROUP_MULT = 5.0

#: The observation stores cash as ``min(cash / 5000, 1)``, so any balance
#: at or over this decodes as exactly this.
_CASH_SCALE = 5000.0

PHASE_PRE_ROLL = "pre_roll"
PHASE_POST_ROLL = "post_roll"
PHASE_OUT_OF_TURN = "out_of_turn"
PHASE_AUCTION = "auction"


def _verify_against_engine() -> None:
    """Fail loudly at import if the engine's tables ever stop matching ours.

    Silence is the dangerous outcome here: a changed price table or a changed
    action layout would make every decision below wrong while still producing
    legal-looking actions.
    """
    try:  # pragma: no cover - depends on how the harness lays out sys.path
        from monopoly_game_engine.actions import OFFSETS as _E_OFF
        from monopoly_game_engine.constants import (
            PROPERTIES as _E_PROPS,
            PROPERTY_IDS as _E_IDS,
        )
    except Exception:
        return
    if tuple(_E_IDS) != PROPERTY_IDS:
        raise RuntimeError("NEMESIS: engine property list changed")
    for square, data in _E_PROPS.items():
        if data["price"] != PRICE[square] or data["mortgage"] != MORTGAGE[square]:
            raise RuntimeError(f"NEMESIS: engine price table changed at {square}")
        if tuple(data["rent"]) != RENT[square]:
            raise RuntimeError(f"NEMESIS: engine rent table changed at {square}")
    for name, value in _E_OFF.items():
        if OFF.get(name) != value:
            raise RuntimeError(f"NEMESIS: action offset '{name}' changed")


_verify_against_engine()


# ══════════════════════════════════════════════════════════════════════════
# Net worth in the engine's own units
# ══════════════════════════════════════════════════════════════════════════


def deed_worth(square: int, houses: int, mortgaged: bool, monopoly: bool) -> float:
    """``Property.calculate_net_worth`` evaluated on hypothetical attributes."""
    price = PRICE[square]
    base = (price - (MORTGAGE[square] if mortgaged else 0)) * (
        GROUP_MULT if monopoly else SOLO_MULT
    )
    if houses <= 0 or not IS_REAL_ESTATE[square]:
        return base
    return base + houses * HOUSE_PRICE[square] * (1.0 + 0.5 * houses)


def _group_worth(env, owner: int, squares: tuple[int, ...], *, monopoly: bool) -> float:
    total = 0.0
    props = env.properties
    for sq in squares:
        prop = props[sq]
        if prop.owner != owner:
            continue
        total += deed_worth(sq, prop.houses, prop.mortgaged, monopoly)
    return total


def _owns_group(env, owner: int, squares: tuple[int, ...]) -> bool:
    props = env.properties
    return all(props[sq].owner == owner for sq in squares)


def acquire_delta(env, pid: int, square: int) -> float:
    """Net worth ``pid`` gains by acquiring ``square``, ignoring what it costs.

    Closing a colour group re-prices every deed already held in it, so this is
    far larger than the deed's own worth at the moment a group completes --
    which is exactly the moment worth paying a premium for.
    """
    squares = GROUP_OF[square]
    props = env.properties
    completes = all(sq == square or props[sq].owner == pid for sq in squares)
    before = _group_worth(env, pid, squares, monopoly=False)
    after = 0.0
    for sq in squares:
        prop = props[sq]
        if sq != square and prop.owner != pid:
            continue
        # A deed only ever changes hands undeveloped: an auction deed is
        # unowned and the engine refuses to trade a built one.
        houses = 0 if sq == square else prop.houses
        after += deed_worth(sq, houses, prop.mortgaged, completes)
    return after - before


def disposal_delta(env, pid: int, square: int) -> float:
    """Net worth ``pid`` loses by giving ``square`` away.

    Every colour group has at least two deeds, so losing one always breaks the
    group and re-prices the rest back down to 2.5x.
    """
    squares = GROUP_OF[square]
    props = env.properties
    monopoly = _owns_group(env, pid, squares)
    before = _group_worth(env, pid, squares, monopoly=monopoly)
    after = 0.0
    for sq in squares:
        if sq == square:
            continue
        prop = props[sq]
        if prop.owner != pid:
            continue
        after += deed_worth(sq, prop.houses, prop.mortgaged, False)
    return before - after


def improve_delta(env, square: int, to_hotel: bool) -> float:
    """Net worth added by one build step, before the cash it costs."""
    prop = env.properties[square]
    now = prop.houses
    then = 5 if to_hotel else now + 1
    monopoly = _owns_group(env, prop.owner, GROUP_OF[square])
    return deed_worth(square, then, prop.mortgaged, monopoly) - deed_worth(
        square, now, prop.mortgaged, monopoly
    )


def mortgage_delta(env, square: int) -> float:
    """Net worth destroyed by mortgaging ``square`` (always positive)."""
    prop = env.properties[square]
    monopoly = _owns_group(env, prop.owner, GROUP_OF[square])
    return deed_worth(square, prop.houses, False, monopoly) - deed_worth(
        square, prop.houses, True, monopoly
    )


def rent_of(env, square: int, dice_total: int = 7) -> int:
    """Rent charged by ``square`` right now, mirroring ``Property.get_rent``."""
    prop = env.properties[square]
    if prop.mortgaged or prop.owner is None:
        return 0
    colour = COLOUR[square]
    owner = env.players[prop.owner]
    if colour == "railroad":
        return RENT[square][min(owner.railroads_owned() - 1, 3)]
    if colour == "utility":
        return RENT[square][0 if owner.utilities_owned() == 1 else 1] * dice_total
    if prop.houses == 0:
        base = RENT[square][0]
        return base * 2 if prop.is_monopoly else base
    return RENT[square][min(prop.houses, 5)]


# ══════════════════════════════════════════════════════════════════════════
# Tuning
#
# Every constant that a measurement could move lives here, so an ablation is
# one keyword argument rather than a patch.
# ══════════════════════════════════════════════════════════════════════════


class Tuning:
    __slots__ = (
        "reserve_floor", "reserve_expected", "reserve_worst", "reserve_cap",
        "buy_relief", "build_relief", "auction_relief", "trade_relief",
        "auction_fraction", "denial_fraction", "auction_step_fraction",
        "first_house_ratio", "rent_weight", "lock_weight", "lock_threshold",
        "block_weight", "trade_accept_margin", "endgame_span", "endgame_reserve",
        "jail_exposure", "jail_unowned", "offer_prior", "blocked_weight",
        "rail_weight", "propose_weight", "refusal_patience",
        "completion_weight", "surplus_weight", "early_reserve",
    )

    def __init__(self, **overrides: float) -> None:
        # Solvency reserve = floor + a*E[rent next roll] + b*worst reachable.
        # Measured: with a thin reserve this agent bankrupted in 11 of 16 games
        # against the strongest opponent here, at a median of round 40. A deed
        # is worth 2.5x what it costs, but a bankruptcy is worth zero, and most
        # games in this field are decided by elimination rather than the cap.
        #
        # Raised after the fair league put three ASU-descended agents at the
        # table. The evidence is split and both halves are recorded here:
        #   ASU-heavy tables, 128 games/arm  : +3.2 pp, +922 score
        #   six-rival tables, 160 paired     : +6.3 pp, +1476 score, z = +1.89
        #   four fast rivals only, 96/arm    : -2.1 pp,  -978 score
        # The two larger and more representative samples win, and bankruptcy
        # fell with them (53.8% -> 50.0%). Against a narrow, cheap field the
        # thinner reserve is still better -- so this is a field-dependent
        # setting, not a law, and the knobs stay exposed.
        self.reserve_floor = 150.0
        self.reserve_expected = 3.5
        self.reserve_worst = 1.20
        self.reserve_cap = 1600.0
        # Share of the reserve each spending family must respect. Development
        # is held to a quarter of it because a house is not consumption: it is
        # the only purchase that raises our score and our income together.
        self.buy_relief = 0.50
        # Measured: holding the climb to any fraction of the reserve cost more
        # than the bankruptcies it avoided. A bare monopoly is the largest
        # unconverted asset on the board, and the ladder above the first house
        # pays 1.5x, 2.5x, 3.5x and 4.5x on every dollar it takes.
        self.build_relief = 0.00
        self.auction_relief = 0.35
        self.trade_relief = 1.00
        # Auction ceiling as a share of what winning is worth to us, plus a
        # share of what it would be worth to the rival who wants it most.
        self.auction_fraction = 0.85
        # Denial is worth far more than the deed. In a field of net-worth
        # maximisers the deed that closes somebody's colour group is the
        # single largest swing on the board -- it re-prices their whole
        # group 2.5x -> 5.0x and unlocks their build ladder -- so taking
        # it away is priced at half of what they would have gained.
        # Measured on the hardest mixed table (us + aline + deniz + emir),
        # 40 paired games: 0.22 -> 37.5%, 0.50 -> 50.0%, 0.80 -> 45.0%.
        self.denial_fraction = 0.50
        self.auction_step_fraction = 0.10
        # Weight on a deed whose colour group an opponent has already entered.
        # Such a deed still books 2.5x list, but it can never carry a house,
        # and unbuilt rent is trivial: New York Avenue pays $16 alone against
        # $1,000 with a hotel. Pricing those deeds at face spread this agent
        # across every group on the board while the opponents concentrated,
        # and the concentrated players took twice as many monopolies.
        # Measured, monotonically, against the strongest opponent in the field:
        # 0.25 -> 15.6%, 0.45 -> 28.1%, 0.70 -> 31.2%, 1.00 -> 34.4% win rate
        # over 32 paired games. Discounting spoiled groups is the right idea
        # for a rent-driven scoring rule and the wrong one for this engine,
        # which pays 2.5x list for a deed whatever colour it is. The knob is
        # kept because the finding is opponent-dependent, not a law.
        self.blocked_weight = 1.00
        # Railroad rent scales with how many you hold (25 / 50 / 100 / 200),
        # which the net-worth delta alone does not see at all.
        self.rail_weight = 6.0
        # Closing a colour group does two things the net-worth delta only
        # counts one of: it re-prices the deeds already held from 2.5x to
        # 5.0x, and it unlocks the build ladder, which is the only route
        # that turns idle cash into score once the board is bought out
        # (around round 25). Instrumented games ended holding $4,500 of
        # cash scoring 1.0x with nothing legal to convert it into.
        self.completion_weight = 0.40
        # Measured and rejected, kept as a knob with its result recorded.
        # Cash above the reserve does score only 1.0x, and raising the auction
        # ceiling to spend it looks like the fix -- but paying more for the same
        # deed buys no extra score, it only moves the bankruptcy forward: at
        # 0.35 the win rate against the heuristic champion fell 40.6% -> 28.1%
        # and bankruptcies rose 46.9% -> 62.5% over 32 paired games.
        self.surplus_weight = 0.0
        # The first house pays only 0.5x on its own, but it is the rung that
        # unlocks 1.5x / 2.5x / 3.5x above it, so it is not ranked myopically.
        self.first_house_ratio = 1.80
        self.rent_weight = 0.9
        # Bonus for builds that drain the shared house bank while an opponent
        # still has somewhere to spend houses.
        self.lock_weight = 0.55
        self.lock_threshold = 10
        # Price on parting with a deed that blocks a rival monopoly, over
        # and above the score it books for us. Same table, 40 paired
        # games: 0.35 -> 37.5%, 0.80 -> 45.0%, 1.20 -> 50.0%, 1.60 -> 40.0%,
        # 2.20 -> 37.5%. Past the peak the agent refuses swaps that are
        # good for it, which is how an anti-goal becomes the goal.
        self.block_weight = 1.20
        self.trade_accept_margin = 1.0
        self.endgame_span = 60.0
        self.endgame_reserve = 0.75
        # Zero means never buy your way out. Jail costs a jailed player
        # nothing the engine charges for -- the whole pre-roll economy is
        # still available, rent cannot reach you, and the third turn
        # releases you anyway -- so the only cost is the deeds you do not
        # land on. Measured against the strongest opponent, never paying
        # bail moved the win rate from 12.5% to 21.9% over 32 paired games
        # and cut bankruptcies from 40.6% to 34.4%. The threshold is kept
        # so the rule can be turned back on if a field ever rewards tempo.
        self.jail_exposure = 0.0
        self.jail_unowned = 4
        # Rezervi, tahtada hala sahipsiz tapu varken olcekler. Erken
        # oyunda kira neredeyse sifirdir ve tapular ucuzdur; gec oyunda
        # tam tersi. Tek bir tehdit-orantili rezerv iki rejimi de ayni
        # sayiyla idare etmeye calisiyor.
        self.early_reserve = 1.0
        self.offer_prior = 0.45
        # Scales every outgoing trade proposal, so the whole branch can be
        # ablated to zero and measured against itself.
        self.propose_weight = 1.0
        # Offers to a counterparty that has refused this many in a row with
        # no acceptance are abandoned. Without it the branch is unkillable:
        # a completing trade re-prices a whole colour group, so its edge
        # stays large enough to outrank every build even after the
        # smoothed acceptance rate has collapsed. Instrumented against the
        # search opponent, offers were 46.6% of all our decisions and none
        # were ever accepted.
        #
        # The number wants to be large, which was not the expectation. Over 96
        # paired games on four mixed tables: 6 -> 60.4%, 10 -> 60.4%,
        # 15 -> 63.5%, 25 -> 65.6%, no cap at all -> 65.6%. A proposal does not
        # end the phase -- the engine asks again and the next-best action is
        # taken -- so an unaccepted offer costs decisions, not moves, and the
        # smoothed rate already suppresses it. The cap is kept at a value that
        # measured identically to removing it, so no field can lock the agent
        # into a branch that has stopped paying.
        self.refusal_patience = 25.0
        for key, value in overrides.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown tuning key {key!r}")
            setattr(self, key, value)


DEFAULT_TUNING = Tuning()

#: Prior that a cash-for-deed offer at each ``TRADE_CASH_LEVELS`` multiplier is
#: accepted. The rule that makes this branch profitable
#: (``fixed_accept_trade_decision``) needs ``price_offered - price_requested``
#: to be non-negative, which 0.75x never is and 1.0x exactly is.
_OFFER_LEVEL_PRIOR = (0.15, 1.00, 1.15)


# ══════════════════════════════════════════════════════════════════════════
# The policy
# ══════════════════════════════════════════════════════════════════════════


class Nemesis:
    """Deterministic, analytic, net-worth-exact policy for one seat."""

    policy_id = "nemesis-v1"

    def __init__(self, player_id: int = 0, tuning: Tuning | None = None) -> None:
        self.player_id = int(player_id)
        self.t = tuning or DEFAULT_TUNING
        # Offer bookkeeping. Opponents differ enormously in what they accept
        # -- the hybrid-PPO family takes any offer whose list-price balance is
        # non-negative, the search agents take almost nothing -- so the
        # acceptance rate is learned per opponent during the game instead of
        # assumed.
        self._offers: list[int] = [0, 0, 0, 0]
        self._accepts: list[int] = [0, 0, 0, 0]
        self._pending: tuple[int, int] | None = None  # (target, square)

    # ── acceptance model ────────────────────────────────────────────────

    def _settle_pending(self, env) -> None:
        """Score the previous offer before making a new one."""
        if self._pending is None:
            return
        target, square = self._pending
        owner = env.properties[square].owner
        if owner == self.player_id:
            self._accepts[target] += 1
        self._pending = None

    def _p_accept(self, target: int) -> float:
        made = self._offers[target]
        if made == 0:
            return self.t.offer_prior
        taken = self._accepts[target]
        if taken == 0 and made >= self.t.refusal_patience:
            return 0.0  # this opponent does not trade; stop paying to find out
        # Laplace-smoothed so one refusal does not zero the branch out.
        return (taken + 1.0) / (made + 2.0)

    # ── position summary ────────────────────────────────────────────────

    def _exposure(self, env) -> tuple[float, float]:
        """(expected, worst) rent payable on our next roll from where we are.

        Only the eleven squares a 2d6 roll can reach matter; insuring against
        the worst rent on the whole board is the mistake that makes an agent
        hoard cash it should have converted into score.
        """
        me = env.players[self.player_id]
        pos = me.position
        props = env.properties
        expected = 0.0
        worst = 0.0
        for total, prob in DICE_P.items():
            square = (pos + total) % 40
            prop = props.get(square)
            if prop is None or prop.owner is None or prop.owner == self.player_id:
                continue
            if prop.mortgaged or env.players[prop.owner].bankrupt:
                continue
            rent = rent_of(env, square, total)
            expected += prob * rent
            if rent > worst:
                worst = rent
        return expected, worst

    def _endgame_weight(self, env) -> float:
        cap = float(getattr(env, "max_rounds", 200) or 200)
        left = cap - float(getattr(env, "round", 0))
        span = self.t.endgame_span
        if left >= span:
            return 0.0
        return 1.0 if left <= 0.0 else 1.0 - left / span

    def _reserve(self, env) -> float:
        """Cash held back against the rent that can arrive before we act again.

        On an empty board almost every landing owes nothing, so this is the
        floor and the agent invests freely; it rises on its own as opponents
        develop. Near the cap it is deliberately relaxed: cash scores 1.0x and
        a deed scores 2.5x, so hoarding into the final rounds is a direct loss.
        """
        expected, worst = self._exposure(env)
        t = self.t
        reserve = t.reserve_floor + t.reserve_expected * expected + t.reserve_worst * worst
        reserve = min(reserve, t.reserve_cap)
        if t.early_reserve != 1.0 and self._unowned_deeds(env) > 0:
            reserve *= t.early_reserve
        return reserve * (1.0 - t.endgame_reserve * self._endgame_weight(env))

    def _affordable(self, env, cost: float, reserve: float) -> bool:
        """Can we pay ``cost`` and still be holding ``reserve`` afterwards?

        The observation clips cash at its $5,000 ceiling, so a saturated
        balance is known only as "at least $5,000". Treating that as exactly
        $5,000 is the safe direction -- it can refuse a purchase we could
        actually afford, never accept one we cannot.
        """
        seat = env.players[self.player_id]
        purse = float(seat.cash)
        if getattr(seat, "cash_saturated", False):
            purse = max(purse, _CASH_SCALE)
        return purse - float(cost) >= reserve

    def _live_rivals(self, env) -> list[int]:
        return [
            p.player_id
            for p in env.players
            if p.player_id != self.player_id and not p.bankrupt
        ]

    def _unowned_deeds(self, env) -> int:
        props = env.properties
        return sum(1 for sq in PROPERTY_IDS if props[sq].owner is None)

    def _gift_value(self, env, square: int) -> float:
        """What handing ``square`` to the hungriest rival would give them.

        This is the price of parting with the deed that is somebody's third
        orange, and it is not symmetric with what the deed books for us.
        """
        props = env.properties
        best = 0.0
        for rival in self._live_rivals(env):
            if props[square].owner == rival:
                continue
            gain = acquire_delta(env, rival, square)
            if all(
                sq == square or props[sq].owner == rival for sq in GROUP_OF[square]
            ):
                gain += self._completion_bonus(env, rival, square)
            if gain > best:
                best = gain
        return best

    def _denial_value(self, env, square: int) -> float:
        """What taking ``square`` costs whoever holds it, or would hold it."""
        prop = env.properties[square]
        owner = prop.owner
        if owner is None:
            return self._gift_value(env, square)
        if owner == self.player_id or env.players[owner].bankrupt:
            return 0.0
        return disposal_delta(env, owner, square)

    def _outlook(self, env, pid: int, square: int) -> float:
        """How much of a deed's booked score is still strategically live.

        Houses need the complete colour group -- the engine refuses to build
        otherwise -- so a deed in a group an opponent has already entered can
        never become a weapon, however much net worth it books. Railroads and
        utilities are exempt: their rent scales with how many you hold, so a
        partial holding still earns.
        """
        if not IS_REAL_ESTATE[square]:
            return 1.0
        props = env.properties
        for sq in GROUP_OF[square]:
            if sq == square:
                continue
            owner = props[sq].owner
            if owner is not None and owner != pid:
                return self.t.blocked_weight
        return 1.0

    def _rail_bonus(self, env, pid: int, square: int) -> float:
        """Extra rent unlocked across the set by adding one railroad/utility."""
        colour = COLOUR[square]
        if colour not in ("railroad", "utility"):
            return 0.0
        props = env.properties
        held = sum(1 for sq in GROUP_SQUARES[colour] if props[sq].owner == pid)
        levels = RENT[square]
        if colour == "railroad":
            before = levels[min(max(held - 1, 0), 3)] * held
            after = levels[min(held, 3)] * (held + 1)
        else:
            before = levels[0 if held <= 1 else 1] * 7 * held
            after = levels[0 if held == 0 else 1] * 7 * (held + 1)
        return self.t.rail_weight * max(0.0, after - before)

    def _group_build_potential(self, env, square: int) -> float:
        """Score the group could still carry if fully developed, less its worth now.

        Only meaningful for a group we would own outright: the engine refuses
        to build otherwise. Railroads and utilities carry no houses at all.
        """
        squares = GROUP_OF[square]
        if not IS_REAL_ESTATE[square]:
            return 0.0
        total = 0.0
        for sq in squares:
            prop = env.properties[sq]
            houses = prop.houses
            total += deed_worth(sq, 5, prop.mortgaged, True) - deed_worth(
                sq, houses, prop.mortgaged, True
            )
            total -= (5 - houses) * HOUSE_PRICE[sq]
        return max(0.0, total)

    def _completion_bonus(self, env, pid: int, square: int) -> float:
        """Build potential unlocked if taking ``square`` closes the group."""
        if self.t.completion_weight <= 0.0 or not IS_REAL_ESTATE[square]:
            return 0.0
        props = env.properties
        for sq in GROUP_OF[square]:
            if sq != square and props[sq].owner != pid:
                return 0.0
        return self.t.completion_weight * self._group_build_potential(env, square)

    def _acquire_value(self, env, square: int) -> float:
        """What taking ``square`` is worth to us as a decision quantity.

        The booked net worth, discounted where the group is already spoiled,
        plus the rent a set unlocks, plus a share of what keeping it away from
        the rival who wants it most is worth.
        """
        pid = self.player_id
        value = acquire_delta(env, pid, square) * self._outlook(env, pid, square)
        value += self._rail_bonus(env, pid, square)
        value += self._completion_bonus(env, pid, square)
        return value + self.t.denial_fraction * self._denial_value(env, square)

    def _house_lock_bonus(self, env, square: int) -> float:
        """Reward for draining the shared house bank while it still bites.

        ``houses_available`` is one pool of 32 for the table. Buying the
        cheap groups out consumes the most houses per dollar, and an opponent
        holding a monopoly with an empty bank simply cannot develop it. The
        bonus is zero once the bank is comfortable or once no opponent has
        anywhere to spend houses, so it never distorts ordinary play.
        """
        t = self.t
        available = int(getattr(env, "houses_available", 32))
        if available > t.lock_threshold:
            return 0.0
        if not self._rival_can_build(env):
            return 0.0
        # Cheaper houses drain more of the bank per dollar spent.
        return t.lock_weight * (200.0 / max(HOUSE_PRICE[square], 1))

    def _rival_can_build(self, env) -> bool:
        props = env.properties
        for rival in self._live_rivals(env):
            for squares in GROUP_SQUARES.values():
                if len(squares) < 2 or not IS_REAL_ESTATE[squares[0]]:
                    continue
                if all(props[sq].owner == rival for sq in squares) and any(
                    props[sq].houses < MAX_HOUSES and not props[sq].mortgaged
                    for sq in squares
                ):
                    return True
        return False

    # ── auction ─────────────────────────────────────────────────────────

    def _auction(self, env, legal: set[int]) -> int:
        square = getattr(env, "auction_property_id", None)
        if square is None:
            return AUCTION_PASS
        t = self.t
        pid = self.player_id

        own = self._acquire_value(env, square)
        ceiling = t.auction_fraction * own
        # Near the cap a deed is simply 2.5x its price and cash is 1.0x, so
        # the ceiling relaxes towards the full conversion value.
        ceiling += (own - ceiling) * self._endgame_weight(env)
        surplus = float(env.players[pid].cash) - self._reserve(env)
        if surplus > 0.0:
            ceiling *= 1.0 + t.surplus_weight * min(1.0, surplus / 1500.0)
        ceiling = min(ceiling, own)

        high = float(getattr(env, "auction_high_bid", 0) or 0)
        reserve = self._reserve(env) * t.auction_relief
        room = ceiling - high
        if room <= 0:
            return AUCTION_PASS

        # Raise in the smallest step that is not a token increase: bidding +1
        # against three opponents is cheap but can cost a hundred decisions,
        # and in this field the opponents are the slow ones.
        floor_step = max(1.0, t.auction_step_fraction * room)
        affordable = [
            (inc, action)
            for action, inc in AUCTION_BIDS
            if action in legal
            and high + inc <= ceiling
            and self._affordable(env, high + inc, reserve)
        ]
        if not affordable:
            return AUCTION_PASS
        affordable.sort()
        for inc, action in affordable:
            if inc >= floor_step:
                return action
        return affordable[-1][1]

    # ── forced liquidation ──────────────────────────────────────────────

    def _liquidate(self, env, legal: set[int]) -> int:
        """Raise cash in strict order of score destroyed per dollar raised.

        Mortgaging an ordinary deed costs 2.5 per dollar, mortgaging one
        inside a monopoly 5.0, selling a deed to the bank 5.0, selling the
        first house 3.0 and breaking a hotel 11.0. Liquidating in action-id
        order -- which is what a masked network does -- pays several times
        over for the same cash.
        """
        pid = self.player_id
        props = env.properties
        options: list[tuple[float, float, int]] = []

        for square in PROPERTY_IDS:
            prop = props[square]
            if prop.owner != pid:
                continue
            index = PROP_INDEX[square]
            monopoly = _owns_group(env, pid, GROUP_OF[square])

            action = OFF["mortgage"] + index
            if action in legal:
                cash = float(MORTGAGE[square])
                loss = mortgage_delta(env, square)
                options.append((loss / max(cash, 1.0), -cash, action))

            action = OFF["sell_prop"] + index
            if action in legal:
                cash = float(MORTGAGE[square])
                loss = deed_worth(square, prop.houses, prop.mortgaged, monopoly)
                options.append((loss / max(cash, 1.0), -cash, action))

            if not IS_REAL_ESTATE[square] or prop.houses <= 0:
                continue
            re_index = RE_INDEX[square]
            cash = float(HOUSE_PRICE[square] // 2)
            if prop.houses == 5:
                action = OFF["sell_hotel"] + re_index
                then = MAX_HOUSES
            else:
                action = OFF["sell_house"] + re_index
                then = prop.houses - 1
            if action in legal:
                loss = deed_worth(square, prop.houses, prop.mortgaged, monopoly) - deed_worth(
                    square, then, prop.mortgaged, monopoly
                )
                options.append((loss / max(cash, 1.0), -cash, action))

        if not options:
            return DECLARE_BANKRUPT if DECLARE_BANKRUPT in legal else min(legal)
        options.sort()
        return options[0][2]

    # ── jail ────────────────────────────────────────────────────────────

    def _jail(self, env, legal: set[int]) -> int | None:
        """Leave jail only while the board is still cheap and still open.

        Staying costs nothing the engine charges for: a jailed player keeps
        the whole pre-roll economy and is released after three turns anyway.
        What it costs is tempo -- the deeds we do not land on -- so the test
        is whether there is anything left to land on.
        """
        me = env.players[self.player_id]
        if not me.in_jail:
            return None
        expected, _worst = self._exposure(env)
        unowned = self._unowned_deeds(env)
        want_out = expected < self.t.jail_exposure and unowned >= self.t.jail_unowned
        if not want_out:
            return None
        if USE_GOOJ_CARD in legal:
            return USE_GOOJ_CARD  # free, so never worth holding
        if PAY_BAIL in legal and self._affordable(env, JAIL_BAIL, self._reserve(env)):
            return PAY_BAIL
        return None

    # ── incoming trades ─────────────────────────────────────────────────

    def _incoming(self, env):
        """(sender, offer) for the offer aimed at us, or None.

        Reimplemented rather than calling ``env._incoming_trade_entry`` so a
        private-method rename cannot silently turn every reply into a decline;
        the engine's method is used when it exists and this is the fallback.
        """
        finder = getattr(env, "_incoming_trade_entry", None)
        if finder is not None:
            try:
                return finder(self.player_id)
            except Exception:
                pass
        order = list(getattr(env, "turn_order", range(NUM_PLAYERS)))
        try:
            start = order.index(self.player_id)
        except ValueError:
            return None
        for step in range(1, len(order)):
            sender = order[(start + step) % len(order)]
            offer = env.pending_trades.get(sender)
            if offer is not None and offer.to_player == self.player_id:
                return sender, offer
        return None

    def _reply(self, env, legal: set[int]) -> int | None:
        if ACCEPT_TRADE not in legal:
            return None
        entry = self._incoming(env)
        if entry is None:
            return DECLINE_TRADE if DECLINE_TRADE in legal else None
        _sender, offer = entry
        pid = self.player_id

        delta = float(offer.cash_offered) - float(offer.cash_requested)
        if offer.offered_prop is not None:
            delta += self._acquire_value(env, offer.offered_prop.square_id)
        if offer.requested_prop is not None:
            square = offer.requested_prop.square_id
            delta -= disposal_delta(env, pid, square)
            # Handing over the deed that blocks somebody's third orange is
            # worth more than the score it books, so it is priced separately.
            delta -= self.t.block_weight * self._gift_value(env, square)

        if delta <= self.t.trade_accept_margin:
            return DECLINE_TRADE
        reserve = self._reserve(env) * self.t.trade_relief
        if not self._affordable(env, float(offer.cash_requested), reserve):
            return DECLINE_TRADE
        return ACCEPT_TRADE

    # ── outgoing trades ─────────────────────────────────────────────────

    def _offer_candidates(self, env, legal: set[int]) -> list[tuple[float, float, int, int, int]]:
        """``(score, cost, action, target, square)`` for every offer worth making.

        The cash-for-deed offer is the largest single edge available against
        part of this field. ``fixed_accept_trade_decision`` -- the accept rule
        inside the hybrid-PPO agents -- takes any offer whose list-price
        balance is non-negative, and a purchase at exactly list price balances
        to zero. That converts $1 of cash into $2.50 of score *and* removes
        $2.50 from the seller, a four-to-one swing per trade. Levels are tried
        cheapest-first and the acceptance rate is measured per opponent, so
        against the agents that refuse everything this costs one action slot
        and nothing else.
        """
        pid = self.player_id
        props = env.properties
        cash = float(env.players[pid].cash)
        reserve = self._reserve(env) * self.t.trade_relief
        others = [i for i in range(NUM_PLAYERS) if i != pid]
        out: list[tuple[float, float, int, int, int]] = []

        # Deeds we could part with, priced once rather than once per pairing.
        mine: list[tuple[int, float]] = []
        for square in PROPERTY_IDS:
            prop = props[square]
            if prop.owner != pid or prop.houses or prop.is_monopoly:
                continue
            loss = disposal_delta(env, pid, square) + self.t.block_weight * (
                self._gift_value(env, square)
            )
            mine.append((square, loss))

        for t_idx, target in enumerate(others):
            if env.players[target].bankrupt:
                continue
            p_accept = self._p_accept(target)
            buy_base = OFF["buy_trade"] + t_idx * _TRADE_STRIDE
            exch_base = OFF["exch_trade"] + t_idx * _EXCH_STRIDE

            for square in PROPERTY_IDS:
                prop = props[square]
                if prop.owner != target or prop.houses:
                    continue
                gain = self._acquire_value(env, square)
                index = PROP_INDEX[square]
                for level, multiplier in enumerate(TRADE_CASH_LEVELS):
                    action = buy_base + index * _N_CASH + level
                    if action not in legal:
                        continue
                    cost = float(int(PRICE[square] * multiplier))
                    if cost > cash - reserve:
                        continue
                    edge = gain - cost
                    if edge <= 0:
                        continue
                    # Level priors, not a single per-target rate: the accept
                    # rule that makes this branch pay needs the list-price
                    # balance to be non-negative, so a 0.75x lowball is the
                    # one price a rational counterparty must refuse.
                    prior = _OFFER_LEVEL_PRIOR[level]
                    score = prior * p_accept * edge / max(cost, 1.0)
                    out.append((score, cost, action, target, square))

            # Deed-for-deed swaps: only when the counterparty also gains, or
            # nobody rational takes it and the slot is wasted.
            if not mine:
                continue
            for want in PROPERTY_IDS:
                if props[want].owner != target or props[want].houses:
                    continue
                want_gain = self._acquire_value(env, want)
                if want_gain <= 0:
                    continue
                req_index = PROP_INDEX[want]
                target_want_loss = disposal_delta(env, target, want)
                for give, loss in mine:
                    # Their gain is priced with the same completion term as
                    # ours. A swap that closes a colour group for both sides is
                    # the only kind a disciplined opponent ever takes, and
                    # rating their side at bare net worth hides exactly those.
                    counter = acquire_delta(env, target, give)
                    if all(
                        sq == give or props[sq].owner == target
                        for sq in GROUP_OF[give]
                    ):
                        counter += self.t.completion_weight * (
                            self._group_build_potential(env, give)
                        )
                    counter -= target_want_loss
                    edge = want_gain - loss
                    if edge <= 0:
                        continue
                    give_index = PROP_INDEX[give]
                    if give_index == req_index:
                        continue
                    raw = req_index if req_index < give_index else req_index - 1
                    action = exch_base + give_index * (_N_PROP - 1) + raw
                    if action not in legal:
                        continue
                    if counter <= 0:
                        continue
                    # Per dollar of capital moved, so a swap competes with a
                    # house on the same scale. Scoring the raw edge instead put
                    # trade proposals above every build: they took 37.8% of all
                    # decisions in an instrumented run that built 14 houses in
                    # eight games.
                    ratio = p_accept * edge / max(float(PRICE[give]), 1.0)
                    out.append((ratio, 0.0, action, target, want))
        return out

    # ── investment ──────────────────────────────────────────────────────

    def _investments(self, env, legal: set[int]) -> list[tuple[float, int, int, int]]:
        """``(score-per-dollar, action, target, square)`` over every way to spend.

        Ranked per dollar rather than per action: within one phase the agent
        keeps being asked, so it takes the best conversion first and the next
        one after that, which is what maximises score out of a fixed purse.
        """
        pid = self.player_id
        props = env.properties
        t = self.t
        reserve = self._reserve(env)
        rivals = max(1, len(self._live_rivals(env)))
        out: list[tuple[float, int, int, int]] = []

        # Redeeming a mortgage buys back 2.5x (or 5.0x) of score for 1.1x of
        # the mortgage value and restores the deed's rent: 1.27 per dollar on
        # an ordinary deed, 3.5 inside a monopoly.
        for square in PROPERTY_IDS:
            action = OFF["unmortgage"] + PROP_INDEX[square]
            if action not in legal:
                continue
            cost = float(int(MORTGAGE[square] * 1.1))
            gain = mortgage_delta(env, square) - cost
            if gain <= 0 or not self._affordable(env, cost, reserve):
                continue
            out.append((gain / max(cost, 1.0), action, -1, square))

        build_reserve = reserve * t.build_relief
        lock_active = self._house_lock_active(env)
        for square in REAL_ESTATE_IDS:
            re_index = RE_INDEX[square]
            cost = float(HOUSE_PRICE[square])
            prop = props[square]

            action = OFF["improve_house"] + re_index
            if action in legal and self._affordable(env, cost, build_reserve):
                level = prop.houses + 1
                ratio = (level - 0.5) if level >= 2 else t.first_house_ratio
                ratio += t.rent_weight * self._rent_ratio(env, square, level, rivals)
                ratio += self._house_lock_bonus(env, square)
                out.append((ratio, action, -1, square))

            action = OFF["improve_hotel"] + re_index
            if action in legal and self._affordable(env, cost, build_reserve):
                # A hotel hands four houses back to the shared bank
                # (env.py:411). While the bank is the thing keeping an
                # opponent's monopoly bare, that refund is worth more to them
                # than the extra 4.5x is to us.
                if not lock_active:
                    ratio = 4.5 + t.rent_weight * self._rent_ratio(env, square, 5, rivals)
                    out.append((ratio, action, -1, square))

        # A monopoly with no houses on it is the largest unconverted asset we
        # own; nothing proposed to an opponent competes with finishing it.
        if t.propose_weight > 0.0 and not self._has_bare_monopoly(env):
            for score, cost, action, target, square in self._offer_candidates(env, legal):
                out.append((score * t.propose_weight, action, target, square))

        out.sort(key=lambda item: (-item[0], item[1]))
        return out

    def _has_bare_monopoly(self, env) -> bool:
        pid = self.player_id
        props = env.properties
        cash = float(env.players[pid].cash)
        for squares in GROUP_SQUARES.values():
            if not IS_REAL_ESTATE[squares[0]]:
                continue
            if not all(props[sq].owner == pid for sq in squares):
                continue
            if any(
                props[sq].houses < MAX_HOUSES
                and not props[sq].mortgaged
                and cash >= HOUSE_PRICE[sq]
                for sq in squares
            ):
                return True
        return False

    def _house_lock_active(self, env) -> bool:
        available = int(getattr(env, "houses_available", 32))
        return available <= self.t.lock_threshold and self._rival_can_build(env)

    def _rent_ratio(self, env, square: int, level: int, rivals: int) -> float:
        """Rent bought per dollar of house price, as a per-dollar side term."""
        levels = RENT[square]
        now = env.properties[square].houses
        before = levels[min(now, 5)] if now else levels[0] * 2
        after = levels[min(level, 5)]
        land = DICE_P.get(7, 0.16)  # a deed is landed on roughly once per lap
        return land * rivals * (after - before) / max(HOUSE_PRICE[square], 1)

    # ── entry point ─────────────────────────────────────────────────────

    def act(self, env, legal_list: Sequence[int]) -> int:
        pid = self.player_id
        if not legal_list:
            return END_TURN
        if len(legal_list) == 1:
            return int(legal_list[0])
        legal = set(int(a) for a in legal_list)
        phase = getattr(env, "phase", PHASE_PRE_ROLL)

        if phase == PHASE_AUCTION:
            return self._auction(env, legal)

        if getattr(env, "debt_player", None) == pid:
            return self._liquidate(env, legal)

        if phase == PHASE_POST_ROLL and not getattr(env, "has_rolled", False):
            jail = self._jail(env, legal)
            if jail is not None:
                return jail
            return ROLL_DICE if ROLL_DICE in legal else int(legal_list[0])

        if phase == PHASE_POST_ROLL:
            if BUY_PROPERTY in legal:
                square = env.players[pid].position
                price = float(PRICE[square])
                gain = self._acquire_value(env, square) - price
                relief = self._reserve(env) * self.t.buy_relief
                # Buying is the cheapest acquisition there is: no auction to
                # lose, no counterparty to refuse. Near the cap the reserve
                # stops mattering, because cash that survives to the cap
                # scores 1.0x against the deed's 2.5x.
                if gain > 0 and self._affordable(env, price, relief):
                    return BUY_PROPERTY
            # Ending the turn on an unowned deed opens the auction
            # (env.py:564), where our ceiling is the highest at the table.
            return END_TURN if END_TURN in legal else int(legal_list[0])

        if phase in (PHASE_PRE_ROLL, PHASE_OUT_OF_TURN):
            self._settle_pending(env)
            reply = self._reply(env, legal)
            if reply is not None:
                return reply
            if phase == PHASE_PRE_ROLL:
                jail = self._jail(env, legal)
                if jail is not None:
                    return jail
            plans = self._investments(env, legal)
            if plans:
                _score, action, target, square = plans[0]
                if target >= 0:
                    self._offers[target] += 1
                    self._pending = (target, square)
                return action
            return END_TURN if END_TURN in legal else int(legal_list[0])

        return int(legal_list[0])


# ══════════════════════════════════════════════════════════════════════════
# Shadow board
#
# The tournament does not hand over the live engine object. A decision arrives
# as a read-only state carrying `vector` -- the same 300 floats
# `monopoly_game_engine.state.build_state_vector` produces, arranged so that
# the acting seat comes first -- plus a board view, the legal action ids and a
# decision seed.
#
# Those 300 values are lossless for everything the policy above reads, with one
# exception: cash is stored as `min(cash / 5000, 1)`, so any balance at or over
# $5,000 decodes as exactly $5,000. That direction is safe -- it can only make
# the agent more careful than it needs to be with a very large purse -- and the
# board view is used to correct it when it carries the real number.
#
# Decoding into an object that quacks like `MonopolyEnv` rather than writing a
# second policy is deliberate: one set of rules is measured, and the local
# tournament can run the same decision through both front ends and assert they
# agree.
# ══════════════════════════════════════════════════════════════════════════

_STATE_DIM = 300
_PHASES = (PHASE_PRE_ROLL, PHASE_POST_ROLL, PHASE_OUT_OF_TURN, PHASE_AUCTION)

# Offsets into the observation, in the order `build_state_vector` writes them.
_V_PLAYERS = 0        # 4 seats x (position, cash, in_jail, gooj)
_V_PROPS = 16         # 28 deeds x (owner one-hot 5, mortgaged, monopoly, houses)
_V_PHASE = 240
_V_WHOSE_TURN = 244
_V_ACTIVE = 248
_V_HAS_ROLLED = 252
_V_DOUBLES = 253
_V_DICE = 254
_V_HOUSES = 256
_V_HOTELS = 257
_V_BANKRUPT = 258
_V_JAIL_TURNS = 262
_V_TURN_ORDER = 266
_V_DEBT_AMOUNT = 270
_V_DEBT_CREDITOR = 271
_V_AUCTION_PROP = 276
_V_AUCTION_BID = 277
_V_ROUND = 278
_V_AUCTION_LEADER = 279
_V_AUCTION_BIDDERS = 284
_V_EXTRA_ROLL = 288
_V_TRADE_SENDER = 289
_V_TRADE_OFFERED = 294
_V_TRADE_REQUESTED = 295
_V_TRADE_CASH_OFFERED = 296
_V_TRADE_CASH_REQUESTED = 297

_MONEY_SCALE = 2000.0
_MAX_ROUNDS = 200


class _Deed:
    """Read-only stand-in for ``monopoly_game_engine.state.Property``."""

    __slots__ = ("square_id", "owner", "mortgaged", "houses", "is_monopoly",
                 "color", "price", "mortgage_v", "data")

    def __init__(self, square: int) -> None:
        self.square_id = square
        self.owner: int | None = None
        self.mortgaged = False
        self.houses = 0
        self.is_monopoly = False
        self.color = COLOUR[square]
        self.price = PRICE[square]
        self.mortgage_v = MORTGAGE[square]
        self.data = {"house_price": HOUSE_PRICE[square], "rent": RENT[square],
                     "price": PRICE[square], "mortgage": MORTGAGE[square]}

    @property
    def is_real_estate(self) -> bool:
        return IS_REAL_ESTATE[self.square_id]


class _Seat:
    """Read-only stand-in for ``monopoly_game_engine.state.Player``."""

    __slots__ = ("player_id", "cash", "cash_saturated", "position", "in_jail",
                 "jail_turns", "gooj_card", "bankrupt", "properties")

    def __init__(self, player_id: int) -> None:
        self.player_id = player_id
        self.cash = 0.0
        #: True when the observation clipped this balance at its $5,000 ceiling.
        self.cash_saturated = False
        self.position = 0
        self.in_jail = False
        self.jail_turns = 0
        self.gooj_card = False
        self.bankrupt = False
        self.properties: list[_Deed] = []

    def net_worth(self) -> float:
        return self.cash + sum(
            deed_worth(p.square_id, p.houses, p.mortgaged, p.is_monopoly)
            for p in self.properties
        )

    def railroads_owned(self) -> int:
        return sum(1 for p in self.properties if p.color == "railroad")

    def utilities_owned(self) -> int:
        return sum(1 for p in self.properties if p.color == "utility")

    def can_afford(self, amount: float) -> bool:
        return self.cash >= amount


class _Offer:
    """Read-only stand-in for ``monopoly_game_engine.env.TradeOffer``."""

    __slots__ = ("from_player", "to_player", "offered_prop", "requested_prop",
                 "cash_offered", "cash_requested")

    def __init__(self, sender: int, receiver: int, offered, requested,
                 cash_offered: float, cash_requested: float) -> None:
        self.from_player = sender
        self.to_player = receiver
        self.offered_prop = offered
        self.requested_prop = requested
        self.cash_offered = cash_offered
        self.cash_requested = cash_requested


class ShadowEnv:
    """Everything the policy reads off the engine, rebuilt from the observation."""

    __slots__ = ("players", "properties", "phase", "round", "max_rounds",
                 "has_rolled", "houses_available", "hotels_available",
                 "turn_order", "debt_player", "debt_creditor", "debt_amount",
                 "auction_property_id", "auction_high_bid", "auction_high_bidder",
                 "auction_bidders", "pending_trades", "extra_roll_pending",
                 "consecutive_doubles", "last_dice", "done", "_acting",
                 "_allowed")

    def __init__(self) -> None:
        self.players: list[_Seat] = [_Seat(i) for i in range(NUM_PLAYERS)]
        self.properties: dict[int, _Deed] = {s: _Deed(s) for s in PROPERTY_IDS}
        self.phase = PHASE_PRE_ROLL
        self.round = 0
        self.max_rounds = _MAX_ROUNDS
        self.has_rolled = False
        self.houses_available = 32
        self.hotels_available = 12
        self.turn_order = list(range(NUM_PLAYERS))
        self.debt_player: int | None = None
        self.debt_creditor: int | None = None
        self.debt_amount = 0.0
        self.auction_property_id: int | None = None
        self.auction_high_bid = 0.0
        self.auction_high_bidder: int | None = None
        self.auction_bidders: list[int] = []
        self.pending_trades: dict[int, _Offer] = {}
        self.extra_roll_pending = False
        self.consecutive_doubles = 0
        self.last_dice = (0, 0)
        self.done = False
        self._acting = 0
        self._allowed: tuple[int, ...] = ()

    def whose_turn(self) -> int:
        return self._acting

    def active_player_id(self) -> int:
        return self._acting

    def get_allowed_actions(self, pid: int | None = None) -> list[int]:
        return list(self._allowed)

    def _incoming_trade_entry(self, pid: int):
        for sender, offer in self.pending_trades.items():
            if offer.to_player == pid:
                return sender, offer
        return None

    def _incoming_trade(self, pid: int):
        entry = self._incoming_trade_entry(pid)
        return None if entry is None else entry[1]


def _one_hot(vector: Sequence[float], start: int, width: int) -> int | None:
    for i in range(width):
        if vector[start + i] > 0.5:
            return i
    return None


def build_shadow(
    vector: Sequence[float],
    player_id: int,
    allowed: Sequence[int],
    *,
    max_rounds: int = _MAX_ROUNDS,
) -> ShadowEnv:
    """Rebuild the board from one observation.

    The player block is written relative to the acting seat and the deed
    ownership block in absolute seat ids, so both mappings are needed and the
    acting seat is the one the caller was handed.
    """
    env = ShadowEnv()
    env.max_rounds = max_rounds
    env._acting = int(player_id)
    env._allowed = tuple(int(a) for a in allowed)

    order = [player_id] + [i for i in range(NUM_PLAYERS) if i != player_id]

    for relative, seat_id in enumerate(order):
        base = _V_PLAYERS + relative * 4
        seat = env.players[seat_id]
        seat.position = int(round(vector[base] * 39.0))
        raw_cash = float(vector[base + 1])
        seat.cash = raw_cash * _CASH_SCALE
        seat.cash_saturated = raw_cash >= 0.999
        seat.in_jail = vector[base + 2] > 0.5
        seat.gooj_card = vector[base + 3] > 0.5
        seat.bankrupt = vector[_V_BANKRUPT + relative] > 0.5
        seat.jail_turns = int(round(vector[_V_JAIL_TURNS + relative] * 3.0))

    for i, square in enumerate(PROPERTY_IDS):
        base = _V_PROPS + i * 8
        deed = env.properties[square]
        owner = _one_hot(vector, base, NUM_PLAYERS)
        deed.owner = owner
        deed.mortgaged = vector[base + 5] > 0.5
        deed.is_monopoly = vector[base + 6] > 0.5
        deed.houses = int(round(vector[base + 7] * 5.0))
        if owner is not None:
            env.players[owner].properties.append(deed)

    phase = _one_hot(vector, _V_PHASE, len(_PHASES))
    env.phase = _PHASES[phase] if phase is not None else PHASE_PRE_ROLL
    whose = _one_hot(vector, _V_WHOSE_TURN, NUM_PLAYERS)
    if whose is not None:
        env._acting = order[whose]
    env.has_rolled = vector[_V_HAS_ROLLED] > 0.5
    env.consecutive_doubles = int(round(vector[_V_DOUBLES] * 3.0))
    env.last_dice = (
        int(round(vector[_V_DICE] * 6.0)),
        int(round(vector[_V_DICE + 1] * 6.0)),
    )
    env.houses_available = int(round(vector[_V_HOUSES] * 32.0))
    env.hotels_available = int(round(vector[_V_HOTELS] * 12.0))
    env.round = int(round(float(vector[_V_ROUND]) * max_rounds))
    env.extra_roll_pending = vector[_V_EXTRA_ROLL] > 0.5

    # `turn_order[slot]` is stored as the seat's index in `order`, scaled.
    turn_order = []
    for slot in range(NUM_PLAYERS):
        relative = int(round(float(vector[_V_TURN_ORDER + slot]) * (NUM_PLAYERS - 1)))
        turn_order.append(order[min(max(relative, 0), NUM_PLAYERS - 1)])
    if sorted(turn_order) == list(range(NUM_PLAYERS)):
        env.turn_order = turn_order

    env.debt_amount = float(vector[_V_DEBT_AMOUNT]) * _MONEY_SCALE
    creditor_slot = _one_hot(vector, _V_DEBT_CREDITOR, NUM_PLAYERS + 1)
    if creditor_slot is not None and creditor_slot > 0:
        env.debt_creditor = order[creditor_slot - 1]
    # The observation never names the debtor. The engine's rescue menu is the
    # tell: while a player owes rent it offers liquidation and nothing else --
    # no END_TURN, no BUY_PROPERTY (env.py:276).
    if env.debt_amount > 0 and env.phase == PHASE_POST_ROLL:
        if END_TURN not in env._allowed and ROLL_DICE not in env._allowed:
            env.debt_player = env._acting

    auction_slot = float(vector[_V_AUCTION_PROP])
    if auction_slot > 0:
        index = int(round(auction_slot * (_N_PROP + 1))) - 1
        if 0 <= index < _N_PROP:
            env.auction_property_id = PROPERTY_IDS[index]
    env.auction_high_bid = float(vector[_V_AUCTION_BID]) * _MONEY_SCALE
    leader_slot = _one_hot(vector, _V_AUCTION_LEADER, NUM_PLAYERS + 1)
    if leader_slot is not None and leader_slot > 0:
        env.auction_high_bidder = order[leader_slot - 1]
    env.auction_bidders = [
        order[i]
        for i in range(NUM_PLAYERS)
        if vector[_V_AUCTION_BIDDERS + i] > 0.5
    ]

    sender_slot = _one_hot(vector, _V_TRADE_SENDER, NUM_PLAYERS + 1)
    if sender_slot is not None and sender_slot > 0:
        sender = order[sender_slot - 1]
        offered = _deed_from_slot(env, vector[_V_TRADE_OFFERED])
        requested = _deed_from_slot(env, vector[_V_TRADE_REQUESTED])
        env.pending_trades[sender] = _Offer(
            sender,
            player_id,
            offered,
            requested,
            float(vector[_V_TRADE_CASH_OFFERED]) * _MONEY_SCALE,
            float(vector[_V_TRADE_CASH_REQUESTED]) * _MONEY_SCALE,
        )
    return env


def _deed_from_slot(env: ShadowEnv, slot: float) -> _Deed | None:
    if slot <= 0:
        return None
    index = int(round(float(slot) * (_N_PROP + 1))) - 1
    if 0 <= index < _N_PROP:
        return env.properties[PROPERTY_IDS[index]]
    return None


# ── board-view enrichment ────────────────────────────────────────────────
#
# The contract also ships a readable board view alongside the vector. Its
# field names are not pinned by the vector layout, so it is read defensively:
# anything recognised is used to sharpen the decode (above all the exact cash
# the vector clips at $5,000), anything unrecognised is ignored. A board view
# that disagrees with the vector never wins, because the vector is the part of
# the contract whose layout is fixed by the engine.

_CASH_KEYS = ("cash", "money", "balance", "funds")
_PLAYERS_KEYS = ("players", "seats", "player_states", "playerStates")
_ROUND_KEYS = ("round", "turn", "round_index", "roundIndex")
_MAX_ROUND_KEYS = ("max_rounds", "maxRounds", "round_limit", "roundLimit")


def _field(obj: Any, names: Sequence[str]) -> Any:
    if obj is None:
        return None
    for name in names:
        if isinstance(obj, dict):
            if name in obj:
                return obj[name]
        elif hasattr(obj, name):
            return getattr(obj, name)
    return None


def enrich_from_board(env: ShadowEnv, board: Any) -> None:
    """Overwrite clipped values with exact ones where the board supplies them."""
    if board is None:
        return
    try:
        rounds = _field(board, _MAX_ROUND_KEYS)
        if isinstance(rounds, (int, float)) and rounds > 0:
            env.max_rounds = int(rounds)
        current = _field(board, _ROUND_KEYS)
        if isinstance(current, (int, float)):
            env.round = int(current)

        seats = _field(board, _PLAYERS_KEYS)
        if not isinstance(seats, (list, tuple)):
            return
        for entry in seats:
            seat_id = _field(entry, ("player_id", "playerId", "id", "seat", "index"))
            if not isinstance(seat_id, (int, float)):
                continue
            seat_id = int(seat_id)
            if not 0 <= seat_id < NUM_PLAYERS:
                continue
            cash = _field(entry, _CASH_KEYS)
            if isinstance(cash, (int, float)):
                env.players[seat_id].cash = float(cash)
                env.players[seat_id].cash_saturated = False
    except Exception:
        # A board view we cannot read is not a reason to stop playing.
        return


# ══════════════════════════════════════════════════════════════════════════
# Harness adapter
#
# Four call shapes are in play and none of them can be assumed:
#
#     choose_action(state, player_id, allowed_actions)   the tournament
#     choose_action(env)                                 FixedPolicyAgent
#     choose_action(state, env, allowed_actions)         DDQNAgent / PPOAgent
#     choose_action(game, player_id, seed)               a bench arena
#
# So the arguments are identified by structure rather than by position, and the
# live engine is preferred whenever one is reachable because it is exact.
# ══════════════════════════════════════════════════════════════════════════


def _tuning_from_environment() -> Tuning | None:
    """Ablation hook: ``NEMESIS_TUNING`` holds a JSON object of overrides.

    Sweeping a constant becomes an environment variable rather than an edit,
    which keeps every measured configuration reproducible from its command
    line. Unset -- the tournament case -- costs one lookup and ships defaults.
    """
    raw = os.environ.get("NEMESIS_TUNING")
    if not raw:
        return None
    try:
        return Tuning(**json.loads(raw))
    except Exception:
        # A malformed sweep must not change how the shipped agent plays.
        return None


def _looks_like_env(obj: Any) -> bool:
    return hasattr(obj, "get_allowed_actions") and hasattr(obj, "players")


def _as_int_list(obj: Any) -> list[int] | None:
    """A short sequence of action ids, as opposed to a 300-float observation."""
    if isinstance(obj, (str, bytes, dict)) or obj is None:
        return None
    try:
        items = list(obj)
    except TypeError:
        return None
    if not items or len(items) > ACTION_SPACE_SIZE:
        return None
    out: list[int] = []
    for item in items:
        if isinstance(item, bool):
            return None
        if isinstance(item, int):
            out.append(item)
        elif isinstance(item, float) and item.is_integer():
            out.append(int(item))
        else:
            return None
    return out if all(0 <= a < ACTION_SPACE_SIZE for a in out) else None


def _as_vector(obj: Any) -> Sequence[float] | None:
    if isinstance(obj, (str, bytes, dict)):
        return None
    try:
        if len(obj) == _STATE_DIM:
            return obj
    except TypeError:
        pass
    return None


_VECTOR_KEYS = ("vector", "observation", "obs", "state_vector", "stateVector")
_BOARD_KEYS = ("board", "board_view", "boardView", "table")
_ACTIONS_KEYS = ("allowed_actions", "allowedActions", "legal_actions",
                 "legalActions", "actions")


def _unpack_state(state: Any) -> tuple[Sequence[float] | None, Any, list[int] | None]:
    """Pull the observation, the board view and the action ids out of a state.

    Handles the tournament's decision object (fields or mapping) and a bare
    300-float sequence alike.
    """
    direct = _as_vector(state)
    if direct is not None:
        return direct, None, None
    vector = _as_vector(_field(state, _VECTOR_KEYS))
    board = _field(state, _BOARD_KEYS)
    actions = _as_int_list(_field(state, _ACTIONS_KEYS))
    return vector, board, actions


class Agent:
    """NEMESIS bound to one seat, with a tolerant call signature."""

    policy_id = Nemesis.policy_id

    def __init__(self, player_id: int = 0, **kwargs: Any) -> None:
        for key in ("player_id", "pid", "agent_id", "seat"):
            value = kwargs.pop(key, None)
            if value is not None:
                player_id = int(value)
                break
        tuning = kwargs.pop("tuning", None)
        if kwargs and tuning is None:
            tuning = Tuning(**kwargs)
        if tuning is None:
            tuning = _tuning_from_environment()
        self.player_id = int(player_id)
        self._tuning = tuning
        self._seats: dict[int, Nemesis] = {}

    def _policy_for(self, seat: int) -> Nemesis:
        policy = self._seats.get(seat)
        if policy is None:
            policy = Nemesis(seat, self._tuning)
            self._seats[seat] = policy
        return policy

    # -- harness interface ------------------------------------------------

    def choose_action(self, *args: Any, **kwargs: Any) -> int:
        env = kwargs.get("env")
        allowed = _as_int_list(kwargs.get("allowed_actions"))
        state = kwargs.get("state")
        seat = kwargs.get("player_id")
        if seat is None:
            seat = kwargs.get("pid")

        # The documented tournament shape is positional and unambiguous:
        # `(state, player_id, allowed_actions)` with a seat index in the
        # middle and action ids at the end. Structural sniffing alone cannot
        # separate it from `(state, env, allowed_actions)` when the state is
        # itself a sequence of numbers -- it reads the state as the action
        # list and then answers with an action nobody offered.
        contract = None
        if len(args) == 3 and env is None and allowed is None:
            first, middle, last = args
            ids = _as_int_list(last)
            if (
                ids is not None
                and isinstance(middle, int)
                and not isinstance(middle, bool)
                and 0 <= middle < NUM_PLAYERS
            ):
                contract = (first, middle, ids)

        if contract is not None:
            state, seat, allowed = contract
        else:
            for arg in args:
                if env is None:
                    candidate = getattr(arg, "env", arg)
                    if _looks_like_env(candidate):
                        env = candidate
                        continue
                ids = _as_int_list(arg)
                if ids is not None and allowed is None and len(ids) != _STATE_DIM:
                    allowed = ids
                    continue
                if state is None and arg is not None and not isinstance(
                    arg, (int, float)
                ):
                    state = arg

        # A bare seat index is the only small integer in any of the shapes.
        if seat is None:
            for arg in args:
                if isinstance(arg, int) and not isinstance(arg, bool):
                    if 0 <= arg < NUM_PLAYERS:
                        seat = arg
                        break
        seat = self.player_id if seat is None else int(seat)

        try:
            return self._decide(env, state, seat, allowed)
        except Exception:
            # Any raise here is a strike and a forfeited decision. There is no
            # circumstance in which crashing beats returning a legal action.
            if allowed:
                return int(END_TURN if END_TURN in allowed else allowed[0])
            return END_TURN

    def _decide(self, env: Any, state: Any, seat: int,
                allowed: list[int] | None) -> int:
        vector, board, state_actions = (None, None, None)
        if state is not None:
            vector, board, state_actions = _unpack_state(state)
        if allowed is None:
            allowed = state_actions

        if env is not None:
            # The engine only ever asks the seat it is waiting on, auctions and
            # out-of-turn responses included, so `whose_turn()` is
            # authoritative. Acting for another seat forfeits the game.
            try:
                seat = int(env.whose_turn())
            except Exception:
                pass
            if allowed is None:
                allowed = [int(a) for a in env.get_allowed_actions(seat)]
            if len(allowed) == 1:
                return int(allowed[0])
            action = int(self._policy_for(seat).act(env, allowed))
            return action if action in allowed else int(
                END_TURN if END_TURN in allowed else allowed[0]
            )

        if not allowed:
            return END_TURN
        if len(allowed) == 1:
            return int(allowed[0])
        if vector is None or len(vector) != _STATE_DIM:
            return int(END_TURN if END_TURN in allowed else allowed[0])

        shadow = build_shadow(vector, seat, allowed)
        enrich_from_board(shadow, board)
        action = int(self._policy_for(seat).act(shadow, allowed))
        return action if action in allowed else int(
            END_TURN if END_TURN in allowed else allowed[0]
        )


def make_agent(player_id: int = 0, **kwargs: Any) -> Agent:
    """Some harnesses look for a module-level factory instead of a class."""
    return Agent(player_id, **kwargs)


_DEFAULT_AGENTS: dict[int, Agent] = {}


def choose_action(*args: Any, **kwargs: Any) -> int:
    """Module-level entry point.

    This is the shape the tournament calls:
    ``choose_action(state, player_id, allowed_actions) -> int``. One agent is
    kept per seat so the per-opponent trade bookkeeping survives across
    decisions within a match.

    A harness that imports this bare function directly -- rather than
    constructing ``Agent(player_id)`` per seat, the pattern every reference
    agent and every measured game in this project actually uses -- has no
    seat to fall back on unless it names one. The keyword aliases checked
    here match ``Agent.__init__`` exactly, so whichever name such a harness
    happens to pass is honoured the same way construction already honours
    it. Positional calls, and every call this repository's own measurements
    make, are unaffected: they hit the same lookup they always did.
    """
    seat = None
    for key in ("player_id", "pid", "agent_id", "seat"):
        value = kwargs.get(key)
        if value is not None:
            seat = int(value)
            break
    if seat is None:
        for arg in args:
            if isinstance(arg, int) and not isinstance(arg, bool) and 0 <= arg < NUM_PLAYERS:
                seat = arg
                break
    seat = 0 if seat is None else int(seat)
    agent = _DEFAULT_AGENTS.get(seat)
    if agent is None:
        agent = Agent(seat)
        _DEFAULT_AGENTS[seat] = agent
    return agent.choose_action(*args, **kwargs)
