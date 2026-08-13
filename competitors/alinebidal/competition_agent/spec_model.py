"""Quantities the SPEC rules are stated in terms of.

Every function here implements a specific, cited finding from `SPEC.md`. None
of it is a reconstruction of the teacher's internals — it is the arithmetic the
observed behaviour was shown to obey, rebuilt from the probe evidence.

The teacher is not consulted here at all; this module never imports
`ASU_FROZEN_TEACHER` (DECISIONS D0.3).
"""

from __future__ import annotations

import functools
import os
from typing import Dict, List

from monopoly_game_engine.constants import (
    COLOR_GROUPS, JAIL_BAIL, PROPERTIES, PROPERTY_IDS,
)

MIN_CASH = 200          # SPEC A2, D1 — the floor, confirmed on buy/build/unmortgage/bail

# The monopoly term `max_group_rent / 2**missing` is hundreds to low thousands
# while list price is $60-$400 and projected rent is tens of dollars, so it
# swamps both. D2.6 measured scaling it to 0.1 as worth +8.2pp of auction
# agreement on 402 RANDOMLY GENERATED auction states, [83.2, 89.8] against
# [74.3, 82.3].
#
# **That gain does not transfer, and the default is therefore 1.0.** An A/B on
# identical code over 5,363 held-out decisions from real play (D2.7):
#
#     scale 1.0 -> auction 90.5%, total 73.4%
#     scale 0.1 -> auction 88.7%, total 70.7%
#
# Auction moves -1.8pp, not +8.2pp, and the total falls 2.7pp. The random
# boards are simply a different distribution from the auction states real play
# produces. Kept switchable so the comparison can be re-run, but 1.0 is what
# ships until a measurement on the real distribution says otherwise.
# Overridable via MONOPOLY_SCALE env var so the A/B can be run on identical
# code rather than on two different commits.
MONOPOLY_SCALE = float(os.environ.get("MONOPOLY_SCALE", "1.0"))
SHORT_TURNS = 5         # SPEC A5 — complete-turn horizon
GO_TO_JAIL = 30
JAIL = 10


# --------------------------------------------------------------------------
# SPEC A4 + A5 — exact 2d6 complete-turn landing enumeration, with doubles
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def turn_landings(start: int) -> tuple:
    """Expected landings per square over ONE complete turn from `start`.

    A complete turn is not one roll: doubles grant another roll, and a third
    consecutive double sends the player to jail without landing. SPEC A5 shows
    this directly — squares 13-14 ahead carry positive landing mass, which a
    single 2d6 roll cannot reach at all.

    Returned as a tuple of (square, expected_count) so it stays hashable and
    cacheable; the table is position-only, so 40 entries cover the board.
    """
    acc: Dict[int, float] = {}

    def walk(pos: int, prob: float, doubles: int) -> None:
        for d1 in range(1, 7):
            for d2 in range(1, 7):
                p = prob / 36.0
                nxt = (pos + d1 + d2) % 40
                if d1 == d2:
                    if doubles + 1 >= 3:
                        continue          # third double -> jail, no landing
                    acc[nxt] = acc.get(nxt, 0.0) + p
                    if nxt != GO_TO_JAIL:
                        walk(nxt, p, doubles + 1)
                else:
                    if nxt == GO_TO_JAIL:
                        continue          # sent to jail, no rent event
                    acc[nxt] = acc.get(nxt, 0.0) + p

    walk(start, 1.0, 0)
    return tuple(sorted(acc.items()))


@functools.lru_cache(maxsize=None)
def multi_turn_landings(start: int, turns: int) -> tuple:
    """Expected landings over `turns` complete turns (SPEC A5 horizon).

    Successive turns are chained through the one-turn distribution, so the
    player's position spreads realistically rather than assuming a uniform lap.
    """
    dist = {start: 1.0}
    acc: Dict[int, float] = {}
    for _ in range(turns):
        nxt: Dict[int, float] = {}
        for pos, w in dist.items():
            for sq, p in turn_landings(pos):
                acc[sq] = acc.get(sq, 0.0) + w * p
                nxt[sq] = nxt.get(sq, 0.0) + w * p
        total = sum(nxt.values()) or 1.0
        dist = {k: v / total for k, v in nxt.items()}
    return tuple(sorted(acc.items()))


# --------------------------------------------------------------------------
# rent
# --------------------------------------------------------------------------
def rent_for(env, sq: int, dice: int = 7) -> int:
    """Rent this deed would charge right now, 0 if unowned or mortgaged."""
    prop = env.properties[sq]
    if prop.owner is None or prop.mortgaged:
        return 0
    owner = env.players[prop.owner]
    return int(prop.get_rent(dice, owner.railroads_owned(),
                             owner.utilities_owned()))


def expected_rent_flow(env, pid: int, turns: int = 1) -> float:
    """Net expected rent over `turns`: collected from opponents minus paid.

    SPEC A3/A4/A6 — projected from every player's ACTUAL board position, not a
    uniform lap model, and summed over opponents. SPEC D5 — the gate-1 term is
    net, so rent we expect to pay counts against us.
    """
    income = 0.0
    for opp in env.players:
        if opp.player_id == pid or opp.bankrupt or opp.in_jail:
            continue
        for sq, p in multi_turn_landings(opp.position, turns):
            prop = env.properties.get(sq)
            if prop is not None and prop.owner == pid:
                income += p * rent_for(env, sq)

    outgo = 0.0
    me = env.players[pid]
    if not me.in_jail:                    # SPEC G4 — jail shelters from rent
        for sq, p in multi_turn_landings(me.position, turns):
            prop = env.properties.get(sq)
            if prop is not None and prop.owner not in (None, pid):
                outgo += p * rent_for(env, sq)
    return income - outgo


def expected_rent_income(env, pid: int, turns: int = SHORT_TURNS) -> float:
    """Gross expected rent collected (SPEC D3's income term)."""
    income = 0.0
    for opp in env.players:
        if opp.player_id == pid or opp.bankrupt:
            continue
        for sq, p in multi_turn_landings(opp.position, turns):
            prop = env.properties.get(sq)
            if prop is not None and prop.owner == pid:
                income += p * rent_for(env, sq)
    return income


def worst_reachable_rent(env, pid: int) -> int:
    """The largest rent any opponent could charge us (SPEC D3).

    D5 shows this term stops depending on our position once it dominates, so
    it is a worst case over the board rather than a position-weighted average.
    """
    worst = 0
    for sq in PROPERTY_IDS:
        prop = env.properties[sq]
        if prop.owner is None or prop.owner == pid or prop.mortgaged:
            continue
        worst = max(worst, rent_for(env, sq, dice=12))
    return worst


def liquidatable_worth(env, pid: int) -> int:
    """Cash raisable without going bankrupt (SPEC D3, F3).

    Mortgage value of every unmortgaged deed, plus half the paid build cost of
    its houses. Calibrated against D3: the orange group unmortgaged and
    undeveloped gives 90+90+100 = 280, and two railroads with one already
    mortgaged give 100 — both exactly as observed.
    """
    total = 0
    for prop in env.players[pid].properties:
        if not prop.mortgaged:
            total += prop.mortgage_v
        if prop.is_real_estate and prop.houses:
            total += int(prop.houses * prop.data["house_price"] * 0.5)
    return total


# --------------------------------------------------------------------------
# SPEC D1-D4 — the two safety gates
# --------------------------------------------------------------------------
def gates_ok(env, pid: int, spend: int, liq_delta: int = 0) -> bool:
    """Both gates, with the binding one being whichever demands more (SPEC D4).

        gate 1:  cash_after + E[net rent next turn]            >= 200
        gate 2:  cash_after + rent_income + liquidatable_after
                   - worst_reachable_rent                      >  0

    **Gate 1 is exact. Gate 2 is approximate — known residual, see below.**

    Gate 1 reproduces every measured threshold in the regime where it binds:
    28/28 buy flip points within $2 (26 exact), and all six gate-1 rows of the
    build and unmortgage sweeps exactly.

    Gate 2's income term is not yet resolved. Fitting `worst - liq - cushion`
    to SPEC D3 implies an income of $49 (build setup) and $100 (unmortgage
    setup); a 5-turn projection gives $28.8 and $18.8. No single horizon fits
    both — build matches at ~8 turns, unmortgage at ~24 — so the shortfall is
    not a horizon choice. `liq_delta` was added to test whether gate 2 is
    evaluated on the *post*-action state (money converted into an asset is
    still liquidatable). That has the right sign but the wrong size: the
    build case needs +$21 against +$50 for a half-price house, and the
    unmortgage case +$82 against +$100 for the freed deed.

    Left as measured rather than curve-fitted, because a fitted constant that
    matched two setups would not generalise and would look like knowledge.
    Callers may pass `liq_delta`, but it defaults to 0 — the plain form, whose
    residual is a known **+$21 to +$81 too strict** in the gate-2 regime. The
    clone is therefore slightly more cautious than the teacher when opponents
    are heavily developed, which is a safe direction to err. Whether this
    costs agreement is measured, not assumed: see `agreement.py`.
    """
    cash_after = env.players[pid].cash - spend
    if cash_after < 0:
        return False

    # Phase 5 module 3, off unless BEYOND_ENDGAME=1: the $200 floor is a
    # constant but the danger it guards is not. Scales with board development.
    floor = MIN_CASH
    try:
        from competition_agent.beyond.endgame import cushion_multiplier
        floor = MIN_CASH * cushion_multiplier(env)
    except Exception:                                      # noqa: BLE001
        pass

    # SPEC A3/D1/D5 — gate 1 uses NET rent over the next complete turn
    if cash_after + expected_rent_flow(env, pid, turns=1) < floor:
        return False

    # SPEC D3/D7 — gate 2 uses gross income and POST-action liquidatable worth
    slack = (cash_after
             + expected_rent_income(env, pid)
             + liquidatable_worth(env, pid) + liq_delta
             - worst_reachable_rent(env, pid))
    return slack > 0


# --------------------------------------------------------------------------
# SPEC B — deed valuation for auctions
# --------------------------------------------------------------------------
def group_state(env, pid: int, sq: int):
    """(owned_by_us_excluding_sq, group_size) for this deed's colour group."""
    group = COLOR_GROUPS[PROPERTIES[sq]["color"]]
    owned = sum(1 for s in group if s != sq and env.properties[s].owner == pid)
    return owned, len(group)


def max_group_rent(env, sq: int) -> int:
    """Rent the group could charge fully developed — the monopoly upside."""
    group = COLOR_GROUPS[PROPERTIES[sq]["color"]]
    total = 0
    for s in group:
        data = PROPERTIES[s]
        if data["color"] == "railroad":
            total += data["rent"][3]
        elif data["color"] == "utility":
            total += data["rent"][1] * 7
        else:
            total += data["rent"][5]
    return total


def marginal_monopoly_value(env, pid: int, sq: int) -> float:
    """SPEC B5 — marginal group value of acquiring `sq`.

        M / 2**missing_after  -  (M / 2**missing_before if owned_before else 0)

    The second term's guard is the clause B5 identifies: a group the player
    owns *nothing* of contributes no monopoly term, which is why owning one
    deed of a three-deed group buys no premium at all (B3, ratio 1.00 in
    18/18 cases) while the completing deed roughly doubles the ceiling (B4).
    """
    owned, size = group_state(env, pid, sq)
    m = float(max_group_rent(env, sq))
    after = m / (2 ** (size - owned - 1))
    before = (m / (2 ** (size - owned))) if owned > 0 else 0.0
    return MONOPOLY_SCALE * (after - before)


def _denial(env, pid, sq):
    """Phase 5 module 2, off unless BEYOND_DENIAL=1. Import is local so the
    default path costs nothing and the flag can be flipped per process."""
    try:
        from competition_agent.beyond.denial import denial_value
        return denial_value(env, pid, sq)
    except Exception:                                      # noqa: BLE001
        return 0.0


def deed_value(env, pid: int, sq: int) -> float:
    """What acquiring `sq` is worth to `pid` (SPEC B1-B5).

    Assets at list price, plus projected rent from opponents' real positions
    over the short horizon, plus the marginal group term. Calibrated so that
    ceilings land in the 1.9x-7.4x band B1 measured and reproduce B3/B4's
    ratios by construction.
    """
    price = PROPERTIES[sq]["price"]
    prop = env.properties[sq]

    rent_flow = 0.0
    saved_owner = prop.owner
    prop.owner = pid                       # counterfactual: we hold it
    try:
        for opp in env.players:
            if opp.player_id == pid or opp.bankrupt:
                continue
            for land_sq, p in multi_turn_landings(opp.position, SHORT_TURNS):
                if land_sq == sq:
                    rent_flow += p * rent_for(env, sq)
    finally:
        prop.owner = saved_owner

    return (price + rent_flow + marginal_monopoly_value(env, pid, sq)
            + _denial(env, pid, sq))


def auction_ceiling(env, pid: int, sq: int) -> float:
    """Most we should bid (SPEC B1). Never negative."""
    return max(0.0, deed_value(env, pid, sq))


__all__ = [
    "MIN_CASH", "SHORT_TURNS", "auction_ceiling", "deed_value",
    "expected_rent_flow", "expected_rent_income", "gates_ok", "group_state",
    "liquidatable_worth", "marginal_monopoly_value", "max_group_rent",
    "multi_turn_landings", "rent_for", "turn_landings", "worst_reachable_rent",
]


# --------------------------------------------------------------------------
# whole-state valuation — the separability fix (see DECISIONS D2.5)
# --------------------------------------------------------------------------
def state_value(env, pid: int) -> float:
    """Value of `pid`'s whole position, evaluated jointly.

    `deed_value` is a *marginal*: what one more deed adds to the current
    holding. Ranking a trade by `deed_value(req) - deed_value(offer)` treats
    the two legs as independent, which they are not — the monopoly term is
    `M / 2**missing`, so removing a deed can collapse a group at the same
    moment another deed completes one. Marginals cannot see that interaction.

    This evaluates the position as a whole, so a swap is scored as
    `state_value(after) - state_value(before)` with both legs applied.

    Components follow the ones the probes evidenced: unmortgaged asset value
    (A2's price basis), projected rent from opponents' real positions
    (A4/A6), and a per-group monopoly term discounted by missing deeds (B5).
    """
    me = env.players[pid]
    assets = 0.0
    for prop in me.properties:
        if not prop.mortgaged:
            assets += prop.price
            if prop.is_real_estate and prop.houses:
                assets += prop.houses * prop.data["house_price"]

    rent = 0.0
    for opp in env.players:
        if opp.player_id == pid or opp.bankrupt:
            continue
        for sq, p in multi_turn_landings(opp.position, SHORT_TURNS):
            prop = env.properties.get(sq)
            if prop is not None and prop.owner == pid:
                rent += p * rent_for(env, sq)

    mono = 0.0
    for color, squares in COLOR_GROUPS.items():
        owned = sum(1 for s in squares if env.properties[s].owner == pid)
        if owned == 0:                      # B5 — no presence, no term
            continue
        missing = len(squares) - owned
        mono += MONOPOLY_SCALE * max_group_rent(env, squares[0]) / (2 ** missing)

    return assets + rent + mono


def swap_delta(env, pid: int, give_sq: int, get_sq: int) -> float:
    """Change in `pid`'s state value from giving `give_sq` and getting `get_sq`.

    Applies both legs to the live objects, measures, then restores exactly.
    The env is never left mutated (the harness contract).
    """
    give, get = env.properties[give_sq], env.properties[get_sq]
    g_owner, t_owner = give.owner, get.owner
    g_mono, t_mono = give.is_monopoly, get.is_monopoly
    before = state_value(env, pid)
    try:
        give.owner = t_owner
        get.owner = pid
        env._update_monopolies()
        after = state_value(env, pid)
    finally:
        give.owner = g_owner
        get.owner = t_owner
        give.is_monopoly, get.is_monopoly = g_mono, t_mono
        env._update_monopolies()
    return after - before
