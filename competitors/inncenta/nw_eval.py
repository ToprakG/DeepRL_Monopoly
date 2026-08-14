"""Net-worth-native position evaluator. No ASU, no imports from it.

WHY THIS EXISTS
---------------
The engine decides every game by Player.net_worth() -- either one player is
left standing, or the 200-round cap is reached and the greatest net worth wins.
That function is not list price: an unmortgaged deed counts 2.5x its price,
5.0x once its colour group is complete, houses count house_price*(1+0.5h), and
cash counts at face value.

Two consequences drive the whole design:

  * SPENDING IS ACCRETIVE. Buying a deed converts $1 of cash into $2.50 of
    score. Building converts at 1.5x (first house) up to 3.5x (hotel).
  * RAISING CASH IS EXPENSIVE, AND UNEVENLY SO. Mortgaging destroys 4.0 of
    score per dollar raised; selling a deed to the bank destroys 5.0.

Our previous evaluator used net_worth() as its base and then added
W_INCOME * rent_per_round with W_INCOME = 26. With rent of 30-150/round that
term contributes 780-3900 against a net worth of ~5000, so a *speculative rent
projection* outweighed the actual score. That is the single biggest defect and
it is why the terms below are all denominated in net-worth units.

WHAT net_worth() ALREADY CAPTURES, so we must not double-count
--------------------------------------------------------------
The agent simulates an action and evaluates the resulting position, so
completing a colour group is already priced -- every deed in the group jumps
2.5x -> 5.0x inside net_worth() itself. No separate "set completion" term is
needed here; adding one would double-count.

WHAT IT MISSES, which is what the terms below are for
-----------------------------------------------------
  flow      net worth is a snapshot. Rent transfers cash every round, and cash
            counts at face value, so expected rent is literally projected
            net-worth change. Horizon is capped: uncapped it would be ~190
            rounds early and reintroduce exactly the swamping bug above.
  solvency  bankruptcy is an absorbing state worth -infinity, not a smooth
            deduction. A penalty proportional to the shortfall against the
            worst rent reachable on the next roll.
  relative  we win by holding the MOST net worth, not the most possible.
            Against a runaway leader, denial beats accumulation.

Variants V0..V3 are selectable so each term can be justified by measurement
rather than by argument.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _find_repo() -> Path:
    env = os.environ.get("DEEPRL_MONOPOLY_ROOT")
    if env and (Path(env) / "monopoly_game_engine").is_dir():
        return Path(env)
    try:
        import monopoly_game_engine as _m
        return Path(_m.__file__).resolve().parent.parent
    except Exception:
        pass
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "monopoly_game_engine").is_dir():
            return parent
        for sib in (parent.iterdir() if parent.is_dir() else []):
            if sib.is_dir() and (sib / "monopoly_game_engine").is_dir():
                return sib
    raise RuntimeError("Could not locate DeepRL_Monopoly; set DEEPRL_MONOPOLY_ROOT")


ROOT = _find_repo()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import board_stats as B  # noqa: E402

# ---- weights (net-worth units throughout) --------------------------------
VARIANT = "V2"       # V0 legacy | V1 nw+solvency | V2 +flow | V3 +relative
HORIZON = 10
# MEASURED: a horizon sweep against the two strongest rivals gave
#     h10 27.1%   h25 24.0%   h50 21.9%   (mean of r2 and r3, same seeds)
# Rent 50 rounds out is speculation; weighting it as realised score distorts
# the ranking. Shorter is better.
W_FLOW = 1.0         # flow is already in net-worth units, so 1.0 is the
                     # principled default, not a fitted constant
W_RISK = 3.0         # bankruptcy is absorbing; worth over-weighting
W_DEV = 1.0         # development potential is already in net-worth units
LIQUIDITY_CREDIT = 1.0
# How much of our mortgage/house-sale capacity counts as available cash when
# testing solvency. At 1.0 (the original) the shortfall fires on only 10.2% of
# positions against 30.2% for a cash-only test -- we look solvent in two thirds
# of the spots where we are actually short. Liquidation is also not free: it
# destroys 4-5 net worth per dollar raised, so crediting it at face value is
# doubly wrong. 0.0 = cash only.
W_REL = 0.35         # how much an opponent's lead subtracts from our position
W_DENY = 0.25
# Denial: net worth an opponent CANNOT reach because we hold a deed in their
# colour group. Taking the deed that would complete someone's set denies them
# the 2.5x -> 5.0x revaluation on every deed they already hold there, plus the
# rent multiplication that follows development. Our evaluator has never priced
# this. A rival's published configuration history reports denial as their single
# largest gain (+5.3pp), which makes it the best-evidenced idea left untried.
MAX_ROUNDS = 200

UTILITY_MULT = {1: 4, 2: 10}
EXPECTED_DICE = 7.0


def net_worth(env, pid: int) -> float:
    return float(env.players[pid].net_worth())


def income_per_round(env, pid: int) -> float:
    """Rent collected per full round, summed over live opponents."""
    player = env.players[pid]
    rivals = sum(1 for i, p in enumerate(env.players) if i != pid and not p.bankrupt)
    if not rivals:
        return 0.0
    active = [p for p in player.properties if not p.mortgaged]
    rails = [p for p in active if p.color == "railroad"]
    utils = [p for p in active if p.color == "utility"]
    total = sum(B.expected_rent(p.square_id, p.houses, p.is_monopoly)
                for p in active if p.color not in ("railroad", "utility"))
    if rails:
        rent = B.railroad_rent().get(len(rails), 200)
        total += sum(B.LAND_PROB[p.square_id] * rent for p in rails)
    if utils:
        mult = UTILITY_MULT.get(len(utils), 10)
        total += sum(B.LAND_PROB[p.square_id] * EXPECTED_DICE * mult for p in utils)
    return total * rivals


def exposure_per_round(env, pid: int) -> float:
    """Rent we expect to PAY per round."""
    total = 0.0
    for i, rival in enumerate(env.players):
        if i == pid or rival.bankrupt:
            continue
        live = max(1, sum(1 for j, p in enumerate(env.players)
                          if j != i and not p.bankrupt))
        total += income_per_round(env, i) / live
    return total


def worst_reachable_rent(env, pid: int) -> float:
    """Largest rent we could owe on the next roll (dice totals 2..12).

    Worst case rather than expected: bankruptcy is absorbing, so the question
    is whether we survive, not what we pay on average.
    """
    player = env.players[pid]
    pos = getattr(player, "position", 0)
    owned = {p.square_id: p for i, q in enumerate(env.players) if i != pid
             for p in q.properties if not p.mortgaged}
    worst = 0.0
    for total in range(2, 13):
        prop = owned.get((pos + total) % 40)
        if prop is None:
            continue
        if prop.color == "railroad":
            n = sum(1 for p in env.players[prop.owner].properties
                    if p.color == "railroad" and not p.mortgaged)
            rent = float(B.railroad_rent().get(n, 200))
        elif prop.color == "utility":
            n = sum(1 for p in env.players[prop.owner].properties
                    if p.color == "utility" and not p.mortgaged)
            rent = float(total * UTILITY_MULT.get(n, 10))
        else:
            rent = float(B.rent_at(prop.square_id, prop.houses, prop.is_monopoly))
        worst = max(worst, rent)
    return worst


def liquid_assets(env, pid: int) -> float:
    """Cash, plus LIQUIDITY_CREDIT of what could be raised by liquidating."""
    player = env.players[pid]
    total = float(player.cash)
    if LIQUIDITY_CREDIT <= 0.0:
        return total
    raisable = 0.0
    for prop in player.properties:
        if not prop.mortgaged:
            raisable += float(prop.mortgage_v)
        raisable += prop.houses * float(prop.data.get("house_price", 0)) * 0.5
    return total + LIQUIDITY_CREDIT * raisable


def development_potential(env, pid: int) -> float:
    """Net worth obtainable by developing monopolies we can afford to develop.

    Derived from monopoly_game_engine/state.py calculate_net_worth(): h houses
    on a property contribute h * house_price * (1 + 0.5h). The marginal net
    worth of the h-th house is therefore

        h*hp*(1+0.5h) - (h-1)*hp*(1+0.5(h-1))

    against a cash cost of hp, so the NET gain per dollar rises with each step:

        house 1   0.5x     house 4   3.5x
        house 2   1.5x     hotel     4.5x
        house 3   2.5x

    Building is increasingly accretive, which inverts the rule our old
    BUILD_TARGET=3 encoded -- that came from RENT per dollar (the 2->3 step is
    the best rent step), a different objective from the one the engine scores.

    Even building is now enforced upstream, so a group develops as a unit and
    the affordable level is the same for every property in it.
    """
    player = env.players[pid]
    budget = float(player.cash)
    gain = 0.0
    groups: dict[str, list] = {}
    for prop in player.properties:
        if prop.is_monopoly and not prop.mortgaged and prop.is_real_estate:
            groups.setdefault(prop.color, []).append(prop)
    for props in groups.values():
        if not props:
            continue
        hp = float(props[0].data.get("house_price", 0) or 0)
        if hp <= 0:
            continue
        n = len(props)
        level = min(p.houses for p in props)
        while level < 5 and budget >= hp * n:
            nxt = level + 1
            marginal = (nxt * hp * (1 + 0.5 * nxt)
                        - level * hp * (1 + 0.5 * level))
            gain += (marginal - hp) * n      # NET of the cash spent
            budget -= hp * n
            level = nxt
    return gain


def denial_value(env, pid: int) -> float:
    """Net worth opponents cannot realise because we hold deeds in their groups.

    For each colour group, if a single opponent holds all of it except deeds we
    own, our holding is what stands between them and a completed set. Price that
    at the revaluation they are denied: every deed they hold in the group would
    jump from 2.5x to 5.0x on completion.
    """
    total = 0.0
    for colour, squares in B.SETS.items():
        if not squares:
            continue
        holders: dict[int, float] = {}
        ours = 0
        for sq in squares:
            prop = env.properties.get(sq)
            if prop is None or prop.owner is None:
                continue
            if prop.owner == pid:
                ours += 1
            else:
                holders[prop.owner] = holders.get(prop.owner, 0.0) + prop.price
        if not ours or len(holders) != 1:
            continue                      # we block nobody, or the group is split
        owner, their_value = next(iter(holders.items()))
        if env.players[owner].bankrupt:
            continue
        held = sum(1 for sq in squares
                   if env.properties[sq].owner == owner)
        if held + ours != len(squares):
            continue                      # a third party also holds part of it
        total += their_value * 2.5        # the revaluation we are denying them
    return total


def best_rival(env, pid: int) -> float:
    vals = [net_worth(env, i) for i, p in enumerate(env.players)
            if i != pid and not p.bankrupt]
    return max(vals) if vals else 0.0


def evaluate(env, pid: int) -> float:
    """Scalar value of this position for `pid`. Higher is better."""
    player = env.players[pid]
    if player.bankrupt:
        return -1e9
    alive = [p for p in env.players if not p.bankrupt]
    if len(alive) == 1 and alive[0].player_id == pid:
        return 1e9

    value = net_worth(env, pid)
    if VARIANT == "V1":
        pass
    else:
        rounds_left = max(0, MAX_ROUNDS - getattr(env, "round", 0))
        horizon = min(HORIZON, rounds_left)
        value += W_FLOW * horizon * (income_per_round(env, pid)
                                     - exposure_per_round(env, pid))

    if W_DENY:
        value += W_DENY * denial_value(env, pid)

    if VARIANT in ("V4", "V5"):
        # an undeveloped monopoly is an OPTION to multiply both net worth and
        # rent; net_worth() alone prices it at 5.0x the deed and stops there
        value += W_DEV * development_potential(env, pid)

    shortfall = max(0.0, worst_reachable_rent(env, pid) - liquid_assets(env, pid))
    value -= W_RISK * shortfall

    if VARIANT in ("V3", "V5"):
        value -= W_REL * best_rival(env, pid)
    return value


def evaluate_all(env) -> list[float]:
    return [evaluate(env, i) for i in range(len(env.players))]
