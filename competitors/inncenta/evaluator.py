"""Our own position evaluator for ppo-plus-v2.

Designed from Monopoly economics and measurements of THIS engine, not derived
from any existing agent. The thesis is that a Monopoly position is worth the
rent stream it will collect, minus the rent it will pay, minus the chance of
being wiped out before collecting it.

    value = liquid assets
          + W_INCOME  * rent collected per round
          - W_DENY    * rent paid per round
          + W_POTENT  * rent unlocked by building we can actually afford
          - W_RISK    * illiquidity relative to the rent we may have to pay

Design notes worth keeping:

* Income is the dominant term, not net worth. Two players with equal net worth
  are not equal if one holds a developed orange set and the other holds
  scattered singles. `net_worth()` cannot see that; expected_rent can.

* The build-potential term is what stops the evaluator being myopic. Holding a
  monopoly with 0 houses looks poor on current income but is worth a great deal,
  because the 2->3 house step returns ~3.5x per dollar (board_stats) and this
  engine does not enforce even building.

* Denial is weighted lower than income. Damage to an opponent only materialises
  when someone lands on them, and in a 4-player game two thirds of that damage
  lands on somebody else.

Weights are deliberately named and grouped so they can be tuned by self-play
later; none of them were fitted to benchmark seeds.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import board_stats as B  # noqa: E402

# ---- weights -------------------------------------------------------------
W_INCOME = 26.0    # rounds of rent we expect to still collect
W_DENY = 9.0       # opponents' income matters, but is diluted across 3 rivals
W_POTENT = 14.0    # value of building we could do right now
W_RISK = 2.5       # penalty scale for being unable to pay a likely rent
CASH_FLOOR = 200   # below this we are one bad landing from liquidating
AUCTION_RESERVE_FRAC = 0.15
# Measured (+5.0pp on std AND hard, 80 games each): an ABSOLUTE reserve in the
# auction path becomes unsatisfiable exactly when cash dips, so the agent stops
# bidding when contesting deeds matters most. A proportional reserve does not.
HARD_GATE = None
# ASU makes unsafe actions INELIGIBLE; we only penalise them via W_RISK. A
# penalty is negotiable -- a large enough income term buys a bankrupting move --
# which is a candidate explanation for losing 88% of games vs ASU by bankruptcy.
# Set to a multiple of per-round exposure to turn the penalty into a filter.
BUILD_TARGET = 3   # the 3-house sweet spot (see board_stats.build_gain)
UTILITY_MULT = {1: 4, 2: 10}
EXPECTED_DICE = 7.0


def income_per_round(env, pid: int) -> float:
    """Rent collected per full round, summed over every live opponent.

    Railroads and utilities are handled separately because their rent depends
    on HOW MANY of the group the owner holds, not on houses.
    """
    player = env.players[pid]
    rivals = sum(1 for i, p in enumerate(env.players) if i != pid and not p.bankrupt)
    if not rivals:
        return 0.0

    active = [p for p in player.properties if not p.mortgaged]
    railroads = [p for p in active if p.color == "railroad"]
    utilities = [p for p in active if p.color == "utility"]

    total = sum(
        B.expected_rent(p.square_id, p.houses, p.is_monopoly)
        for p in active
        if p.color not in ("railroad", "utility")
    )
    if railroads:
        rent = B.railroad_rent().get(len(railroads), 200)
        total += sum(B.LAND_PROB[p.square_id] * rent for p in railroads)
    if utilities:
        mult = UTILITY_MULT.get(len(utilities), 10)
        total += sum(B.LAND_PROB[p.square_id] * EXPECTED_DICE * mult for p in utilities)
    return total * rivals


def build_potential(env, pid: int) -> float:
    """Rent per round unlocked by building we can afford right now.

    Only counts progress toward BUILD_TARGET houses, because that is where the
    marginal return per dollar peaks. Capped by actual cash so that a broke
    player holding a monopoly is not credited with rent it cannot buy.
    """
    player = env.players[pid]
    rivals = sum(1 for i, p in enumerate(env.players) if i != pid and not p.bankrupt)
    if not rivals:
        return 0.0
    budget = max(0.0, player.cash - CASH_FLOOR)
    gains = []
    for prop in player.properties:
        if not prop.is_monopoly or prop.mortgaged or not prop.is_real_estate:
            continue
        if prop.houses >= BUILD_TARGET:
            continue
        step_cost = prop.data["house_price"]
        for level in range(prop.houses + 1, BUILD_TARGET + 1):
            gain = (B.expected_rent(prop.square_id, level, True)
                    - B.expected_rent(prop.square_id, level - 1, True))
            gains.append((gain / step_cost, gain, step_cost))
    gains.sort(reverse=True)          # spend the budget on the best steps first
    unlocked = 0.0
    for _, gain, cost in gains:
        if budget < cost:
            break
        budget -= cost
        unlocked += gain
    return unlocked * rivals


def exposure_per_round(env, pid: int) -> float:
    """Rent we expect to PAY per round to everyone else."""
    total = 0.0
    for i, rival in enumerate(env.players):
        if i == pid or rival.bankrupt:
            continue
        total += income_per_round(env, i) / max(
            1, sum(1 for j, p in enumerate(env.players) if j != i and not p.bankrupt))
    return total


def liquid_assets(env, pid: int) -> float:
    """Cash plus what we could raise without going bankrupt."""
    player = env.players[pid]
    total = float(player.cash)
    for prop in player.properties:
        if not prop.mortgaged:
            total += prop.mortgage_v
        total += prop.houses * prop.data.get("house_price", 0) * 0.5
    return total


# ---- hard solvency gate --------------------------------------------------
# We lose 88% of our games against ASU BY BANKRUPTCY. W_RISK only subtracts
# from a score, so a large enough income term can always buy a move that kills
# us. This makes survival a CONSTRAINT rather than a preference: an action whose
# resulting position cannot cover the worst rent reachable on the next roll is
# ineligible, not merely penalised.
#
# "Worst reachable" rather than "expected": bankruptcy is absorbing. An expected
# value is the right way to compare two survivable futures and the wrong way to
# decide whether you survive at all.
SOLVENCY_FLOOR = 0.0     # margin must exceed this; raise to play more safely
SOLVENCY_GATE = False
# MEASURED INERT: flags 7.4% of positions as insolvent but changes 0/108
# decisions. While solvent every candidate stays solvent; once insolvent no
# candidate is solvent and the fallback fires. Bankruptcy against ASU is a slow
# economic slide, not one reckless move, so a single-step filter cannot prevent
# it. Left switchable, off by default -- it costs a worst_reachable_rent scan
# per candidate and buys nothing.


def worst_reachable_rent(env, pid: int) -> float:
    """Largest rent we could owe on the next roll, over dice totals 2..12."""
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
            count = sum(1 for p in env.players[prop.owner].properties
                        if p.color == "railroad" and not p.mortgaged)
            rent = float(B.railroad_rent().get(count, 200))
        elif prop.color == "utility":
            count = sum(1 for p in env.players[prop.owner].properties
                        if p.color == "utility" and not p.mortgaged)
            rent = float(total * UTILITY_MULT.get(count, 10))
        else:
            rent = float(B.rent_at(prop.square_id, prop.houses, prop.is_monopoly))
        worst = max(worst, rent)
    return worst


def raisable(env, pid: int) -> float:
    """Cash we could raise by liquidating -- EXCLUDING cash already in hand."""
    total = 0.0
    for prop in env.players[pid].properties:
        if not prop.mortgaged:
            total += float(prop.mortgage_v)
        total += prop.houses * float(prop.data.get("house_price", 0)) * 0.5
    return total


def solvency_margin(env, pid: int) -> float:
    player = env.players[pid]
    return (float(player.cash) + raisable(env, pid)
            - worst_reachable_rent(env, pid))


def is_solvent(env, pid: int) -> bool:
    """True if we can still pay the worst rent reachable next roll."""
    if env.players[pid].bankrupt:
        return False
    return solvency_margin(env, pid) > SOLVENCY_FLOOR


def evaluate(env, pid: int) -> float:
    """Scalar value of this position for `pid`. Higher is better."""
    player = env.players[pid]
    if player.bankrupt:
        return -1e9

    alive = [p for p in env.players if not p.bankrupt]
    if len(alive) == 1 and alive[0].player_id == pid:
        return 1e9                        # we won

    income = income_per_round(env, pid)
    exposure = exposure_per_round(env, pid)
    potential = build_potential(env, pid)

    value = float(player.net_worth())
    value += W_INCOME * income
    value -= W_DENY * exposure
    value += W_POTENT * potential

    # illiquidity: being unable to cover a plausible rent is what actually kills
    shortfall = max(0.0, (CASH_FLOOR + 3.0 * exposure) - liquid_assets(env, pid))
    value -= W_RISK * shortfall
    return value


def evaluate_all(env) -> list[float]:
    return [evaluate(env, i) for i in range(len(env.players))]
