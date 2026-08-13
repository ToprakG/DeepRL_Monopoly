
from __future__ import annotations

from . import board_stats as B


W_INCOME = 26.0
W_DENY = 9.0
W_POTENT = 14.0
W_RISK = 2.5
CASH_FLOOR = 200
AUCTION_RESERVE_FRAC = 0.15


HARD_GATE = None

BUILD_TARGET = 3
UTILITY_MULT = {1: 4, 2: 10}
EXPECTED_DICE = 7.0


def income_per_round(env, pid: int) -> float:
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
    gains.sort(reverse=True)
    unlocked = 0.0
    for _, gain, cost in gains:
        if budget < cost:
            break
        budget -= cost
        unlocked += gain
    return unlocked * rivals


def exposure_per_round(env, pid: int) -> float:
    total = 0.0
    for i, rival in enumerate(env.players):
        if i == pid or rival.bankrupt:
            continue
        total += income_per_round(env, i) / max(
            1, sum(1 for j, p in enumerate(env.players) if j != i and not p.bankrupt))
    return total


def liquid_assets(env, pid: int) -> float:
    player = env.players[pid]
    total = float(player.cash)
    for prop in player.properties:
        if not prop.mortgaged:
            total += prop.mortgage_v
        total += prop.houses * prop.data.get("house_price", 0) * 0.5
    return total


SOLVENCY_FLOOR = 0.0
SOLVENCY_GATE = False


def worst_reachable_rent(env, pid: int) -> float:
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
    if env.players[pid].bankrupt:
        return False
    return solvency_margin(env, pid) > SOLVENCY_FLOOR


def evaluate(env, pid: int) -> float:
    player = env.players[pid]
    if player.bankrupt:
        return -1e9

    alive = [p for p in env.players if not p.bankrupt]
    if len(alive) == 1 and alive[0].player_id == pid:
        return 1e9

    income = income_per_round(env, pid)
    exposure = exposure_per_round(env, pid)
    potential = build_potential(env, pid)

    value = float(player.net_worth())
    value += W_INCOME * income
    value -= W_DENY * exposure
    value += W_POTENT * potential


    shortfall = max(0.0, (CASH_FLOOR + 3.0 * exposure) - liquid_assets(env, pid))
    value -= W_RISK * shortfall
    return value


def evaluate_all(env) -> list[float]:
    return [evaluate(env, i) for i in range(len(env.players))]
