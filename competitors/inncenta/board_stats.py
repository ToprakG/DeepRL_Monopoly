
from __future__ import annotations

from functools import lru_cache

import numpy as np

from monopoly_game_engine.constants import (
    BOARD, COLOR_GROUPS, GO_TO_JAIL_SQUARE, JAIL_SQUARE, PROPERTIES,
    RAILROAD_IDS, UTILITY_IDS,
)

N_SQUARES = len(BOARD)
MAX_HOUSES = 4
HOTEL_LEVEL = 5


def _dice_pmf() -> np.ndarray:
    pmf = np.zeros(13)
    for a in range(1, 7):
        for b in range(1, 7):
            pmf[a + b] += 1.0 / 36.0
    return pmf


def _landing_probabilities(iterations: int = 20_000) -> np.ndarray:
    pmf = _dice_pmf()
    transition = np.zeros((N_SQUARES, N_SQUARES))
    for src in range(N_SQUARES):
        for roll in range(2, 13):
            dst = (src + roll) % N_SQUARES
            if dst == GO_TO_JAIL_SQUARE:
                dst = JAIL_SQUARE
            transition[src, dst] += pmf[roll]
    dist = np.full(N_SQUARES, 1.0 / N_SQUARES)
    for _ in range(iterations):
        dist = dist @ transition
    return dist / dist.sum()


LAND_PROB = _landing_probabilities()


REAL_ESTATE = tuple(s for s in PROPERTIES if "house_price" in PROPERTIES[s])
COLOR_OF = {s: PROPERTIES[s]["color"] for s in PROPERTIES}

SETS = {
    color: tuple(s for s in squares if s in PROPERTIES and "house_price" in PROPERTIES[s])
    for color, squares in COLOR_GROUPS.items()
}
SETS = {c: sq for c, sq in SETS.items() if sq}


def rent_at(square: int, houses: int, monopoly: bool) -> int:
    data = PROPERTIES[square]
    if "rent" not in data:
        return 0
    if houses > 0:
        return int(data["rent"][min(houses, HOTEL_LEVEL)])
    base = int(data["rent"][0])
    return base * 2 if monopoly else base


def expected_rent(square: int, houses: int, monopoly: bool) -> float:
    return LAND_PROB[square] * rent_at(square, houses, monopoly)


@lru_cache(maxsize=1)
def set_quality() -> dict[str, float]:
    out = {}
    for color, squares in SETS.items():
        cost = sum(PROPERTIES[s]["price"] + HOTEL_LEVEL * PROPERTIES[s]["house_price"]
                   for s in squares)
        rent = sum(expected_rent(s, HOTEL_LEVEL, True) for s in squares)
        out[color] = rent / cost
    return out


@lru_cache(maxsize=4096)
def build_gain(square: int, from_houses: int, to_houses: int) -> float:
    if to_houses <= from_houses:
        return 0.0
    data = PROPERTIES[square]
    cost = (to_houses - from_houses) * data["house_price"]
    gain = expected_rent(square, to_houses, True) - expected_rent(square, from_houses, True)
    return gain / cost if cost else 0.0


@lru_cache(maxsize=1)
def railroad_rent() -> dict[int, int]:
    sample = next(iter(RAILROAD_IDS))
    data = PROPERTIES[sample]
    table = data.get("rent")
    if table:
        return {i + 1: int(table[i]) for i in range(min(4, len(table)))}
    return {1: 25, 2: 50, 3: 100, 4: 200}


def summary() -> str:
    lines = ["colour sets by rent-per-dollar (best first):"]
    for color, eff in sorted(set_quality().items(), key=lambda kv: -kv[1]):
        squares = SETS[color]
        land = sum(LAND_PROB[s] for s in squares)
        cost = sum(PROPERTIES[s]["price"] + HOTEL_LEVEL * PROPERTIES[s]["house_price"]
                   for s in squares)
        lines.append(f"  {color:<10} land={100*land:5.2f}%  cost=${cost:<6,}  eff={eff:.5f}")
    lines.append("")
    lines.append("marginal rent per dollar by house step (orange, Tennessee Ave):")
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)):
        lines.append(f"  {a}->{b} houses: {build_gain(18, a, b):.4f}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
