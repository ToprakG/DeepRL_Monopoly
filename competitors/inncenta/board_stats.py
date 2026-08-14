"""Board statistics for ppo-plus-v2, derived from the engine's own constants.

Everything here is computed from BOARD/PROPERTIES/COLOR_GROUPS, not copied from
any existing agent. Two facts drive the whole evaluator:

1. Squares are NOT equally likely. Average roll is 7 and Go-To-Jail(30) pumps
   probability into Jail(10), so squares 6-9 past Jail are the most-landed-on.
   Orange is the best set on this board; dark blue is 7th of 8 on rent-per-dollar.

2. The 2->3 house step returns ~3.5x per dollar, roughly 2.7x better than any
   other increment. Combined with this engine NOT enforcing the even-build rule
   (env.py:911-941 has no such check), the right policy is to bring every
   monopoly square to exactly 3 houses before adding a 4th anywhere.

Caveat: Chance/Community-Chest teleports are not modelled. They push further
toward Illinois/Boardwalk/railroads, so the orange edge below is a lower bound.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

def _find_repo() -> Path:
    """Locate DeepRL_Monopoly without hardcoding anyone's home directory.

    Order: DEEPRL_MONOPOLY_ROOT env var, already-importable, then walk up from
    this file looking for a directory containing monopoly_game_engine.
    """
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
        for sib in parent.iterdir() if parent.is_dir() else []:
            if sib.is_dir() and (sib / "monopoly_game_engine").is_dir():
                return sib
    raise RuntimeError(
        "Could not locate DeepRL_Monopoly. Set DEEPRL_MONOPOLY_ROOT, or place "
        "this directory next to the DeepRL_Monopoly checkout.")


ROOT = _find_repo()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monopoly_game_engine.constants import (  # noqa: E402
    BOARD, COLOR_GROUPS, GO_TO_JAIL_SQUARE, JAIL_SQUARE, PROPERTIES,
    RAILROAD_IDS, UTILITY_IDS,
)

N_SQUARES = len(BOARD)
MAX_HOUSES = 4          # 5 == hotel in the rent table
HOTEL_LEVEL = 5


def _dice_pmf() -> np.ndarray:
    pmf = np.zeros(13)
    for a in range(1, 7):
        for b in range(1, 7):
            pmf[a + b] += 1.0 / 36.0
    return pmf


def _landing_probabilities(iterations: int = 20_000) -> np.ndarray:
    """Stationary distribution of a 2d6 walk with Go-To-Jail redirection."""
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

#: real-estate squares only (railroads and utilities have no house_price)
REAL_ESTATE = tuple(s for s in PROPERTIES if "house_price" in PROPERTIES[s])
COLOR_OF = {s: PROPERTIES[s]["color"] for s in PROPERTIES}
#: colour -> squares, restricted to buildable sets
SETS = {
    color: tuple(s for s in squares if s in PROPERTIES and "house_price" in PROPERTIES[s])
    for color, squares in COLOR_GROUPS.items()
}
SETS = {c: sq for c, sq in SETS.items() if sq}


def rent_at(square: int, houses: int, monopoly: bool) -> int:
    """Rent this square charges. Undeveloped monopolies pay double base rent."""
    data = PROPERTIES[square]
    if "rent" not in data:
        return 0
    if houses > 0:
        return int(data["rent"][min(houses, HOTEL_LEVEL)])
    base = int(data["rent"][0])
    return base * 2 if monopoly else base


def expected_rent(square: int, houses: int, monopoly: bool) -> float:
    """Rent per opponent-turn: landing probability x rent charged."""
    return LAND_PROB[square] * rent_at(square, houses, monopoly)


@lru_cache(maxsize=1)
def set_quality() -> dict[str, float]:
    """Rent per dollar for a fully-hotelled set -- how good the colour is.

    orange > lightblue > yellow > pink > red > darkblue > brown > green
    """
    out = {}
    for color, squares in SETS.items():
        cost = sum(PROPERTIES[s]["price"] + HOTEL_LEVEL * PROPERTIES[s]["house_price"]
                   for s in squares)
        rent = sum(expected_rent(s, HOTEL_LEVEL, True) for s in squares)
        out[color] = rent / cost
    return out


@lru_cache(maxsize=4096)
def build_gain(square: int, from_houses: int, to_houses: int) -> float:
    """Expected rent gained per dollar spent taking a square from N to M houses."""
    if to_houses <= from_houses:
        return 0.0
    data = PROPERTIES[square]
    cost = (to_houses - from_houses) * data["house_price"]
    gain = expected_rent(square, to_houses, True) - expected_rent(square, from_houses, True)
    return gain / cost if cost else 0.0


@lru_cache(maxsize=1)
def railroad_rent() -> dict[int, int]:
    """Railroad rent by count owned, read from the engine rather than assumed."""
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
