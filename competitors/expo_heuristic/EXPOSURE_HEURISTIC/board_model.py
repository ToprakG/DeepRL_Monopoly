"""
Board traffic model for the ``ppo-plus-v2`` simulator.

This module answers one question: *how often does a moving player actually
land on each of the 40 squares?*  Monopoly's board is not uniform.  The
"Go To Jail" square at 30 teleports players back to 10, which pumps extra
traffic into everything roughly 5-9 squares past the jail (the orange and
red groups).  Any rent estimate that assumes "one lap = one visit to every
square" systematically misprices those groups.

We compute the answer exactly instead of guessing it, by building the
40x40 transition matrix of the *actual* engine dynamics in
``monopoly_game_engine/env.py`` and taking its stationary distribution:

  - two fair dice, all 36 ordered outcomes;
  - landing on square 30 sends the token to square 10;
  - three consecutive doubles sends the token to square 10, folded in as a
    small extra jail inflow since the doubles chain restarts every turn;
  - no Chance / Community Chest movement, because this ruleset gives those
    squares no card effect at all (see PPO_PLUS_RULES.md).

Everything here is deterministic and computed once at import.
"""

from __future__ import annotations

import numpy as np

BOARD_SIZE = 40
JAIL_SQUARE = 10
GO_TO_JAIL_SQUARE = 30

# Probability of each two-dice total, 2..12.
_DICE_TOTAL_P: dict[int, float] = {
    total: (6 - abs(total - 7)) / 36.0 for total in range(2, 13)
}
# P(a roll is doubles) = 6/36.  Three in a row ends the turn in jail.
_P_DOUBLES = 6.0 / 36.0
_P_TRIPLE_DOUBLES = _P_DOUBLES ** 3


def _transition_matrix() -> np.ndarray:
    """One-move transition matrix including the Go-To-Jail teleport."""
    matrix = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float64)
    for origin in range(BOARD_SIZE):
        for total, probability in _DICE_TOTAL_P.items():
            destination = (origin + total) % BOARD_SIZE
            if destination == GO_TO_JAIL_SQUARE:
                destination = JAIL_SQUARE
            matrix[origin, destination] += probability
        # Fold the triple-doubles jail rule in as a small uniform leak.
        matrix[origin] *= 1.0 - _P_TRIPLE_DOUBLES
        matrix[origin, JAIL_SQUARE] += _P_TRIPLE_DOUBLES
    return matrix


def _stationary(matrix: np.ndarray, iterations: int = 4000) -> np.ndarray:
    """Power-iterate to the stationary landing distribution."""
    distribution = np.full(BOARD_SIZE, 1.0 / BOARD_SIZE, dtype=np.float64)
    for _ in range(iterations):
        nxt = distribution @ matrix
        if np.abs(nxt - distribution).max() < 1e-14:
            distribution = nxt
            break
        distribution = nxt
    return distribution / distribution.sum()


#: P(a given move ends on square s).  Index by board square.
LANDING_PROBABILITY: np.ndarray = _stationary(_transition_matrix())

#: Mean number of moves a player makes per lap of the board.
MOVES_PER_LAP: float = BOARD_SIZE / 7.0


def landing_odds(square: int) -> float:
    """Probability that one opponent move ends on ``square``."""
    return float(LANDING_PROBABILITY[square])


def visits_per_lap(square: int) -> float:
    """Expected times one opponent lands on ``square`` per lap of the board."""
    return float(LANDING_PROBABILITY[square]) * MOVES_PER_LAP


__all__ = [
    "LANDING_PROBABILITY",
    "MOVES_PER_LAP",
    "landing_odds",
    "visits_per_lap",
]
