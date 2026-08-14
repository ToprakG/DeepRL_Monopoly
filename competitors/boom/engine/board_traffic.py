"""
Exact board landing-probability model.

Own implementation of a standard technique (Markov-chain stationary
distribution over the 40 squares, computed once at import time) — not
derived from ASU or any other agent's code. Used to replace ad-hoc,
single-colour-group buffers in our buy/auction heuristics with a precise,
per-square landing probability.
"""

import numpy as np

from .constants import GO_TO_JAIL_SQUARE, JAIL_SQUARE

_BOARD_SIZE = 40
_DICE_TOTAL_P = {total: (6 - abs(total - 7)) / 36.0 for total in range(2, 13)}


def _transition_matrix() -> np.ndarray:
    matrix = np.zeros((_BOARD_SIZE, _BOARD_SIZE), dtype=np.float64)
    for origin in range(_BOARD_SIZE):
        for total, probability in _DICE_TOTAL_P.items():
            destination = (origin + total) % _BOARD_SIZE
            if destination == GO_TO_JAIL_SQUARE:
                destination = JAIL_SQUARE
            matrix[origin, destination] += probability
    return matrix


def _stationary(matrix: np.ndarray, iterations: int = 4000) -> np.ndarray:
    distribution = np.full(_BOARD_SIZE, 1.0 / _BOARD_SIZE, dtype=np.float64)
    for _ in range(iterations):
        nxt = distribution @ matrix
        if np.abs(nxt - distribution).max() < 1e-14:
            distribution = nxt
            break
        distribution = nxt
    return distribution / distribution.sum()


LANDING_PROBABILITY: np.ndarray = _stationary(_transition_matrix())
_MEAN_PROBABILITY: float = float(LANDING_PROBABILITY.mean())


def landing_odds(square: int) -> float:
    """Probability that a single move ends on this square."""
    return float(LANDING_PROBABILITY[square])


def landing_relative(square: int) -> float:
    """This square's landing odds relative to the board average (1.0 = average)."""
    return float(LANDING_PROBABILITY[square]) / _MEAN_PROBABILITY
