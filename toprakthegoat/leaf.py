"""Opponent-aware leaf for goat Max-N. Engine net-worth plus set / 3H / cash risk."""

from __future__ import annotations

import numpy as np

from monopoly_game_engine.constants import COLOR_GROUPS, NUM_PLAYERS
from monopoly_game_engine.env import MonopolyEnv

from .helpers import REAL_COLOURS, max_live_rent
from oracle.rollout_leaf import margin_vector_from_scores


SET_BONUS = 2500.0
HOUSE_BONUS = 400.0
CASH_SHORT_WEIGHT = 1.5
THREE = 3


def seat_threat_score(env: MonopolyEnv, seat: int) -> float:
    player = env.players[seat]
    if player.bankrupt:
        return -1.0e12
    score = float(player.net_worth())
    best_houses = 0
    for color in REAL_COLOURS:
        group = COLOR_GROUPS.get(color) or ()
        if not group or any(env.properties[sq].owner != seat for sq in group):
            continue
        score += SET_BONUS
        best_houses = max(
            best_houses,
            max(int(getattr(env.properties[sq], "houses", 0) or 0) for sq in group),
        )
    score += HOUSE_BONUS * min(best_houses, THREE)
    short = max(0.0, float(max_live_rent(env, seat)) - float(player.cash))
    return score - CASH_SHORT_WEIGHT * short


def threat_leaf(env: MonopolyEnv, *, temperature: float = 2000.0) -> np.ndarray:
    scores = np.array(
        [seat_threat_score(env, seat) for seat in range(NUM_PLAYERS)],
        dtype=np.float64,
    )
    return margin_vector_from_scores(scores, env, temperature=temperature)


__all__ = ["seat_threat_score", "threat_leaf"]
