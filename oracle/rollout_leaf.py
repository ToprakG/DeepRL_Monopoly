"""Truncated greedy rollouts → bounded per-player leaf value vector."""

from __future__ import annotations

import random
from typing import Callable

import numpy as np

from monopoly_bench.engine import NUM_PLAYERS, clone_env, terminal_value
from monopoly_game_engine.env import MonopolyEnv

from .rollout_policy import greedy_rollout_action

PolicyFn = Callable[[MonopolyEnv, int], int]

DEFAULT_HORIZON = 30
DEFAULT_ROLLOUTS = 2
DEFAULT_MARGIN_TEMPERATURE = 2000.0


def margin_vector_from_scores(
    scores: np.ndarray,
    env: MonopolyEnv,
    *,
    temperature: float = DEFAULT_MARGIN_TEMPERATURE,
) -> np.ndarray:
    """Softmax of (own score − best opponent score) / temperature."""

    if env.done:
        return terminal_value(env).astype(np.float64)

    worth = np.asarray(scores, dtype=np.float64)
    if worth.shape != (NUM_PLAYERS,):
        raise ValueError(f"Expected {NUM_PLAYERS} scores, got {worth.shape}")
    margins = np.empty(NUM_PLAYERS, dtype=np.float64)
    for seat in range(NUM_PLAYERS):
        others = [worth[j] for j in range(NUM_PLAYERS) if j != seat]
        best_opp = max(others) if others else 0.0
        margins[seat] = worth[seat] - best_opp
    for seat, player in enumerate(env.players):
        if player.bankrupt:
            margins[seat] = -1.0e9
    scale = max(float(temperature), 1.0)
    shifted = margins - np.max(margins)
    exp = np.exp(shifted / scale)
    total = float(exp.sum())
    if total <= 0 or not np.isfinite(total):
        return np.full(NUM_PLAYERS, 1.0 / NUM_PLAYERS, dtype=np.float64)
    return exp / total


def net_worth_margin_vector(
    env: MonopolyEnv,
    *,
    temperature: float = DEFAULT_MARGIN_TEMPERATURE,
) -> np.ndarray:
    """Softmax of (own net worth − best opponent net worth) / temperature."""

    if env.done:
        return terminal_value(env).astype(np.float64)

    worth = np.asarray(
        [float(player.net_worth()) for player in env.players],
        dtype=np.float64,
    )
    return margin_vector_from_scores(worth, env, temperature=temperature)


def _one_rollout(
    env: MonopolyEnv,
    *,
    horizon: int,
    policy: PolicyFn,
    rng: random.Random,
) -> np.ndarray:
    cloned = clone_env(env)
    outer = random.getstate()
    try:
        random.setstate(rng.getstate())
        decisions = 0
        while not cloned.done and decisions < horizon:
            actor = cloned.whose_turn()
            action = policy(cloned, actor)
            legal = cloned.get_allowed_actions(actor)
            if action not in legal:
                action = legal[0]
            cloned.step(action)
            decisions += 1
        rng.setstate(random.getstate())
    finally:
        random.setstate(outer)
    return net_worth_margin_vector(cloned)


def rollout_leaf_value(
    env: MonopolyEnv,
    num_rollouts: int = DEFAULT_ROLLOUTS,
    horizon: int = DEFAULT_HORIZON,
    *,
    policy: PolicyFn = greedy_rollout_action,
    temperature: float = DEFAULT_MARGIN_TEMPERATURE,
    seed: int | None = None,
) -> np.ndarray:
    """Average ``num_rollouts`` truncated greedy trajectories into a 4-vector."""

    if num_rollouts < 1:
        raise ValueError("num_rollouts must be positive")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if env.done:
        return terminal_value(env).astype(np.float64)

    rng = random.Random(seed)
    acc = np.zeros(NUM_PLAYERS, dtype=np.float64)
    for index in range(num_rollouts):
        child = random.Random(rng.randint(0, 2**31 - 1))
        acc += _one_rollout(env, horizon=horizon, policy=policy, rng=child)
    return acc / float(num_rollouts)


__all__ = [
    "DEFAULT_HORIZON",
    "DEFAULT_MARGIN_TEMPERATURE",
    "DEFAULT_ROLLOUTS",
    "margin_vector_from_scores",
    "net_worth_margin_vector",
    "rollout_leaf_value",
]
