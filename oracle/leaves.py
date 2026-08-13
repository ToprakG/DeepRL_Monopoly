"""Pluggable Max-N leaves for oracle H2H experiments.

``rollout`` is the calibrated teacher (greedy DealBuilder trajectories, then
net-worth margin). The others are Wave-1 swaps: score the current position
without a noisy 16-step rollout.

Leaf functions return a physical 4-vector that sums to 1.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np

from monopoly_bench.engine import NUM_PLAYERS, terminal_value
from monopoly_game_engine.env import MonopolyEnv

from .agent import OracleConfig
from .rollout_leaf import (
    margin_vector_from_scores,
    net_worth_margin_vector,
    rollout_leaf_value,
)
from .rollout_policy import greedy_rollout_action

LEAF_ROLLOUT = "rollout"
LEAF_NETWORTH = "networth"
LEAF_ASU = "asu"
LEAF_ASU_PLUS = "asu_plus"
LEAF_CLONE = "clone"
LEAF_KINDS = (
    LEAF_ROLLOUT,
    LEAF_NETWORTH,
    LEAF_ASU,
    LEAF_ASU_PLUS,
    LEAF_CLONE,
)
DEFAULT_CLONE_CHECKPOINT = Path(
    "monopoly_bench/runs/oracle_hybrid_bc_new25k/snapshots/hybrid_clone_0000.pt"
)

LeafFn = Callable[[MonopolyEnv], np.ndarray]
_CLONE_MODELS: dict[str, object] = {}


def _scorer_leaf(
    env: MonopolyEnv,
    *,
    scorer: Callable[[MonopolyEnv, int], float],
    temperature: float,
) -> np.ndarray:
    if env.done:
        return terminal_value(env).astype(np.float64)
    scores = np.array(
        [float(scorer(env, seat)) for seat in range(NUM_PLAYERS)],
        dtype=np.float64,
    )
    return margin_vector_from_scores(scores, env, temperature=temperature)


def networth_leaf(env: MonopolyEnv, *, temperature: float) -> np.ndarray:
    return net_worth_margin_vector(env, temperature=temperature)


def asu_leaf(env: MonopolyEnv, *, temperature: float) -> np.ndarray:
    from ASU_FROZEN_TEACHER import evaluate_value

    return _scorer_leaf(
        env,
        scorer=lambda game, seat: evaluate_value(game, seat).total,
        temperature=temperature,
    )


def asu_plus_leaf(env: MonopolyEnv, *, temperature: float) -> np.ndarray:
    from asu_plus import evaluate_value_plus

    return _scorer_leaf(
        env,
        scorer=lambda game, seat: evaluate_value_plus(game, seat).total,
        temperature=temperature,
    )


def _clone_model(path: str):
    model = _CLONE_MODELS.get(path)
    if model is None:
        from monopoly_bench.model import MonopolyZeroNet

        model = MonopolyZeroNet.load_inference(path, device="cpu")
        _CLONE_MODELS[path] = model
    return model


def clone_leaf(env: MonopolyEnv, *, checkpoint: str) -> np.ndarray:
    if env.done:
        return terminal_value(env).astype(np.float64)
    model = _clone_model(checkpoint)
    actor = int(env.whose_turn())
    legal = env.get_allowed_actions(actor)
    if not legal:
        legal = [0]
    _priors, value = model.predict(env._get_state(actor), legal, actor, env=env)
    return np.asarray(value, dtype=np.float64)


def _seeded_rollout_leaf(
    env: MonopolyEnv,
    *,
    num_rollouts: int,
    horizon: int,
    temperature: float,
) -> np.ndarray:
    from oracle_v2.position import key_seed, position_key

    return rollout_leaf_value(
        env,
        num_rollouts=num_rollouts,
        horizon=horizon,
        temperature=temperature,
        seed=key_seed(position_key(env)),
        policy=greedy_rollout_action,
    )


def resolve_clone_checkpoint(path: str | Path | None) -> str:
    candidate = Path(path) if path else DEFAULT_CLONE_CHECKPOINT
    if not candidate.is_file():
        raise FileNotFoundError(f"clone leaf checkpoint missing: {candidate}")
    return str(candidate.resolve())


def build_leaf_fn(config: OracleConfig) -> LeafFn:
    """Build the Max-N leaf implied by ``config.leaf``."""

    kind = str(getattr(config, "leaf", LEAF_ROLLOUT) or LEAF_ROLLOUT)
    if kind not in LEAF_KINDS:
        raise ValueError(f"Unknown oracle leaf {kind!r}; expected one of {LEAF_KINDS}")
    temperature = float(config.margin_temperature)
    if kind == LEAF_ROLLOUT:
        return partial(
            _seeded_rollout_leaf,
            num_rollouts=config.rollouts_per_leaf,
            horizon=config.rollout_horizon,
            temperature=temperature,
        )
    if kind == LEAF_NETWORTH:
        return partial(networth_leaf, temperature=temperature)
    if kind == LEAF_ASU:
        return partial(asu_leaf, temperature=temperature)
    if kind == LEAF_ASU_PLUS:
        return partial(asu_plus_leaf, temperature=temperature)
    checkpoint = resolve_clone_checkpoint(getattr(config, "leaf_checkpoint", None))
    return partial(clone_leaf, checkpoint=checkpoint)


__all__ = [
    "DEFAULT_CLONE_CHECKPOINT",
    "LEAF_ASU",
    "LEAF_ASU_PLUS",
    "LEAF_CLONE",
    "LEAF_KINDS",
    "LEAF_NETWORTH",
    "LEAF_ROLLOUT",
    "asu_leaf",
    "asu_plus_leaf",
    "build_leaf_fn",
    "clone_leaf",
    "networth_leaf",
    "resolve_clone_checkpoint",
]
