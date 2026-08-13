"""Semantically equivalent env clone; pickle is much cheaper than deepcopy."""

from __future__ import annotations

import pickle

from monopoly_bench.engine import SharedGame, unwrap
from monopoly_game_engine.env import MonopolyEnv


def fast_clone_env(game: MonopolyEnv | SharedGame) -> MonopolyEnv:
    return pickle.loads(pickle.dumps(unwrap(game), protocol=pickle.HIGHEST_PROTOCOL))
