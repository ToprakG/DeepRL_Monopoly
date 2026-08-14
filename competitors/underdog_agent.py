"""UNDERDOG GBM submission from github.com/denizzmcr/DeepRL_Monopoly.

Occupies our engine in ``sys.modules`` first so their heuristic fallback
cannot bind ``underdog/engine`` as ``monopoly_game_engine``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Torch and LightGBM both ship OpenMP. On macOS that aborts the process
# unless this is set before LightGBM loads.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

UNDERDOG_ID = "underdog-v1"
_SRC = Path(__file__).resolve().parent / "underdog_src"


def _load_agent_class():
    import monopoly_game_engine  # noqa: F401
    import monopoly_game_engine.actions  # noqa: F401
    import monopoly_game_engine.constants  # noqa: F401
    import monopoly_game_engine.env  # noqa: F401
    import monopoly_game_engine.state  # noqa: F401

    root = str(_SRC)
    if root not in sys.path:
        sys.path.insert(0, root)
    from agent import Agent

    return Agent


class UnderdogAgent:
    def __init__(self, player_id: int):
        self.player_id = int(player_id)
        self._agent = _load_agent_class()(player_id)

    def choose_action(self, env) -> int:
        return int(
            self._agent.choose_action(
                env=env,
                player_id=self.player_id,
                allowed_actions=list(env.get_allowed_actions(self.player_id)),
            )
        )
