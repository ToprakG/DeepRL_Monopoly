"""UNDERDOG (ChampionPlus) from github.com/denizzmcr/DeepRL_Monopoly.

Their heuristic package aliases its vendored ``engine/`` as
``monopoly_game_engine`` on import. Occupy our engine in ``sys.modules``
first so H2H keeps a single MonopolyEnv class.
"""

from __future__ import annotations

import sys
from pathlib import Path

UNDERDOG_ID = "underdog-v1"
_SRC = (
    Path(__file__).resolve().parents[1]
    / "artifacts_scratch"
    / "denizzmcr_underdog"
    / "external"
    / "kuzey"
    / "Kuzeys_heuristic"
)


def _champion_plus():
    import monopoly_game_engine  # noqa: F401
    import monopoly_game_engine.actions  # noqa: F401
    import monopoly_game_engine.constants  # noqa: F401
    import monopoly_game_engine.env  # noqa: F401

    if not _SRC.is_dir():
        raise FileNotFoundError(f"UNDERDOG heuristic not found at {_SRC}")
    root = str(_SRC)
    if root not in sys.path:
        sys.path.insert(0, root)
    from heuristic import ChampionPlus

    return ChampionPlus


class UnderdogAgent:
    def __init__(self, player_id: int):
        self.player_id = int(player_id)
        self._agent = _champion_plus()()

    def choose_action(self, env) -> int:
        return int(self._agent.choose_action(env, self.player_id, 0))
