"""Inncenta 1-ply net-worth heuristic, from github.com/Inncenta/monopoly."""

from .agent import Agent as _InncentaCore

__all__ = ["InncentaAgent"]


class InncentaAgent:
    """Seat-bound wrapper. Their core keys off ``whose_turn()``."""

    def __init__(self, player_id: int):
        self.player_id = int(player_id)
        self._inner = _InncentaCore(player_id)

    def choose_action(self, env) -> int:
        allowed = list(env.get_allowed_actions(self.player_id))
        if not allowed:
            return 0
        return int(self._inner.choose_action(env, allowed))
