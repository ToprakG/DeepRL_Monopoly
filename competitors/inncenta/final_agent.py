
from __future__ import annotations

from . import heuristic_agent as _ha

from monopoly_game_engine.actions import ActionType


VALUE_FUNCTION = "asu"


class Agent:

    name = "monopoly_agent"

    def __init__(self, player_id: int):
        self.player_id = player_id
        _ha.use_value_fn(VALUE_FUNCTION)
        self._inner = _ha.HeuristicAgent(player_id)

    def choose_action(self, env) -> int:
        legal = list(env.get_allowed_actions(self.player_id))
        if not legal:
            return int(ActionType.END_TURN)
        if len(legal) == 1:
            return legal[0]
        try:
            action = self._inner.choose_action(env)
        except Exception:
            return legal[0]
        return action if action in legal else legal[0]


MonopolyAgent = Agent
Player = Agent
