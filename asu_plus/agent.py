"""ASU+ agent: ASUValueV1 scaffolding with an augmented value total."""

from __future__ import annotations

from ASU_FROZEN_TEACHER.core import ASUValueV1  # noqa: E402
from ASU_FROZEN_TEACHER.types import ValueBreakdown  # noqa: E402
from monopoly_game_engine.constants import NUM_PLAYERS  # noqa: E402

from .value import ASUPlusWeights, evaluate_value_plus

ASU_PLUS_V1 = "asu-plus-v1"


class ASUPlusV1(ASUValueV1):
    """One-step ASU+ teacher — inherits legality, safety, trade, and auction logic."""

    policy_id = ASU_PLUS_V1

    def __init__(self, player_id: int, weights: ASUPlusWeights | None = None):
        if not 0 <= player_id < NUM_PLAYERS:
            raise ValueError(f"player_id must be in [0, {NUM_PLAYERS - 1}]")
        super().__init__(player_id)
        self.weights = weights if weights is not None else ASUPlusWeights()

    def value(self, env, player_id: int | None = None) -> ValueBreakdown:
        pid = self.player_id if player_id is None else player_id
        return evaluate_value_plus(env, pid, self.weights)


__all__ = ["ASU_PLUS_V1", "ASUPlusV1"]
