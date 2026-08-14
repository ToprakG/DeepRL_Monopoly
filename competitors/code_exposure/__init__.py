"""6c0de NEMESIS (github.com/6c0de/exposure-monopoly-agent)."""

from .agent import Agent as _NemesisCore

__all__ = ["CodeExposureAgent"]


class CodeExposureAgent:
    """Seat-bound wrapper. Their core prefers ``whose_turn()`` when given an env."""

    def __init__(self, player_id: int):
        self.player_id = int(player_id)
        self._inner = _NemesisCore(player_id)

    def choose_action(self, env) -> int:
        allowed = list(env.get_allowed_actions(self.player_id))
        if not allowed:
            return 0
        return int(
            self._inner.choose_action(
                env=env,
                player_id=self.player_id,
                allowed_actions=allowed,
            )
        )
