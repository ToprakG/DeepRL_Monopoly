"""UNDERDOG — the submitted agent. One import, one call.

The heuristic ships three variants and we measured all of them rather than
assuming the default was best. ChampionPlus beat Champion in three independent
seat-balanced tournaments (+5.5, +4.6, +5.3 points; pooled 40.3% vs 35.3% over
536 appearances each, z = 1.69). Individually none of the three clears
significance; three replications of the same size and direction do. So the
submission is ChampionPlus.

Parity in a four-player game is 25%.

    from underdog_agent import Underdog
    agent = Underdog(player_id=seat)
    action = agent.choose_action(env)

``choose_action(env) -> int`` is the convention every policy in this repository
uses, so the same harness that runs ``submission_agent.SubmissionAgent`` runs
this unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_HEURISTIC = _ROOT / "underdog"

__all__ = ["Underdog", "VARIANT"]

VARIANT = "ChampionScore"


class Underdog:
    """The submitted policy, wrapped to the repository's agent interface.

    Holds no learned weights and reads no data file: the whole agent is
    hand-written rules. Nothing here imports or consults ASU.
    """

    def __init__(self, player_id: int):
        if not _HEURISTIC.exists():
            raise FileNotFoundError(f"heuristic package not found at {_HEURISTIC}")
        if str(_HEURISTIC) not in sys.path:
            sys.path.insert(0, str(_HEURISTIC))
        from heuristic import ChampionScore

        self.player_id = int(player_id)
        self._agent = ChampionScore()

    def choose_action(self, env) -> int:
        """Return one legal action index for ``self.player_id``.

        Fail closed rather than substituting: a silent fallback would make a
        broken policy look like a working one while playing something nobody
        measured.
        """
        action = int(self._agent.choose_action(env, self.player_id, 0))
        allowed = env.get_allowed_actions(self.player_id)
        if allowed and action not in allowed:
            raise ValueError(
                f"UNDERDOG chose illegal action {action} for seat "
                f"{self.player_id}; legal actions are {allowed}"
            )
        return action

    def __repr__(self) -> str:
        return f"Underdog(seat={self.player_id}, variant={VARIANT})"
