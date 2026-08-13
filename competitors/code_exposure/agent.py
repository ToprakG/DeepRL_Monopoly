"""Competition entrypoint."""

from __future__ import annotations

from typing import Any, Sequence

from ASU_FROZEN_TEACHER import ASUValueV1
from monopoly_game_engine.actions import AUCTION_ACTION_TO_INCREMENT, AuctionAction
from monopoly_game_engine.env import PHASE_AUCTION

_RAISES = frozenset(int(a) for a in AUCTION_ACTION_TO_INCREMENT)


def _is_env(obj: Any) -> bool:
    return hasattr(obj, "get_allowed_actions") and hasattr(obj, "players")


def _is_actions(obj: Any) -> bool:
    if isinstance(obj, (str, bytes)) or not isinstance(obj, Sequence):
        return False
    if not len(obj):
        return False
    return all(isinstance(x, int) and not isinstance(x, bool) for x in obj)


class Agent:
    """Accepts every call shape used by the reference agents.

    FixedPolicyAgent.choose_action(env)
    DDQNAgent/PPOAgent.choose_action(state, env, allowed_actions)
    arena adapters: choose_action(game, player_id, decision_seed)
    """

    def __init__(self, player_id: int = 0, **kwargs: Any) -> None:
        for key in ("player_id", "pid", "agent_id", "seat"):
            if kwargs.get(key) is not None:
                player_id = int(kwargs[key])
                break
        self.player_id = int(player_id)
        self._base: dict[int, ASUValueV1] = {}

    def _base_for(self, seat: int) -> ASUValueV1:
        base = self._base.get(seat)
        if base is None:
            base = ASUValueV1(seat)
            self._base[seat] = base
        return base

    def choose_action(self, *args: Any, **kwargs: Any) -> int:
        env = kwargs.get("env")
        allowed = kwargs.get("allowed_actions")
        for arg in args:
            if env is None:
                candidate = getattr(arg, "env", arg)
                if _is_env(candidate):
                    env = candidate
                    continue
            if allowed is None and _is_actions(arg):
                allowed = list(arg)

        if env is None:
            if allowed:
                return int(allowed[0])
            raise TypeError("choose_action needs the environment or allowed actions")

        # The engine only asks an agent to act on its own turn, so the acting
        # seat comes from the engine rather than from construction.
        seat = self.player_id
        try:
            seat = int(env.whose_turn())
        except Exception:
            pass

        if allowed is None:
            allowed = list(env.get_allowed_actions(seat))
        if len(allowed) == 1:
            return int(allowed[0])

        try:
            action = int(self._base_for(seat).decide(env).selected_action)
            if env.phase == PHASE_AUCTION and action in _RAISES:
                bids = [(AUCTION_ACTION_TO_INCREMENT[AuctionAction(a)], a)
                        for a in allowed if a in _RAISES]
                if bids:
                    action = int(min(bids)[1])
        except Exception:
            return int(allowed[0])

        return action if action in allowed else int(allowed[0])


def make_agent(player_id: int = 0, **kwargs: Any) -> Agent:
    return Agent(player_id, **kwargs)


_DEFAULT: Agent | None = None


def choose_action(*args: Any, **kwargs: Any) -> int:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Agent()
    return _DEFAULT.choose_action(*args, **kwargs)
