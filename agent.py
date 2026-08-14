"""Tournament entry: ``choose_action(state, player_id, allowed_actions) -> int``.

The harness pins this file and hands a read-only decision state: actor-relative
300-float ``vector``, readable ``board``, legal action ids. A raise, a 2s
timeout, or an illegal id is a strike.
"""

from __future__ import annotations

import sys
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from toprakthegoat.bootstrap import install_engine_namespace

install_engine_namespace(_ROOT)

from monopoly_game_engine.actions import ACTION_SPACE_SIZE, ActionType
from monopoly_game_engine.constants import NUM_PLAYERS
from monopoly_game_engine.env import MonopolyEnv
from monopoly_game_engine.state import STATE_DIM

from toprakthegoat.agent import heuristic_action
from toprakthegoat.hydrate import hydrate_env

__all__ = ["Agent", "choose_action", "make_agent"]

_END_TURN = int(ActionType.END_TURN)
_VECTOR_KEYS = ("vector", "observation", "obs", "state_vector", "stateVector")
_BOARD_KEYS = ("board", "board_view", "boardView", "table")
_ACTIONS_KEYS = (
    "allowed_actions",
    "allowedActions",
    "legal_actions",
    "legalActions",
    "actions",
)


def _looks_like_env(obj: Any) -> bool:
    return all(
        hasattr(obj, name)
        for name in ("get_allowed_actions", "players", "properties")
    )


def _field(obj: Any, names: Sequence[str]) -> Any:
    if obj is None:
        return None
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _as_seat(obj: Any) -> int | None:
    if isinstance(obj, bool):
        return None
    try:
        if isinstance(obj, Integral) or (
            isinstance(obj, Real) and float(obj).is_integer()
        ):
            seat = int(obj)
            if 0 <= seat < NUM_PLAYERS:
                return seat
    except (TypeError, ValueError, OverflowError):
        return None
    return None


def _as_int_list(obj: Any) -> list[int] | None:
    if isinstance(obj, (str, bytes, dict)) or obj is None:
        return None
    try:
        items = list(obj)
    except TypeError:
        return None
    if not items or len(items) > ACTION_SPACE_SIZE:
        return None
    out: list[int] = []
    for item in items:
        if isinstance(item, bool):
            return None
        try:
            if isinstance(item, Integral):
                value = int(item)
            elif isinstance(item, Real) and float(item).is_integer():
                value = int(item)
            else:
                return None
        except (TypeError, ValueError, OverflowError):
            return None
        out.append(value)
    return out if all(0 <= a < ACTION_SPACE_SIZE for a in out) else None


def _as_vector(obj: Any) -> Sequence[float] | None:
    if isinstance(obj, (str, bytes, dict)):
        return None
    try:
        if len(obj) == STATE_DIM:
            return obj
    except TypeError:
        pass
    return None


def _unpack_state(
    state: Any,
) -> tuple[Sequence[float] | None, Any, list[int] | None]:
    direct = _as_vector(state)
    if direct is not None:
        return direct, None, None
    return (
        _as_vector(_field(state, _VECTOR_KEYS)),
        _field(state, _BOARD_KEYS),
        _as_int_list(_field(state, _ACTIONS_KEYS)),
    )


def _fallback(allowed: list[int] | None) -> int:
    if allowed:
        return int(_END_TURN if _END_TURN in allowed else allowed[0])
    return _END_TURN


class Agent:
    """One seat. The module-level function keeps one of these per player_id."""

    policy_id = "toprakthegoat-v1"

    def __init__(self, player_id: int = 0, **kwargs: Any) -> None:
        for key in ("player_id", "pid", "agent_id", "seat"):
            value = kwargs.pop(key, None)
            if value is not None:
                player_id = int(value)
                break
        self.player_id = int(player_id)

    def choose_action(self, *args: Any, **kwargs: Any) -> int:
        env = kwargs.get("env")
        allowed = _as_int_list(kwargs.get("allowed_actions"))
        state = kwargs.get("state")
        seat = kwargs.get("player_id")
        if seat is None:
            seat = kwargs.get("pid")

        contract = None
        if len(args) == 3 and env is None and allowed is None:
            first, middle, last = args
            ids = _as_int_list(last)
            seat_arg = _as_seat(middle)
            if ids is not None and seat_arg is not None:
                contract = (first, seat_arg, ids)

        if contract is not None:
            state, seat, allowed = contract
        else:
            for arg in args:
                if env is None:
                    candidate = getattr(arg, "env", arg)
                    if _looks_like_env(candidate):
                        env = candidate
                        continue
                ids = _as_int_list(arg)
                if ids is not None and allowed is None and len(ids) != STATE_DIM:
                    allowed = ids
                    continue
                if state is None and arg is not None and not isinstance(
                    arg, (int, float)
                ):
                    state = arg
            if seat is None:
                for arg in args:
                    found = _as_seat(arg)
                    if found is not None:
                        seat = found
                        break
        seat = self.player_id if seat is None else int(seat)

        try:
            return self._decide(env, state, seat, allowed)
        except Exception:
            return _fallback(allowed)

    def _decide(
        self,
        env: Any,
        state: Any,
        seat: int,
        allowed: list[int] | None,
    ) -> int:
        vector, board, state_actions = (None, None, None)
        if state is not None:
            vector, board, state_actions = _unpack_state(state)
        if allowed is None:
            allowed = state_actions

        if env is not None and _looks_like_env(env):
            try:
                seat = int(env.whose_turn())
            except Exception:
                pass
            if allowed is None:
                allowed = [int(a) for a in env.get_allowed_actions(seat)]
            return self._act(env, seat, allowed)

        if not allowed:
            return _END_TURN
        if len(allowed) == 1:
            return int(allowed[0])
        if vector is None or len(vector) != STATE_DIM:
            return _fallback(allowed)
        shadow = hydrate_env(vector, seat, allowed, board)
        return self._act(shadow, seat, allowed)

    def _act(self, env: MonopolyEnv, seat: int, allowed: list[int]) -> int:
        if not allowed:
            return _END_TURN
        if len(allowed) == 1:
            return int(allowed[0])
        action = int(heuristic_action(env, seat, allowed))
        return action if action in allowed else _fallback(allowed)


def make_agent(player_id: int = 0, **kwargs: Any) -> Agent:
    return Agent(player_id, **kwargs)


_DEFAULT_AGENTS: dict[int, Agent] = {}


def choose_action(*args: Any, **kwargs: Any) -> int:
    """``choose_action(state, player_id, allowed_actions) -> int``."""

    seat = None
    for key in ("player_id", "pid", "agent_id", "seat"):
        value = kwargs.get(key)
        if value is not None:
            seat = int(value)
            break
    if seat is None:
        for arg in args:
            found = _as_seat(arg)
            if found is not None:
                seat = found
                break
    seat = 0 if seat is None else int(seat)
    agent = _DEFAULT_AGENTS.get(seat)
    if agent is None:
        agent = Agent(seat)
        _DEFAULT_AGENTS[seat] = agent
    return agent.choose_action(*args, **kwargs)
