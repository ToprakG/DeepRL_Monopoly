"""
Submission entry point.

Wraps our hybrid PPO agent (see ALGORITHM.md for the 9 hand-designed
heuristics + network split). Takes env because most of the heuristics need
the board state, not just the 300-float vector.

Robust to calling convention: different spec drafts for this competition
disagree on whether ``choose_action`` receives ``(state, allowed_actions, env)``
or ``(state, player_id, allowed_actions)`` or keyword-only, and on whether
``env``/``player_id`` are passed at all. Own implementation, independent of
any other repo, but the risk itself was confirmed by seeing another team's
public commit history document the same two-spec ambiguity. Rather than
trust one fixed positional order, every call is resolved by inspecting the
*shape* of what was actually handed over: a 300-float vector is the state,
an object with ``get_allowed_actions``/``players`` is the env, a small
sequence of ints is the legal-action list, and a lone int is the seat.
"""

from pathlib import Path
from typing import Any, Sequence

from competitors.boom.engine.agent_ppo import PPOAgent
from competitors.boom.engine.state import STATE_DIM

_ROOT = Path(__file__).resolve().parent

_MODEL_PATH = _ROOT / "artifacts" / "submission" / "model.pt"


def _looks_like_env(obj: Any) -> bool:
    return hasattr(obj, "get_allowed_actions") and hasattr(obj, "players")


def _looks_like_state(obj: Any) -> bool:
    try:
        return len(obj) == STATE_DIM and not _looks_like_env(obj)
    except TypeError:
        return False


def _looks_like_actions(obj: Any) -> bool:
    if isinstance(obj, (str, bytes)) or _looks_like_env(obj):
        return False
    try:
        items = list(obj)
    except TypeError:
        return False
    return len(items) > 0 and all(isinstance(x, int) and not isinstance(x, bool) for x in items)


def _resolve(args: Sequence[Any], kwargs: dict) -> tuple:
    """Sort every positional/keyword value by shape, ignoring the order or
    names they arrived under. Returns (state, allowed_actions, env, player_id)."""
    state = kwargs.get("state")
    allowed = kwargs.get("allowed_actions")
    env = kwargs.get("env")
    player_id = None
    for key in ("player_id", "pid", "agent_id", "seat"):
        if kwargs.get(key) is not None:
            player_id = int(kwargs[key])
            break

    consumed_keys = ("state", "allowed_actions", "env", "player_id", "pid", "agent_id", "seat")
    candidates = list(args) + [v for k, v in kwargs.items() if k not in consumed_keys]
    for value in candidates:
        if value is None:
            continue
        if env is None and _looks_like_env(value):
            env = value
        elif state is None and _looks_like_state(value):
            state = value
        elif allowed is None and _looks_like_actions(value):
            allowed = value
        elif player_id is None and isinstance(value, int) and not isinstance(value, bool):
            player_id = value
    return state, allowed, env, player_id


class Agent:
    def __init__(self, player_id: int = 0, **kwargs: Any):
        for key in ("player_id", "pid", "agent_id", "seat"):
            if kwargs.get(key) is not None:
                player_id = int(kwargs[key])
                break
        self.player_id = int(player_id)
        self._agent = PPOAgent(player_id=self.player_id, hybrid=True)
        if _MODEL_PATH.exists():
            import torch

            ckpt = torch.load(
                str(_MODEL_PATH),
                map_location=self._agent.device,
                weights_only=True,
            )
            # Checkpoint was trained as seat 0; weights are seat-agnostic.
            self._agent.actor.load_state_dict(ckpt["actor"])
            self._agent.critic.load_state_dict(ckpt["critic"])
        if hasattr(self._agent, "epsilon"):
            self._agent.epsilon = 0.0

    def choose_action(self, *args: Any, **kwargs: Any) -> int:
        state, allowed, env, seat = _resolve(args, kwargs)
        if seat is None:
            seat = self.player_id
        if env is None:
            # No board available: nothing our heuristics can use, but the
            # contract still requires a legal action back.
            if allowed:
                return int(allowed[0])
            raise TypeError("choose_action needs the environment or allowed_actions")
        if allowed is None:
            allowed = list(env.get_allowed_actions(seat))
        if state is None:
            state = env._get_state(seat)
        if len(allowed) == 1:
            return int(allowed[0])

        action, _log_prob, _value, _nn_allowed = self._agent.choose_action(
            state, env, list(allowed)
        )
        return action if action in allowed else int(allowed[0])


_DEFAULT_AGENTS: dict = {}


def choose_action(*args: Any, **kwargs: Any) -> int:
    """Module-level fallback for a harness that calls a bare function
    instead of instantiating ``Agent``."""
    state, allowed, env, seat = _resolve(args, kwargs)
    seat = 0 if seat is None else seat
    agent = _DEFAULT_AGENTS.get(seat)
    if agent is None:
        agent = Agent(seat)
        _DEFAULT_AGENTS[seat] = agent
    return agent.choose_action(state=state, allowed_actions=allowed, env=env, player_id=seat)
