"""UNDERDOG — competition entrypoint.

    choose_action(state, allowed_actions, env, player_id) -> int

``env`` and ``player_id`` are the optional extras the harness passes by keyword
when they are declared after the two required parameters. This agent declares
them because it needs the board itself: its features read deeds, colour groups,
houses, cash, the house bank and pending trades, and the 300-float state vector
cannot be turned back into an environment.

Both the function form and the class form are exported, so either calling
convention works. A module-level ``Agent`` keeps one policy object per seat.

The policy
----------
Two gradient-boosted models over engineered features, in ``underdog_gbm/``.
Model A owns the auction family, Model B every other family; each scores the
legal candidates and the agent takes the argmax. See
``underdog_gbm/gbm_policy.py`` for the split and ``docs/GAUNTLET.md`` for what
it measured against the other teams' agents.

Contract compliance
-------------------
* **Only legal actions.** Every return value is checked against
  ``allowed_actions`` and replaced if it is not a member. The rest of this
  repository prefers to fail closed on an illegal action, which is right for
  development because it surfaces bugs — but in a scored match an exception and
  an illegal action both lose the game, so here it substitutes instead.
* **The global RNG is never touched.** The policy is a deterministic argmax and
  draws from no random source, global or private. Nothing here calls
  ``random.*``, ``np.random.*`` or ``torch.rand*``.
* **Latency.** LightGBM inference over a few hundred candidate rows: single-core
  milliseconds, against the per-decision limit.
* **Dependencies.** ``numpy`` and ``lightgbm``, both wheel-only, both in
  ``requirements.txt``. Nothing is downloaded at match time.

The fallback, and why it is here
--------------------------------
LightGBM is a compiled extension and the boosters are files on disk. If either
fails to load — a wheel that will not import in the sandbox, a missing OpenMP
runtime, a truncated checkout — this falls back to ``ChampionScore``, the
hand-written rule agent in ``underdog/``, which needs no dependency beyond the
engine. That is a real downgrade and it is announced on ``stderr`` once rather
than absorbed silently, because a submission that quietly plays a weaker policy
than the one that was measured is worse than one that says so. It is preferred
to the alternative: without it, an import failure forfeits every game.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parent

# Only the policy package goes on the path, and only this one.
#
# ``underdog/`` is deliberately NOT added here even though the fallback lives
# in it: it contains ``underdog/engine/``, a complete vendored copy of the
# simulator under the top-level name ``engine``. Putting that on sys.path[0]
# inside the harness's process would let it answer a later ``import engine``
# and silently decide the rules for the whole table -- the exact failure this
# project pinned its own engine to avoid (docs/GAUNTLET.md). ``_fallback()``
# adds it only if it is ever needed, and only after ``engine`` is already
# bound to the harness's simulator.
_PKG = str(_ROOT / "underdog_gbm")
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

__all__ = ["Agent", "choose_action", "make_agent", "VARIANT"]

VARIANT = "GBM"

# Resolved lazily so importing this module cannot fail on a missing package.
_POLICY: Any = None
_FALLBACK = False


def _policy() -> Any:
    """Return the policy, loading it on first use.

    The boosters are ~4.3 MB of text and LightGBM is a compiled extension, so
    neither is touched at import time. A failure here is reported once and then
    served by the rule agent for the rest of the process.
    """
    global _POLICY, _FALLBACK
    if _POLICY is not None:
        return _POLICY
    try:
        # Imported before LightGBM on purpose. This module runs ``engine_shim``,
        # which binds the top-level name ``engine`` to whichever
        # ``monopoly_game_engine`` the harness already has. Doing it first means
        # that even when LightGBM is what fails, ``engine`` is correctly bound
        # before the fallback -- whose own imports would otherwise resolve
        # against ``underdog/engine/`` and run on the wrong copy of the rules.
        from underdog_gbm.gbm_policy import MonopolyAgent
        import lightgbm as lgb
        agent = MonopolyAgent()
        # Load both boosters now. They are lazy by default, and a truncated or
        # unreadable model file should surface here rather than mid-game on the
        # first decision of its family -- Model A's family is auctions, which a
        # game can reach hundreds of decisions in.
        for booster in (agent.model_a, agent.model_b):
            booster._b = lgb.Booster(model_file=booster.path)
        _POLICY = agent
    except Exception as exc:
        print(f"[UNDERDOG] gradient-boosted policy unavailable ({type(exc).__name__}:"
              f" {exc}); falling back to the rule agent", file=sys.stderr, flush=True)
        _POLICY = _fallback()
        _FALLBACK = True
    return _POLICY


def _fallback() -> Any:
    """The rule agent, imported without letting its engine copy win.

    ``underdog/heuristic`` reads ``from engine.actions import ...``. The name
    ``engine`` must already point at the harness's simulator before that import
    runs, or Python will bind it to ``underdog/engine/`` -- a vendored copy --
    and the fallback would then play by rules the rest of the table is not
    using. So bind it explicitly first, and refuse to import at all if there is
    no simulator to bind it to.
    """
    import importlib
    if "engine" not in sys.modules:
        engine = importlib.import_module("monopoly_game_engine")
        sys.modules["engine"] = engine
        for sub in ("actions", "constants", "env", "state"):
            sys.modules.setdefault(
                f"engine.{sub}", importlib.import_module(f"monopoly_game_engine.{sub}"))
    path = str(_ROOT / "underdog")
    if path not in sys.path:
        sys.path.append(path)      # appended, never ahead of the harness
    from heuristic import ChampionScore
    return ChampionScore()


def _as_actions(value: Any) -> list[int] | None:
    """Coerce ``value`` to a list of action ids, or ``None`` if it is not one.

    Deliberately tolerant. The action list arrives as a ``list[int]`` in one
    spec version and could plausibly arrive as a tuple, an integer array or a
    sequence of descriptor objects carrying an ``action`` field; none of those
    is worth failing a match over.
    """
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return None
    try:
        items = list(value)
    except TypeError:
        return None
    if not items:
        return []
    out: list[int] = []
    for item in items:
        for attr in ("action", "action_id", "id", "index"):
            if hasattr(item, attr):
                item = getattr(item, attr)
                break
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            return None
    return out


def _looks_like_env(obj: Any) -> bool:
    """The board surface the policy actually reads."""
    return obj is not None and hasattr(obj, "properties") and hasattr(obj, "players")


class _EnvView:
    """A board snapshot wearing the one method the policy also needs.

    The policy asks the board for ``get_allowed_actions(pid)``. A read-only
    snapshot may expose the deeds and players without that method, in which
    case the legal list we were handed separately is the same information.
    Everything else passes straight through.
    """

    __slots__ = ("_board", "_allowed")

    def __init__(self, board: Any, allowed: Sequence[int]) -> None:
        object.__setattr__(self, "_board", board)
        object.__setattr__(self, "_allowed", [int(a) for a in allowed])

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_board"), name)

    def get_allowed_actions(self, player_id: int | None = None) -> list[int]:
        return list(object.__getattribute__(self, "_allowed"))


def _unpack(args: tuple, kwargs: dict) -> tuple[Any, list[int], Any, int | None]:
    """Sort whatever we were called with into (state, allowed, env, seat).

    Two published spec versions order the parameters differently --
    ``(state, allowed_actions)`` with ``env``/``player_id`` as declared extras,
    and ``(state, player_id, allowed_actions)`` -- so position alone cannot say
    what a value is. Types can: the legal actions are a sequence of ints, a
    seat is a bare int, and the board answers to ``properties`` and
    ``players``. Sorting by shape rather than by position means a reordered
    call is handled instead of being a crash on every decision.
    """
    pool: list = []

    def named(names, fits, convert=lambda v: v):
        """Take a keyword value only if its *shape* matches its name.

        A name is good evidence but not proof: a caller that passed the four
        values in the other order still labels them, and believing a label
        over the value is how the whole decision loop ends up crashing. A
        value that does not fit its name is not discarded -- it goes into the
        pool and is sorted by shape with everything else.
        """
        for name in names:
            value = kwargs.get(name)
            if value is None:
                continue
            if fits(value):
                return convert(value)
            pool.append(value)
        return None

    env = named(("env", "game", "environment"), _looks_like_env)
    allowed = named(("allowed_actions", "allowed", "legal_actions", "actions"),
                    lambda v: _as_actions(v) is not None, _as_actions)
    seat = named(("player_id", "pid", "seat", "agent_id"),
                 lambda v: isinstance(v, int) and not isinstance(v, bool))
    state = kwargs.get("state", kwargs.get("observation"))
    pool.extend(args)

    for value in pool:
        if value is None:
            continue
        if env is None and _looks_like_env(value):
            env = value
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            # A bare int is the seat; a second one is the decision seed, which
            # this policy does not use (it is deterministic).
            if seat is None:
                seat = value
            continue
        actions = _as_actions(value)
        # The 300-float observation is also a sequence, so length and content
        # separate it from the legal list: action ids are small and few.
        if actions is not None and allowed is None and len(actions) != 300:
            allowed = actions
            continue
        if state is None:
            state = value

    # A read-only decision state carries the pieces as attributes.
    if state is not None and not _looks_like_env(env):
        for attr in ("env", "game", "board"):
            candidate = getattr(state, attr, None)
            if _looks_like_env(candidate):
                env = candidate
                break
        if allowed is None:
            allowed = _as_actions(getattr(state, "actions", None))
        if seat is None:
            seat = getattr(state, "player_id", None)

    try:
        seat = int(seat) if seat is not None else None
    except (TypeError, ValueError):
        seat = None
    return state, (allowed or []), env, seat


_WARNED_NO_BOARD = False


def _warn_no_board() -> None:
    """Say once that we are playing blind, then stop.

    This is the loudest signal available that the harness is not handing over
    a board: every feature this agent has is computed from one, so without it
    the policy never runs at all. Printed once rather than per decision --
    a few thousand identical lines would bury it.
    """
    global _WARNED_NO_BOARD
    if not _WARNED_NO_BOARD:
        _WARNED_NO_BOARD = True
        print("[UNDERDOG] no board in the decision state; the policy cannot "
              "run and legal actions are being played instead",
              file=sys.stderr, flush=True)


def _legal(action: Any, allowed: Sequence[int]) -> int:
    """Return ``action`` if it is legal, else a legal substitute.

    Never raises and never returns a non-member of ``allowed``: an illegal
    return fails the match outright, so there is no value in propagating.
    """
    allowed = _as_actions(allowed) or []
    if not allowed:
        return 0
    try:
        action = int(action)
    except (TypeError, ValueError):
        return allowed[0]
    if action in allowed:
        return action
    try:
        from monopoly_game_engine.actions import ActionType
        end = int(ActionType.END_TURN)
        if end in allowed:
            return end
    except Exception:
        pass
    return allowed[0]


class Agent:
    """Class form. One instance per seat; holds no state between games."""

    name = "UNDERDOG"

    def __init__(self, player_id: int = 0, **kwargs: Any) -> None:
        for key in ("player_id", "pid", "agent_id", "seat"):
            if kwargs.get(key) is not None:
                player_id = int(kwargs[key])
                break
        self.player_id = int(player_id)

    def choose_action(self, state=None, allowed_actions=None, env=None,
                      player_id=None, *args: Any, **kwargs: Any) -> int:
        """Accepts either published parameter order; never raises.

        The parameters are *declared* in the older order because that is what a
        harness reads to decide what to hand over: one published contract
        injects ``env`` and ``player_id`` by keyword only when they appear
        after the first two positional parameters. Declaring them elsewhere
        means never being given a board at all.

        The declaration is not trusted at runtime. Every value goes through
        ``_unpack``, which sorts by shape, so the newer
        ``(state, player_id, allowed_actions)`` order arrives correctly even
        though it binds to the wrong names on the way in.
        """
        state, allowed, env, seat = _unpack(
            (state, allowed_actions, env, player_id) + args, kwargs)
        if seat is None:
            seat = self.player_id
        if not _looks_like_env(env):
            # No board. Nothing the policy reads is recoverable from the
            # 300-float vector, so take a legal action rather than guess --
            # a weak move scores; a crash is a strike.
            _warn_no_board()
            return _legal(None, allowed)
        # The engine only asks an agent to act on its own turn, so trust its
        # view of the acting seat over the one passed at construction.
        try:
            seat = int(env.whose_turn())
        except Exception:
            pass
        if not allowed:
            try:
                allowed = _as_actions(env.get_allowed_actions(seat)) or []
            except Exception:
                allowed = []
        if not hasattr(env, "get_allowed_actions"):
            env = _EnvView(env, allowed)
        try:
            action = _policy().choose_action(env, seat, 0)
        except Exception:
            action = None
        return _legal(action, allowed)

    def __repr__(self) -> str:
        kind = "rules (fallback)" if _FALLBACK else VARIANT
        return f"Agent(seat={self.player_id}, policy={kind})"


def make_agent(player_id: int = 0, **kwargs: Any) -> Agent:
    return Agent(player_id, **kwargs)


_SEATS: dict[int, Agent] = {}


def choose_action(state=None, allowed_actions=None, env=None, player_id=None,
                  *args: Any, **kwargs: Any) -> int:
    """The required contract.

    Declared in the older order for the reason given on ``Agent.choose_action``
    -- a harness reads these names to decide whether to hand over a board --
    and order-insensitive at runtime because ``_unpack`` sorts by shape.
    """
    state, allowed, env, seat = _unpack(
        (state, allowed_actions, env, player_id) + args, kwargs)
    seat = 0 if seat is None else seat
    agent = _SEATS.get(seat)
    if agent is None:
        agent = Agent(seat)
        _SEATS[seat] = agent
    # Already resolved: pass by keyword so they are not sorted a second time.
    return agent.choose_action(state=state, allowed_actions=allowed,
                               env=env, player_id=seat)
