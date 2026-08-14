"""Policy registry for the competition agent benchmark harness.

Every policy exposes ``choose_action(env) -> int``. Construction is
``factory(player_id, rng_seed)`` so that stochastic policies are explicitly
seeded and therefore reproducible (non-negotiable constraint: determinism).

The ASU teachers are used strictly as opaque ``decide(env) -> action``
functions. Nothing in this package reads ASU_FROZEN_TEACHER/core.py or
evaluate.py; only their public callables are invoked.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Callable, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ASU_FROZEN_TEACHER import ASURolloutV1, ASUValueV1  # noqa: E402
from ASU_FROZEN_TEACHER.evaluate import _ScriptedAdapter  # noqa: E402
from monopoly_game_engine.agents_fixed import FP_AGENT_CLASSES  # noqa: E402

# FP_AGENT_CLASSES order is fixed by the engine:
# TheHoarder, TheDealMaker, TheGambler, TheBuilder, TheBlocker, TheRailBaron
FIXED_NAMES = ("fixed-a", "fixed-b", "fixed-c", "fixed-d", "fixed-e", "fixed-f")


class RandomPolicy:
    """Uniform random legal action. Seeded per seat for reproducibility."""

    policy_id = "random"

    def __init__(self, player_id: int, rng_seed: int) -> None:
        self.player_id = player_id
        self._rng = random.Random(rng_seed)

    def choose_action(self, env) -> int:
        legal = env.get_allowed_actions(self.player_id)
        return self._rng.choice(list(legal))


def _fixed_factory(index: int) -> Callable[[int, int], object]:
    def build(player_id: int, rng_seed: int):
        # _ScriptedAdapter applies the same END_TURN-otherwise-first-legal
        # compatibility fallback the repo's own evaluator uses; the scripted
        # agents occasionally emit END_TURN where only liquidation is legal.
        return _ScriptedAdapter(FP_AGENT_CLASSES[index](player_id), player_id)

    return build


def _spec_factory(player_id: int, rng_seed: int):
    from competition_agent.spec_policy import SpecPolicy
    return SpecPolicy(player_id, rng_seed)


def _rollout_spec_factory(player_id: int, rng_seed: int):
    from competition_agent.rollout_policy import RolloutPolicy
    return RolloutPolicy(player_id, rng_seed)


def _final_factory(player_id: int, rng_seed: int):
    from competition_agent.final_agent import FinalAgent
    return FinalAgent(player_id, rng_seed)


def _hybrid_factory(player_id: int, rng_seed: int):
    from competition_agent.hybrid_policy import HybridPolicy
    return HybridPolicy(player_id, rng_seed)


REGISTRY: Dict[str, Callable[[int, int], object]] = {
    "spec": _spec_factory,
    "hybrid": _hybrid_factory,
    "final": _final_factory,
    "rollout_spec": _rollout_spec_factory,
    # The frozen teacher, both variants. "teacher" aliases the value variant
    # because it is the Phase 2 agreement target.
    "teacher": lambda pid, seed: ASUValueV1(pid),
    "value": lambda pid, seed: ASUValueV1(pid),
    "rollout": lambda pid, seed: ASURolloutV1(pid),
    "random": RandomPolicy,
}
for _i, _name in enumerate(FIXED_NAMES):
    REGISTRY[_name] = _fixed_factory(_i)


def build_policy(name: str, player_id: int, rng_seed: int):
    """Instantiate a registered policy for a seat."""
    try:
        factory = REGISTRY[name]
    except KeyError:
        raise SystemExit(
            f"unknown policy {name!r}; choose from {sorted(REGISTRY)}"
        ) from None
    return factory(player_id, rng_seed)
