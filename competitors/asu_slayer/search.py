"""``slayer-rollout-v1``: truncated rollout search over the net-worth policy.

The greedy policy is myopic in exactly one place that matters: it cannot see
that a purchase which is accretive today leaves it unable to pay rent three
turns from now. This module keeps the greedy policy as the rollout driver and
only pays for search where the shortlist is genuinely contested.
"""

from __future__ import annotations

import copy
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from monopoly_game_engine.actions import ActionType, AuctionAction
from monopoly_game_engine.constants import NUM_PLAYERS
from monopoly_game_engine.env import (
    PHASE_AUCTION,
    PHASE_OUT_OF_TURN,
    PHASE_POST_ROLL,
    PHASE_PRE_ROLL,
)

from .policy import DEFAULT_CONFIG, SlayerConfig, SlayerV1
from .scoring import equity


SLAYER_ROLLOUT_V1 = "slayer-rollout-v1"


@dataclass(frozen=True, slots=True)
class SearchConfig:
    shortlist: int = 4
    rollouts: int = 6
    depth: int = 44
    seed: int = 0
    rent_horizon: float = 3.0
    survival_bonus: float = 4000.0
    contest_margin: float = 0.02


DEFAULT_SEARCH = SearchConfig()


@contextmanager
def preserve_global_rng() -> Iterator[None]:
    """Restore Python, NumPy, and imported Torch RNG state on exit."""

    python_state = random.getstate()
    numpy = sys.modules.get("numpy")
    numpy_state = numpy.random.get_state() if numpy is not None else None
    torch = sys.modules.get("torch")
    torch_state = torch.get_rng_state() if torch is not None else None
    try:
        yield
    finally:
        random.setstate(python_state)
        if numpy is not None and numpy_state is not None:
            numpy.random.set_state(numpy_state)
        if torch is not None and torch_state is not None:
            torch.set_rng_state(torch_state)


class _PrivateGame:
    """A cloned environment driven by its own stdlib random stream."""

    __slots__ = ("env", "state")

    def __init__(self, env, seed: int):
        self.env = copy.deepcopy(env)
        self.state = random.Random(seed).getstate()

    def step(self, action: int):
        outer = random.getstate()
        try:
            random.setstate(self.state)
            result = self.env.step(action)
            self.state = random.getstate()
            return result
        finally:
            random.setstate(outer)


class SlayerRolloutV1:
    """Greedy net-worth policy refined by common-random-number rollouts."""

    policy_id = SLAYER_ROLLOUT_V1

    def __init__(
        self,
        player_id: int,
        config: SlayerConfig = DEFAULT_CONFIG,
        search: SearchConfig = DEFAULT_SEARCH,
    ):
        self.player_id = player_id
        self.config = config
        self.search = search
        self.base = SlayerV1(player_id, config)
        self.searched_decisions = 0

    # ── Candidate shortlist ───────────────────────────────────────────────

    def _candidates(self, env, legal: set[int]) -> list[int]:
        """Plausible actions worth spending rollouts on, best-first."""

        greedy = self.base.choose_action(env)
        candidates = [greedy]

        def add(action: int) -> None:
            if action in legal and action not in candidates:
                candidates.append(action)

        if env.phase == PHASE_AUCTION:
            add(int(AuctionAction.PASS))
        elif env.phase == PHASE_POST_ROLL and env.has_rolled:
            add(int(ActionType.BUY_PROPERTY))
            add(int(ActionType.END_TURN))
        elif env.phase in (PHASE_PRE_ROLL, PHASE_OUT_OF_TURN):
            if int(ActionType.ACCEPT_TRADE) in legal:
                add(int(ActionType.ACCEPT_TRADE))
                add(int(ActionType.DECLINE_TRADE))
            for _gain, action in self.base._investments(env, legal):
                add(action)
            add(int(ActionType.END_TURN))
        return candidates[: self.search.shortlist]

    # ── Rollout ───────────────────────────────────────────────────────────

    def _rollout(self, env, action: int, seed: int) -> float:
        game = _PrivateGame(env, seed)
        game.step(action)
        drivers = [SlayerV1(seat, self.config) for seat in range(NUM_PLAYERS)]
        for _ in range(self.search.depth):
            if game.env.done:
                break
            actor = game.env.whose_turn()
            game.step(drivers[actor].choose_action(game.env))
        return equity(
            game.env,
            self.player_id,
            self.search.rent_horizon,
            self.search.survival_bonus,
        )

    def choose_action(self, env) -> int:
        legal_list = env.get_allowed_actions(self.player_id)
        if len(legal_list) == 1:
            return legal_list[0]
        legal = set(legal_list)

        with preserve_global_rng():
            candidates = self._candidates(env, legal)
            if len(candidates) < 2:
                return candidates[0]

            self.searched_decisions += 1
            # Salt the rollout seeds by the decision context: identical seeds
            # for every candidate at THIS decision (common random numbers),
            # but different dice from one decision to the next. A fixed
            # 0..rollouts-1 seed set replayed the same six dice sequences for
            # the whole game and correlated every decision's error
            # (SLAYER_REVIEW.md 4.4).
            salt = (
                31 * int(env.round)
                + 7 * int(env.players[self.player_id].position)
                + self.searched_decisions
            ) * 1009
            seeds = [
                self.search.seed + salt + index
                for index in range(self.search.rollouts)
            ]
            best_action = candidates[0]
            best_score = None
            for candidate in candidates:
                # Common random numbers: every candidate meets the same dice.
                score = sum(self._rollout(env, candidate, seed) for seed in seeds)
                score /= len(seeds)
                if best_score is None or score > best_score:
                    best_action, best_score = candidate, score
            return best_action


__all__ = [
    "DEFAULT_SEARCH",
    "SLAYER_ROLLOUT_V1",
    "SearchConfig",
    "SlayerRolloutV1",
    "preserve_global_rng",
]
