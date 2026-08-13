"""Oracle agent: MaxNPUCT + hybrid-prior + rollout leaf."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from monopoly_bench.config import SearchConfig
from monopoly_bench.contracts import SearchResult
from monopoly_bench.engine import ACTION_SPACE_SIZE, NUM_PLAYERS, STATE_DIM, action_family
from monopoly_bench.search import MaxNPUCT
from monopoly_game_engine.actions import OFFSETS, ActionType
from monopoly_game_engine.env import MonopolyEnv

from .rollout_leaf import (
    DEFAULT_HORIZON,
    DEFAULT_MARGIN_TEMPERATURE,
    DEFAULT_ROLLOUTS,
    rollout_leaf_value,
)
from .rollout_policy import greedy_rollout_action

ORACLE_V1 = "oracle-rollout-v1"

# Keep sims focused: progressive widening still opens families, but the prior
# peaks on the DealMaker+Builder action so visits are not near-uniform.
PRIOR_PEAK = 0.55
STRUCTURAL_BOOST = {
    "BUY_PROPERTY": 4.0,
    "ACCEPT_TRADE": 2.0,
    "DECLINE_TRADE": 3.0,
    "ROLL_DICE": 2.0,
    "END_TURN": 1.5,
    "improve_house": 4.0,
    "improve_hotel": 4.0,
    "buy_trade": 2.0,
    "exch_trade": 1.5,
    "auction": 1.5,
}
STRUCTURAL_PENALTY = {
    "mortgage": 0.25,
    "unmortgage": 0.4,
    "sell_house": 0.2,
    "sell_hotel": 0.2,
    "sell_prop": 0.15,
    "sell_trade": 0.3,
}


@dataclass(frozen=True, slots=True)
class OracleConfig:
    simulations: int = 200
    c_puct: float = 1.5
    max_depth: int = 64
    max_width: int = 16
    rollout_horizon: int = DEFAULT_HORIZON
    rollouts_per_leaf: int = DEFAULT_ROLLOUTS
    margin_temperature: float = DEFAULT_MARGIN_TEMPERATURE
    prior_peak: float = PRIOR_PEAK
    # Live play only. Labeling leaves these None so every checkpoint still spends
    # the full sim budget (same teacher as oracle-rollout-v1).
    deadline_s: float | None = None
    early_stop_visit_lead: int | None = None
    early_stop_min_sims: int = 16

    def search_config(self) -> SearchConfig:
        return SearchConfig(
            simulations=self.simulations,
            c_puct=self.c_puct,
            max_depth=self.max_depth,
            max_width=self.max_width,
        )


class UniformPriorModel:
    """Legal-uniform priors; value unused when ``leaf_fn`` is set."""

    def predict(
        self,
        state: np.ndarray,
        legal_actions,
        actor_id: int,
        env: MonopolyEnv | None = None,
    ) -> tuple[dict[int, float], np.ndarray]:
        del state, actor_id, env
        legal = tuple(int(action) for action in legal_actions)
        if not legal:
            raise ValueError("Cannot predict without a legal action")
        mass = 1.0 / len(legal)
        priors = {action: mass for action in legal}
        value = np.full(NUM_PLAYERS, 1.0 / NUM_PLAYERS, dtype=np.float32)
        return priors, value


class HybridPriorModel:
    """Peak prior on DealMaker+Builder action; soft family weights elsewhere."""

    def __init__(self, peak: float = PRIOR_PEAK):
        if not 0.0 < peak < 1.0:
            raise ValueError("prior peak must be in (0, 1)")
        self.peak = float(peak)

    def predict(
        self,
        state: np.ndarray,
        legal_actions,
        actor_id: int,
        env: MonopolyEnv | None = None,
    ) -> tuple[dict[int, float], np.ndarray]:
        del state
        legal = tuple(int(action) for action in legal_actions)
        if not legal:
            raise ValueError("Cannot predict without a legal action")

        weights = {action: self._family_weight(action) for action in legal}
        if env is not None:
            try:
                preferred = int(greedy_rollout_action(env, actor_id))
            except Exception:
                preferred = None
            if preferred in weights:
                # Dirac-ish peak so PUCT concentrates sims.
                others = [action for action in legal if action != preferred]
                if not others:
                    priors = {preferred: 1.0}
                else:
                    remainder = 1.0 - self.peak
                    other_total = sum(weights[action] for action in others)
                    priors = {preferred: self.peak}
                    if other_total <= 0:
                        share = remainder / len(others)
                        for action in others:
                            priors[action] = share
                    else:
                        for action in others:
                            priors[action] = remainder * (weights[action] / other_total)
                value = np.full(NUM_PLAYERS, 1.0 / NUM_PLAYERS, dtype=np.float32)
                return priors, value

        total = sum(weights.values())
        priors = {action: weights[action] / total for action in legal}
        value = np.full(NUM_PLAYERS, 1.0 / NUM_PLAYERS, dtype=np.float32)
        return priors, value

    @staticmethod
    def _family_weight(action: int) -> float:
        if action < OFFSETS["mortgage"]:
            try:
                name = ActionType(action).name
            except ValueError:
                name = "binary"
            return STRUCTURAL_BOOST.get(name, 1.0)
        family = action_family(action)
        if family in STRUCTURAL_BOOST:
            return STRUCTURAL_BOOST[family]
        if family in STRUCTURAL_PENALTY:
            return STRUCTURAL_PENALTY[family]
        return 1.0


def build_oracle_search(config: OracleConfig | None = None) -> MaxNPUCT:
    cfg = config or OracleConfig()
    leaf = partial(
        rollout_leaf_value,
        num_rollouts=cfg.rollouts_per_leaf,
        horizon=cfg.rollout_horizon,
        temperature=cfg.margin_temperature,
    )
    return MaxNPUCT(
        HybridPriorModel(peak=cfg.prior_peak),
        cfg.search_config(),
        self_play=False,
        leaf_fn=leaf,
        deadline_s=cfg.deadline_s,
        early_stop_visit_lead=cfg.early_stop_visit_lead,
        early_stop_min_sims=cfg.early_stop_min_sims,
    )


class OracleAgent:
    """``choose_action(env)`` wrapper compatible with ASU evaluate factories."""

    policy_id = ORACLE_V1

    def __init__(
        self,
        player_id: int,
        config: OracleConfig | None = None,
        *,
        search: MaxNPUCT | None = None,
        seed: int = 0,
    ):
        if not 0 <= player_id < NUM_PLAYERS:
            raise ValueError(f"player_id must be in [0, {NUM_PLAYERS - 1}]")
        self.player_id = player_id
        self.config = config or OracleConfig()
        self.search = search or build_oracle_search(self.config)
        self._seed = int(seed)
        self._decisions = 0
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_SPACE_SIZE

    def _decision_seed(self) -> int:
        seed = self._seed + self._decisions * 1_000_003 + self.player_id * 17
        self._decisions += 1
        return seed

    def search_action(self, env: MonopolyEnv) -> SearchResult:
        return self.search.choose_action(env, self.player_id, self._decision_seed())

    def choose_action(self, env: MonopolyEnv) -> int:
        legal = env.get_allowed_actions(self.player_id)
        if len(legal) == 1:
            self._decisions += 1
            return int(legal[0])
        result = self.search_action(env)
        action = int(result.chosen_action)
        if action not in legal:
            raise RuntimeError(f"{ORACLE_V1} selected illegal action {action}")
        return action


def oracle_config_from_args(args: Any) -> OracleConfig:
    deadline = getattr(args, "deadline_s", None)
    lead = getattr(args, "early_stop_lead", None)
    return OracleConfig(
        simulations=int(args.sims),
        rollout_horizon=int(args.horizon),
        rollouts_per_leaf=int(args.rollouts),
        margin_temperature=float(args.margin_temperature),
        deadline_s=None if deadline in (None, 0) else float(deadline),
        early_stop_visit_lead=None if not lead else int(lead),
        early_stop_min_sims=int(getattr(args, "early_stop_min_sims", 16)),
    )


__all__ = [
    "ORACLE_V1",
    "OracleAgent",
    "OracleConfig",
    "HybridPriorModel",
    "UniformPriorModel",
    "build_oracle_search",
    "oracle_config_from_args",
]
