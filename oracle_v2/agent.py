"""Faster oracle-rollout teacher: same Max-N + hybrid prior + rollout leaf.

Speed comes from transposition / leaf cache, not fewer sims or a weaker net.
Default budget matches the calibrated hybrid teacher (128 / 16 / 1).

Live 4s/turn mode (``live=True``): Max-N only on buy/build/incoming-trade/auction-open,
DealBuilder greedy elsewhere, remaining wall shared across the seat's turn.
"""

from __future__ import annotations

import time
from typing import Any

from monopoly_bench.contracts import SearchResult
from monopoly_bench.engine import ACTION_SPACE_SIZE, NUM_PLAYERS, STATE_DIM
from monopoly_game_engine.env import PHASE_AUCTION, PHASE_OUT_OF_TURN, MonopolyEnv

from oracle.agent import HybridPriorModel, OracleConfig
from oracle.hybrid_config import checkpoint_kind, is_event_checkpoint
from oracle.leaves import build_leaf_fn
from oracle.rollout_policy import greedy_rollout_action

from .search import CachedMaxNPUCT

ORACLE_V2 = "oracle-fast-v1"
LIVE_TURN_DEADLINE_S = 4.0


def default_v2_config(**overrides: Any) -> OracleConfig:
    payload = dict(
        simulations=128,
        c_puct=1.5,
        max_depth=64,
        max_width=16,
        rollout_horizon=16,
        rollouts_per_leaf=1,
        deadline_s=None,
        early_stop_visit_lead=None,
        early_stop_min_sims=16,
    )
    payload.update(overrides)
    return OracleConfig(**payload)


def build_oracle_v2_search(config: OracleConfig | None = None) -> CachedMaxNPUCT:
    cfg = config or default_v2_config()
    leaf = build_leaf_fn(cfg)
    return CachedMaxNPUCT(
        HybridPriorModel(peak=cfg.prior_peak),
        cfg.search_config(),
        self_play=False,
        leaf_fn=leaf,
        deadline_s=cfg.deadline_s,
        early_stop_visit_lead=cfg.early_stop_visit_lead,
        early_stop_min_sims=cfg.early_stop_min_sims,
    )


def _turn_id(env: MonopolyEnv) -> tuple:
    """One 4s budget for a seat's pre+post roll, one for an auction, one for OOT."""

    if env.phase == PHASE_AUCTION:
        return ("auction", int(env.round), env.auction_property_id)
    if env.phase == PHASE_OUT_OF_TURN:
        return ("oot", int(env.round), int(env.whose_turn()))
    return ("seat", int(env.round), int(env.active_player_id()))


class OracleV2Agent:
    """Drop-in ``choose_action(env)`` teacher with cached Max-N.

    ``live=True`` caps each of this seat's turns at ``turn_deadline_s`` (default 4s)
    and searches only event checkpoints so the cap is actually hittable.
    """

    policy_id = ORACLE_V2

    def __init__(
        self,
        player_id: int,
        config: OracleConfig | None = None,
        *,
        search: CachedMaxNPUCT | None = None,
        seed: int = 0,
        live: bool = False,
        turn_deadline_s: float | None = None,
        event_search_only: bool | None = None,
    ):
        if not 0 <= player_id < NUM_PLAYERS:
            raise ValueError(f"player_id must be in [0, {NUM_PLAYERS - 1}]")
        self.player_id = player_id
        self.config = config or default_v2_config()
        self.search = search or build_oracle_v2_search(self.config)
        self._seed = int(seed)
        self._decisions = 0
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_SPACE_SIZE
        self.live = bool(live)
        if turn_deadline_s is None:
            self.turn_deadline_s = LIVE_TURN_DEADLINE_S if self.live else None
        else:
            self.turn_deadline_s = None if turn_deadline_s <= 0 else float(turn_deadline_s)
        if event_search_only is None:
            self.event_search_only = self.live
        else:
            self.event_search_only = bool(event_search_only)
        self._turn_key: tuple | None = None
        self._turn_deadline_at: float | None = None
        self._build_menu_labeled: set[tuple[int, int]] = set()
        self._trade_round_labeled: set[tuple[int, int]] = set()
        self._last_round: int | None = None
        self.last_used_search = False
        self.last_kind: str | None = None

    def _decision_seed(self) -> int:
        seed = self._seed + self._decisions * 1_000_003 + self.player_id * 17
        self._decisions += 1
        return seed

    def _refresh_turn_clock(self, env: MonopolyEnv) -> float | None:
        if self.turn_deadline_s is None:
            return None
        if env.round != self._last_round:
            self._trade_round_labeled.clear()
            self._build_menu_labeled.clear()
            self._last_round = env.round
        key = _turn_id(env)
        now = time.perf_counter()
        if key != self._turn_key:
            self._turn_key = key
            self._turn_deadline_at = now + self.turn_deadline_s
        assert self._turn_deadline_at is not None
        return self._turn_deadline_at - now

    def _is_event(self, env: MonopolyEnv, legal: tuple[int, ...]) -> tuple[bool, str | None]:
        kind = checkpoint_kind(env, legal)
        build_key = (env.round, self.player_id)
        trade_key = (env.round, self.player_id)
        return (
            is_event_checkpoint(
                env,
                legal,
                already_labeled_build_menu=build_key in self._build_menu_labeled,
                already_labeled_trade_round=trade_key in self._trade_round_labeled,
            ),
            kind,
        )

    def search_action(self, env: MonopolyEnv) -> SearchResult:
        return self.search.choose_action(env, self.player_id, self._decision_seed())

    def choose_action(self, env: MonopolyEnv) -> int:
        legal = tuple(env.get_allowed_actions(self.player_id))
        if len(legal) == 1:
            self._decisions += 1
            self.last_used_search = False
            self.last_kind = "forced"
            return int(legal[0])

        remaining = self._refresh_turn_clock(env)
        is_event, kind = self._is_event(env, legal)
        search = True
        if self.event_search_only and not is_event:
            search = False
        if remaining is not None and remaining <= 0:
            search = False

        if not search:
            self._decisions += 1
            self.last_used_search = False
            self.last_kind = kind or "greedy"
            action = int(greedy_rollout_action(env, self.player_id))
            if action not in legal:
                action = int(legal[0])
            return action

        previous_deadline = self.search.deadline_s
        if remaining is not None:
            self.search.deadline_s = max(float(remaining), 1e-3)
        try:
            result = self.search_action(env)
        finally:
            self.search.deadline_s = previous_deadline
        action = int(result.chosen_action)
        if action not in legal:
            raise RuntimeError(f"{ORACLE_V2} selected illegal action {action}")
        self.last_used_search = True
        self.last_kind = kind or "search"
        if kind == "build":
            self._build_menu_labeled.add((env.round, self.player_id))
        elif kind == "trade":
            self._trade_round_labeled.add((env.round, self.player_id))
        return action


__all__ = [
    "LIVE_TURN_DEADLINE_S",
    "ORACLE_V2",
    "OracleV2Agent",
    "build_oracle_v2_search",
    "default_v2_config",
]
