"""Same Max-N PUCT as v1, with transposition + leaf cache (no sim cuts)."""

from __future__ import annotations

import time

import numpy as np

from monopoly_bench.config import SearchConfig
from monopoly_bench.contracts import SearchResult
from monopoly_bench.engine import NUM_PLAYERS, SharedGame
from monopoly_bench.search import DecisionNode, LeafFn, MaxNPUCT, PriorValueModel
from monopoly_game_engine.env import MonopolyEnv

from .clone import fast_clone_env
from .position import position_key


def _vector(value: np.ndarray) -> tuple[float, ...]:
    return tuple(float(item) for item in value)


class CachedMaxNPUCT(MaxNPUCT):
    """v1 search + per-decision TT. Unique positions still get a full leaf + priors."""

    def __init__(
        self,
        model: PriorValueModel,
        config: SearchConfig | None = None,
        *,
        self_play: bool = False,
        leaf_fn: LeafFn | None = None,
        deadline_s: float | None = None,
        early_stop_visit_lead: int | None = None,
        early_stop_min_sims: int = 16,
    ):
        super().__init__(
            model,
            config,
            self_play=self_play,
            leaf_fn=leaf_fn,
            deadline_s=deadline_s,
            early_stop_visit_lead=early_stop_visit_lead,
            early_stop_min_sims=early_stop_min_sims,
        )
        self._tt: dict[tuple, DecisionNode] = {}
        self._leaf_cache: dict[tuple, np.ndarray] = {}
        self._in_path: set[int] = set()
        self.tt_hits = 0
        self.leaf_evals = 0

    def _reset_caches(self) -> None:
        self._tt.clear()
        self._leaf_cache.clear()
        self._in_path.clear()
        self.tt_hits = 0
        self.leaf_evals = 0

    def _evaluate(self, env: MonopolyEnv) -> DecisionNode:
        key = position_key(env)
        cached = self._tt.get(key)
        if cached is not None:
            self.tt_hits += 1
            return cached
        actor = env.whose_turn()
        legal = tuple(env.get_allowed_actions(actor))
        priors, value = self.model.predict(env._get_state(actor), legal, actor, env=env)
        probabilities = np.asarray(tuple(priors.values()), dtype=np.float64)
        if not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise RuntimeError("Policy produced invalid legal priors")
        total = float(probabilities.sum())
        if total <= 0:
            raise RuntimeError("Policy assigned no probability to legal actions")
        priors = {action: probability / total for action, probability in priors.items()}
        if self.leaf_fn is not None:
            cached_leaf = self._leaf_cache.get(key)
            if cached_leaf is None:
                value = np.asarray(self.leaf_fn(env), dtype=np.float64)
                self._leaf_cache[key] = value
                self.leaf_evals += 1
            else:
                value = cached_leaf
            if value.shape != (NUM_PLAYERS,):
                raise RuntimeError(
                    f"leaf_fn must return shape ({NUM_PLAYERS},), got {value.shape}"
                )
            if not np.isfinite(value).all():
                raise RuntimeError("leaf_fn produced non-finite values")
        node = DecisionNode(env, actor, legal, priors, value.astype(np.float64))
        self._tt[key] = node
        return node

    def _simulate(self, node: DecisionNode, depth: int, rng: np.random.Generator) -> np.ndarray:
        nid = id(node)
        if nid in self._in_path:
            return (
                node.mean_value.copy()
                if node.visits
                else node.initial_value.copy()
            )
        self._in_path.add(nid)
        try:
            return super()._simulate(node, depth, rng)
        finally:
            self._in_path.discard(nid)

    def choose_action(
        self,
        game: MonopolyEnv | SharedGame,
        player_id: int,
        decision_seed: int,
    ) -> SearchResult:
        self._reset_caches()
        import monopoly_bench.engine as engine
        import oracle.rollout_leaf as rollout_leaf

        previous = engine.clone_env
        previous_leaf = rollout_leaf.clone_env
        engine.clone_env = fast_clone_env
        rollout_leaf.clone_env = fast_clone_env
        try:
            return self._choose_action_impl(game, player_id, decision_seed)
        finally:
            engine.clone_env = previous
            rollout_leaf.clone_env = previous_leaf

    def _choose_action_impl(
        self,
        game: MonopolyEnv | SharedGame,
        player_id: int,
        decision_seed: int,
    ) -> SearchResult:
        started = time.perf_counter()
        env = fast_clone_env(game)
        if env.done:
            raise ValueError("Cannot choose an action in a finished game")
        if env.whose_turn() != player_id:
            raise ValueError(f"Player {player_id} is not the current actor")
        rng = np.random.default_rng(decision_seed)
        root = self._evaluate(env)
        self.last_root = root
        if len(root.legal_actions) == 1:
            action = root.legal_actions[0]
            return SearchResult(
                chosen_action=action,
                visits={action: 1},
                q_values={action: _vector(root.initial_value)},
                root_value=_vector(root.initial_value),
                simulations=0,
                latency_s=time.perf_counter() - started,
            )
        if self.self_play:
            self._add_root_noise(root, rng)
        sims_used = 0
        for sim in range(self.config.simulations):
            if (
                sim > 0
                and self.deadline_s is not None
                and (time.perf_counter() - started) >= self.deadline_s
            ):
                break
            if sim >= self.early_stop_min_sims and self._visit_lead_stop(root):
                break
            self._simulate(root, 0, rng)
            sims_used = sim + 1
        action = self._final_action(root, rng)
        return SearchResult(
            chosen_action=action,
            visits={candidate: edge.visits for candidate, edge in root.edges.items()},
            q_values={candidate: _vector(edge.q) for candidate, edge in root.edges.items()},
            root_value=_vector(root.mean_value),
            simulations=sims_used,
            latency_s=time.perf_counter() - started,
        )
