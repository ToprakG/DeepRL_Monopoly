"""Short-budget Max-N PUCT. Goat heuristic is the prior; threat_leaf is the value."""

from __future__ import annotations

import time

import numpy as np

from monopoly_bench.config import SearchConfig
from monopoly_bench.contracts import SearchResult
from monopoly_bench.engine import NUM_PLAYERS, SharedGame
from monopoly_bench.search import DecisionNode, MaxNPUCT
from monopoly_game_engine.env import MonopolyEnv
from oracle_v2.clone import fast_clone_env

from .leaf import threat_leaf

SEARCH_SIMS = 12
SEARCH_DEPTH = 6
SEARCH_WIDTH = 8
SEARCH_DEADLINE_S = 0.08
EARLY_STOP_LEAD = 4
EARLY_STOP_MIN = 6
PRIOR_PEAK = 0.55


def _vector(value: np.ndarray) -> tuple[float, ...]:
    return tuple(float(item) for item in value)


class GoatPriorModel:
    """Peak prior on the goat heuristic; leftover mass is legal-uniform."""

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
        value = np.full(NUM_PLAYERS, 1.0 / NUM_PLAYERS, dtype=np.float32)
        preferred = None
        if env is not None:
            from .agent import heuristic_action

            preferred = int(heuristic_action(env, actor_id, list(legal)))
        if preferred not in legal:
            mass = 1.0 / len(legal)
            return {action: mass for action in legal}, value
        others = [action for action in legal if action != preferred]
        if not others:
            return {preferred: 1.0}, value
        share = (1.0 - self.peak) / len(others)
        priors = {preferred: self.peak}
        for action in others:
            priors[action] = share
        return priors, value


class GoatMaxNPUCT(MaxNPUCT):
    def __init__(
        self,
        *,
        simulations: int = SEARCH_SIMS,
        deadline_s: float = SEARCH_DEADLINE_S,
    ):
        super().__init__(
            GoatPriorModel(),
            SearchConfig(
                simulations=simulations,
                c_puct=1.5,
                max_depth=SEARCH_DEPTH,
                max_width=SEARCH_WIDTH,
            ),
            self_play=False,
        )
        self.leaf_fn = threat_leaf
        self.deadline_s = float(deadline_s)
        self.early_stop_visit_lead = EARLY_STOP_LEAD
        self.early_stop_min_sims = EARLY_STOP_MIN

    def _evaluate(self, env: MonopolyEnv) -> DecisionNode:
        actor = env.whose_turn()
        legal = tuple(env.get_allowed_actions(actor))
        priors, _ignored = self.model.predict(
            env._get_state(actor), legal, actor, env=env
        )
        total = float(sum(priors.values()))
        priors = {action: weight / total for action, weight in priors.items()}
        value = np.asarray(self.leaf_fn(env), dtype=np.float64)
        return DecisionNode(env, actor, legal, priors, value)

    def _visit_lead_stop(self, root: DecisionNode) -> bool:
        if not root.edges:
            return False
        ranked = sorted((edge.visits for edge in root.edges.values()), reverse=True)
        if len(ranked) == 1:
            return ranked[0] >= self.early_stop_visit_lead
        return ranked[0] - ranked[1] >= self.early_stop_visit_lead

    def choose_action(
        self,
        game: MonopolyEnv | SharedGame,
        player_id: int,
        decision_seed: int,
    ) -> SearchResult:
        import monopoly_bench.engine as engine

        started = time.perf_counter()
        previous = engine.clone_env
        engine.clone_env = fast_clone_env
        try:
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
            sims_used = 0
            for sim in range(self.config.simulations):
                if sim > 0 and (time.perf_counter() - started) >= self.deadline_s:
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
        finally:
            engine.clone_env = previous


__all__ = [
    "EARLY_STOP_LEAD",
    "GoatMaxNPUCT",
    "GoatPriorModel",
    "PRIOR_PEAK",
    "SEARCH_DEADLINE_S",
    "SEARCH_SIMS",
]
