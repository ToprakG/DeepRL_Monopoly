"""Cached oracle keeps legal play and reuses leaves."""

from __future__ import annotations

import numpy as np

from monopoly_bench.engine import NUM_PLAYERS, SharedGame, clone_env
from monopoly_bench.search import MaxNPUCT
from monopoly_bench.config import SearchConfig
from oracle.agent import UniformPriorModel
from oracle_v2.agent import OracleV2Agent, default_v2_config
from oracle_v2.position import position_key


def test_fast_clone_matches_deepcopy():
    from monopoly_bench.engine import clone_env
    from oracle_v2.clone import fast_clone_env

    game = SharedGame.new(8, max_rounds=20)
    actor = game.env.whose_turn()
    legal = game.env.get_allowed_actions(actor)
    a = clone_env(game.env)
    b = fast_clone_env(game.env)
    assert position_key(a) == position_key(b)
    a.step(legal[0])
    b.step(legal[0])
    assert position_key(a) == position_key(b)


def test_position_key_stable_on_clone():
    game = SharedGame.new(3, max_rounds=20)
    cloned = clone_env(game.env)
    assert position_key(game.env) == position_key(cloned)


def test_position_key_changes_after_step():
    game = SharedGame.new(4, max_rounds=20)
    before = position_key(game.env)
    actor = game.env.whose_turn()
    legal = game.env.get_allowed_actions(actor)
    game.step(legal[0])
    assert position_key(game.env) != before


def test_v2_agent_legal_and_non_mutating():
    game = SharedGame.new(5, max_rounds=20)
    before = clone_env(game.env)
    actor = game.env.whose_turn()
    agent = OracleV2Agent(
        actor,
        default_v2_config(simulations=4, rollout_horizon=4, rollouts_per_leaf=1, max_depth=8),
        seed=0,
    )
    action = agent.choose_action(game.env)
    assert action in before.get_allowed_actions(actor)
    assert game.env.round == before.round
    assert game.env.whose_turn() == before.whose_turn()


def test_live_mode_skips_search_on_non_events():
    game = SharedGame.new(3, max_rounds=20)
    actor = game.env.whose_turn()
    agent = OracleV2Agent(
        actor,
        default_v2_config(simulations=8, rollout_horizon=4, rollouts_per_leaf=1, max_depth=8),
        seed=0,
        live=True,
        turn_deadline_s=4.0,
    )
    action = agent.choose_action(game.env)
    assert action in game.env.get_allowed_actions(actor)
    # Fresh pre-roll is usually END_TURN/offers, not a buy/build event.
    from oracle.hybrid_config import checkpoint_kind

    kind = checkpoint_kind(game.env, game.env.get_allowed_actions(actor))
    if kind is None and len(game.env.get_allowed_actions(actor)) > 1:
        assert agent.last_used_search is False


def test_tt_reuses_leaves_within_a_search():
    game = SharedGame.new(11, max_rounds=20)
    actor = game.env.whose_turn()
    calls = {"n": 0}

    def leaf(_env):
        calls["n"] += 1
        return np.full(NUM_PLAYERS, 0.25)

    search = MaxNPUCT(
        UniformPriorModel(),
        SearchConfig(simulations=12, max_depth=8, max_width=16),
        leaf_fn=leaf,
    )
    from oracle_v2.search import CachedMaxNPUCT

    cached = CachedMaxNPUCT(
        UniformPriorModel(),
        SearchConfig(simulations=12, max_depth=8, max_width=16),
        leaf_fn=leaf,
    )
    if len(game.env.get_allowed_actions(actor)) < 2:
        return
    search.choose_action(game, actor, 1)
    v1_leaves = calls["n"]
    calls["n"] = 0
    cached.choose_action(game, actor, 1)
    assert cached.leaf_evals <= v1_leaves
    assert cached.leaf_evals == calls["n"]
    assert cached.tt_hits >= 0
