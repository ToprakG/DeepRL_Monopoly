"""Wave-1 Max-N leaf swap: rollout / networth / asu / clone."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from monopoly_bench.engine import NUM_PLAYERS, SharedGame, clone_env
from oracle.agent import OracleConfig, OracleAgent, oracle_config_from_args
from oracle.leaves import (
    LEAF_KINDS,
    asu_leaf,
    build_leaf_fn,
    networth_leaf,
    resolve_clone_checkpoint,
)
from oracle.rollout_leaf import margin_vector_from_scores, net_worth_margin_vector
from oracle_v2.agent import OracleV2Agent, default_v2_config


def test_margin_vector_from_scores_matches_net_worth():
    game = SharedGame.new(4, max_rounds=20)
    worth = np.array(
        [float(player.net_worth()) for player in game.env.players],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        margin_vector_from_scores(worth, game.env),
        net_worth_margin_vector(game.env),
    )


def test_unknown_leaf_raises():
    with pytest.raises(ValueError, match="Unknown oracle leaf"):
        build_leaf_fn(OracleConfig(leaf="nope"))


def test_oracle_config_roundtrip_keeps_leaf():
    cfg = OracleConfig(leaf="networth", simulations=32)
    restored = OracleConfig(**asdict(cfg))
    assert restored.leaf == "networth"
    assert restored.simulations == 32


def test_oracle_config_from_args_reads_leaf(tmp_path):
    class Args:
        sims = 32
        horizon = 16
        rollouts = 1
        margin_temperature = 2000.0
        deadline_s = 0
        early_stop_lead = 0
        early_stop_min_sims = 16
        leaf = "clone"
        leaf_checkpoint = tmp_path / "missing.pt"

    cfg = oracle_config_from_args(Args())
    assert cfg.leaf == "clone"
    assert cfg.leaf_checkpoint.endswith("missing.pt")


@pytest.mark.parametrize("kind", ["rollout", "networth", "asu", "asu_plus"])
def test_leaf_shape_and_v2_legal(kind):
    game = SharedGame.new(5, max_rounds=20)
    before = clone_env(game.env)
    actor = game.env.whose_turn()
    cfg = default_v2_config(
        simulations=4,
        rollout_horizon=4,
        rollouts_per_leaf=1,
        max_depth=8,
        leaf=kind,
    )
    leaf = build_leaf_fn(cfg)
    value = leaf(game.env)
    assert value.shape == (NUM_PLAYERS,)
    assert np.isfinite(value).all()
    assert abs(float(value.sum()) - 1.0) < 1e-5
    agent = OracleV2Agent(actor, cfg, seed=0)
    action = agent.choose_action(game.env)
    assert action in before.get_allowed_actions(actor)
    assert game.env.round == before.round


def test_clone_leaf_loads_checkpoint():
    path = resolve_clone_checkpoint(None)
    game = SharedGame.new(6, max_rounds=20)
    before = clone_env(game.env)
    actor = game.env.whose_turn()
    cfg = default_v2_config(
        simulations=4,
        rollout_horizon=4,
        rollouts_per_leaf=1,
        max_depth=8,
        leaf="clone",
        leaf_checkpoint=path,
    )
    agent = OracleV2Agent(actor, cfg, seed=1)
    action = agent.choose_action(game.env)
    assert action in before.get_allowed_actions(actor)


def test_v1_agent_accepts_networth_leaf():
    game = SharedGame.new(7, max_rounds=20)
    actor = game.env.whose_turn()
    agent = OracleAgent(
        actor,
        OracleConfig(
            simulations=4,
            rollout_horizon=4,
            rollouts_per_leaf=1,
            max_depth=8,
            leaf="networth",
        ),
        seed=0,
    )
    assert agent.choose_action(game.env) in game.env.get_allowed_actions(actor)


def test_leaf_kinds_cover_wave1():
    assert LEAF_KINDS == ("rollout", "networth", "asu", "asu_plus", "clone")
    # Touch the direct helpers so a rename breaks this test.
    game = SharedGame.new(8, max_rounds=20)
    assert networth_leaf(game.env, temperature=2000.0).shape == (NUM_PLAYERS,)
    assert asu_leaf(game.env, temperature=2000.0).shape == (NUM_PLAYERS,)
