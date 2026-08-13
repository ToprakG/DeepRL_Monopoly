"""Coverage labeling: all incoming trades + routine subsample."""

from __future__ import annotations

from monopoly_bench.engine import SharedGame
from monopoly_game_engine.actions import ActionType
from monopoly_game_engine.env import PHASE_OUT_OF_TURN, TradeOffer
from oracle.hybrid_config import (
    HybridLabelConfig,
    is_event_checkpoint,
    is_monopoly_relevant_trade,
    is_trade_checkpoint,
    should_label_routine,
)


def test_cash_trade_is_event_even_if_not_monopoly():
    env = SharedGame.new(101, max_rounds=50).env
    env.turn_order = [1, 0, 2, 3]
    env.current_turn_idx = 0
    env.pending_trades = {1: TradeOffer(1, 0, cash_offered=50)}
    env.phase = PHASE_OUT_OF_TURN
    env.out_of_turn_pids = [0]
    legal = set(env.get_allowed_actions(0))
    assert int(ActionType.ACCEPT_TRADE) in legal or int(ActionType.DECLINE_TRADE) in legal
    assert is_trade_checkpoint(env, legal)
    assert is_event_checkpoint(env, legal)
    assert not is_monopoly_relevant_trade(env)


def test_routine_subsample_is_deterministic():
    hits = [
        should_label_routine(seed=7, step=i, actor=0, prob=0.08) for i in range(10_000)
    ]
    rate = sum(hits) / len(hits)
    assert 0.06 <= rate <= 0.10
    assert [
        should_label_routine(seed=7, step=i, actor=0, prob=0.08) for i in range(100)
    ] == [should_label_routine(seed=7, step=i, actor=0, prob=0.08) for i in range(100)]


def test_hybrid_config_default_routine_prob():
    cfg = HybridLabelConfig()
    assert cfg.routine_label_prob == 0.08
    assert "routine_label_prob" in cfg.as_dict()


def test_broad_value_raises_routine_prob():
    from argparse import Namespace

    from oracle.hybrid_config import (
        BROAD_ROUTINE_LABEL_PROB,
        hybrid_label_config_from_args,
    )

    default = HybridLabelConfig()
    broad = HybridLabelConfig.broad_value()
    assert broad.routine_label_prob == BROAD_ROUTINE_LABEL_PROB
    assert broad.routine_label_prob > default.routine_label_prob
    assert broad.simulations == default.simulations

    calibrated = hybrid_label_config_from_args(
        Namespace(
            calibrate=True,
            broad_value=True,
            routine_label_prob=None,
            sims=200,
            horizon=30,
            rollouts=2,
            margin_temperature=2000.0,
        )
    )
    assert calibrated.routine_label_prob == BROAD_ROUTINE_LABEL_PROB
    assert calibrated.simulations == default.simulations

    explicit = hybrid_label_config_from_args(
        Namespace(
            calibrate=True,
            broad_value=True,
            routine_label_prob=0.4,
            sims=200,
            horizon=30,
            rollouts=2,
            margin_temperature=2000.0,
        )
    )
    assert explicit.routine_label_prob == 0.4
