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
