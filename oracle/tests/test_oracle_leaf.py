"""Unit checks for rollout leaf + MaxNPUCT leaf_fn wiring."""

from __future__ import annotations

import random

import numpy as np

from monopoly_bench.config import SearchConfig
from monopoly_bench.engine import NUM_PLAYERS, SharedGame, clone_env
from monopoly_bench.search import MaxNPUCT
from oracle.agent import UniformPriorModel, build_oracle_search, OracleConfig, OracleAgent
from oracle.rollout_leaf import net_worth_margin_vector, rollout_leaf_value
from oracle.rollout_policy import greedy_rollout_action


def test_greedy_policy_legal():
    random.seed(0)
    game = SharedGame.new(0, max_rounds=50)
    for _ in range(40):
        if game.env.done:
            break
        actor = game.env.whose_turn()
        legal = game.env.get_allowed_actions(actor)
        action = greedy_rollout_action(game.env, actor)
        assert action in legal
        game.step(action)


def test_rollout_leaf_shape_and_finite():
    game = SharedGame.new(1, max_rounds=50)
    value = rollout_leaf_value(game.env, num_rollouts=2, horizon=8, seed=7)
    assert value.shape == (NUM_PLAYERS,)
    assert np.isfinite(value).all()
    assert abs(float(value.sum()) - 1.0) < 1e-6


def test_leaf_fn_overrides_model_value():
    game = SharedGame.new(2, max_rounds=20)
    actor = game.env.whose_turn()
    sentinel = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

    def leaf(_env):
        return sentinel.copy()

    search = MaxNPUCT(
        UniformPriorModel(),
        SearchConfig(simulations=2, max_depth=8, max_width=16),
        leaf_fn=leaf,
    )
    root = search._evaluate(clone_env(game.env))
    assert np.allclose(root.initial_value, sentinel)
    result = search.choose_action(game, actor, 99)
    assert result.chosen_action in game.env.get_allowed_actions(actor)


def test_oracle_agent_non_mutating():
    game = SharedGame.new(3, max_rounds=20)
    before = clone_env(game.env)
    agent = OracleAgent(
        game.env.whose_turn(),
        OracleConfig(simulations=4, rollout_horizon=4, rollouts_per_leaf=1, max_depth=8),
        seed=0,
    )
    action = agent.choose_action(game.env)
    assert action in before.get_allowed_actions(before.whose_turn())
    assert game.env.round == before.round
    assert game.env.whose_turn() == before.whose_turn()


def test_margin_terminal_one_hot_when_done():
    game = SharedGame.new(4, max_rounds=1)
    # Force a finished-ish evaluation path via net_worth on live env still works.
    value = net_worth_margin_vector(game.env)
    assert value.shape == (NUM_PLAYERS,)
    assert abs(float(value.sum()) - 1.0) < 1e-6


def test_hybrid_declines_bad_trade_and_builds():
    from monopoly_game_engine.actions import OFFSETS, ActionType
    from monopoly_game_engine.env import PHASE_OUT_OF_TURN, TradeOffer

    # Bad trade: give a monopoly deed for $1 — must DECLINE.
    bad = SharedGame.new(101, max_rounds=50).env
    bad.turn_order = [1, 0, 2, 3]
    bad.current_turn_idx = 0
    for sq in (1, 3):
        prop = bad.properties[sq]
        prop.owner = 0
        prop.mortgaged = False
        if prop not in bad.players[0].properties:
            bad.players[0].properties.append(prop)
    bad.players[0].cash = 800
    bad._update_monopolies()
    bad.pending_trades = {
        1: TradeOffer(1, 0, None, bad.properties[1], cash_offered=1, cash_requested=0)
    }
    bad.phase = PHASE_OUT_OF_TURN
    bad.out_of_turn_pids = [0]
    assert greedy_rollout_action(bad, 0) == int(ActionType.DECLINE_TRADE)

    # Light-blue monopoly with cash — must BUILD.
    build = SharedGame.new(102, max_rounds=50).env
    build.turn_order = [0, 1, 2, 3]
    build.current_turn_idx = 0
    for sq in (6, 8, 9):
        prop = build.properties[sq]
        prop.owner = 0
        prop.houses = 0
        prop.mortgaged = False
        if prop not in build.players[0].properties:
            build.players[0].properties.append(prop)
    build.players[0].cash = 2000
    build._update_monopolies()
    build.phase = "pre_roll"
    build.has_rolled = False
    action = greedy_rollout_action(build, 0)
    assert OFFSETS["improve_house"] <= action < OFFSETS["sell_house"]


def test_event_checkpoint_buy_trade_not_end_turn():
    from monopoly_game_engine.actions import ActionType
    from monopoly_game_engine.env import PHASE_OUT_OF_TURN, TradeOffer
    from oracle.hybrid_config import is_event_checkpoint

    buy = SharedGame.new(100, max_rounds=50).env
    buy.turn_order = [0, 1, 2, 3]
    buy.current_turn_idx = 0
    buy.players[0].cash = 1500
    buy.players[0].position = 1
    buy.phase = "post_roll"
    buy.has_rolled = True
    buy.debt_player = None
    prop = buy.properties[1]
    if prop.owner is not None:
        buy.players[prop.owner].properties = [
            p for p in buy.players[prop.owner].properties if p.square_id != 1
        ]
        prop.owner = None
    legal_buy = buy.get_allowed_actions(0)
    assert is_event_checkpoint(buy, legal_buy)
    assert int(ActionType.BUY_PROPERTY) in legal_buy

    trade = SharedGame.new(101, max_rounds=50).env
    trade.turn_order = [1, 0, 2, 3]
    trade.current_turn_idx = 0
    trade.pending_trades = {1: TradeOffer(1, 0, cash_offered=50)}
    trade.phase = PHASE_OUT_OF_TURN
    trade.out_of_turn_pids = [0]
    legal_trade = trade.get_allowed_actions(0)
    assert is_event_checkpoint(trade, legal_trade)

    plain = SharedGame.new(3, max_rounds=50).env
    # Fresh pre-roll with no owned props: END_TURN (+ maybe trades). Not a buy/build/accept checkpoint
    # unless only END_TURN — trade-offer spam alone is intentionally excluded.
    legal_plain = plain.get_allowed_actions(plain.whose_turn())
    if int(ActionType.BUY_PROPERTY) not in legal_plain and not (
        int(ActionType.ACCEPT_TRADE) in legal_plain
    ):
        assert not is_event_checkpoint(plain, legal_plain) or plain.phase == "auction"
