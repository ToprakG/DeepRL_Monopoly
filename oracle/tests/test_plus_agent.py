"""oracle-plus-v1 toggles return a legal action."""

from __future__ import annotations

from monopoly_bench.engine import SharedGame, clone_env
from monopoly_game_engine.constants import COLOR_GROUPS
from oracle.agent import OracleConfig
from oracle.eval_h2h import ORACLE_PLUS_ID, PLUS_FIELD_LINEUP, _is_oracle_policy, _parse_lineup
from oracle.plus_agent import OraclePlusAgent, resolve_plus_config


def test_plus_id_is_oracle_policy():
    assert _is_oracle_policy(ORACLE_PLUS_ID)
    assert PLUS_FIELD_LINEUP[0] == ORACLE_PLUS_ID
    assert len(_parse_lineup(",".join(PLUS_FIELD_LINEUP))) == 4


def test_resolve_plus_defaults_networth_and_all_on():
    cfg = resolve_plus_config(OracleConfig())
    assert cfg.leaf == "asu"
    assert cfg.one_ply is False
    assert cfg.solvency is False
    assert cfg.denial is True
    assert cfg.completing_trade is True
    assert cfg.auction is True
    assert cfg.inncenta_trade is True
    assert cfg.networth_mix == 0.0
    assert cfg.auction_kind == "inncenta"
    assert cfg.family_body == "rules"
    assert cfg.one_ply_trades is False
    assert cfg.phase_switch is True
    assert cfg.cash_gate is True
    assert cfg.build_first is True
    assert cfg.race_buy is True
    assert cfg.lethal_jail is True


def test_explicit_off_survives():
    cfg = resolve_plus_config(
        OracleConfig(one_ply=False, denial=False, auction=False, inncenta_trade=False, leaf="asu")
    )
    assert cfg.one_ply is False
    assert cfg.denial is False
    assert cfg.auction is False
    assert cfg.inncenta_trade is False
    assert cfg.leaf == "asu"
    assert cfg.solvency is False


def test_plus_agent_legal_and_non_mutating():
    game = SharedGame.new(9, max_rounds=20)
    before = clone_env(game.env)
    actor = game.env.whose_turn()
    agent = OraclePlusAgent(actor, OracleConfig(leaf="networth"), seed=0)
    action = agent.choose_action(game.env)
    assert action in before.get_allowed_actions(actor)
    assert game.env.round == before.round


def test_greedy_toggle_still_legal():
    game = SharedGame.new(10, max_rounds=20)
    actor = game.env.whose_turn()
    agent = OraclePlusAgent(
        actor,
        OracleConfig(one_ply=False, solvency=True, denial=True, completing_trade=True),
        seed=1,
    )
    assert agent.choose_action(game.env) in game.env.get_allowed_actions(actor)


def test_race_role_contests_opponent_presence():
    from oracle.plus_loop import race_role

    game = SharedGame.new(11, max_rounds=20)
    env = game.env
    assert race_role(env, 0, 1) == "start"
    brown = COLOR_GROUPS["brown"]
    env.properties[brown[0]].owner = 1
    if env.properties[brown[0]] not in env.players[1].properties:
        env.players[1].properties.append(env.properties[brown[0]])
    assert race_role(env, 0, brown[1]) == "veto"
    assert race_role(env, 0, brown[1], deny=False) == "skip"


def test_auction_skips_when_not_in_auction():
    game = SharedGame.new(12, max_rounds=20)
    actor = game.env.whose_turn()
    agent = OraclePlusAgent(actor, OracleConfig(leaf="networth"), seed=0)
    legal = list(game.env.get_allowed_actions(actor))
    assert agent._auction_action(game.env, legal) is None


def test_auction_bids_when_headroom_exists():
    from monopoly_game_engine.actions import AuctionAction
    from monopoly_game_engine.env import PHASE_AUCTION

    game = SharedGame.new(13, max_rounds=20)
    env = game.env
    brown = COLOR_GROUPS["brown"]
    env.phase = PHASE_AUCTION
    env.auction_property_id = brown[1]
    env.auction_current_pid = 0
    env.auction_bidders = [0, 1, 2, 3]
    env.auction_high_bid = 0
    env.auction_highest_bidder = None
    env.properties[brown[0]].owner = 0
    if env.properties[brown[0]] not in env.players[0].properties:
        env.players[0].properties.append(env.properties[brown[0]])
    agent = OraclePlusAgent(0, OracleConfig(leaf="networth"), seed=0)
    legal = list(env.get_allowed_actions(0))
    bid = agent._auction_action(env, legal)
    assert bid is not None
    assert bid in legal
    assert bid != int(AuctionAction.PASS)
    before_cash = env.players[0].cash
    action = agent.choose_action(env)
    assert action == bid
    assert env.players[0].cash == before_cash


def test_no_auction_still_legal_in_auction_phase():
    from monopoly_game_engine.env import PHASE_AUCTION

    game = SharedGame.new(14, max_rounds=20)
    env = game.env
    env.phase = PHASE_AUCTION
    env.auction_property_id = 1
    env.auction_current_pid = 0
    env.auction_bidders = [0, 1, 2, 3]
    env.auction_high_bid = 0
    agent = OraclePlusAgent(
        0, OracleConfig(leaf="networth", auction=False, one_ply=True), seed=0
    )
    action = agent.choose_action(env)
    assert action in env.get_allowed_actions(0)


def _auction_env(seed: int = 20):
    from monopoly_game_engine.env import PHASE_AUCTION

    game = SharedGame.new(seed, max_rounds=20)
    env = game.env
    brown = COLOR_GROUPS["brown"]
    env.phase = PHASE_AUCTION
    env.auction_property_id = brown[1]
    env.auction_current_pid = 0
    env.auction_bidders = [0, 1, 2, 3]
    env.auction_high_bid = 0
    env.auction_highest_bidder = None
    env.properties[brown[0]].owner = 0
    if env.properties[brown[0]] not in env.players[0].properties:
        env.players[0].properties.append(env.properties[brown[0]])
    return env


def test_asu_delta_auction_returns_legal():
    env = _auction_env(21)
    agent = OraclePlusAgent(
        0, OracleConfig(leaf="networth", auction=True, auction_kind="asu_delta"), seed=0
    )
    action = agent.choose_action(env)
    assert action in env.get_allowed_actions(0)


def test_unique_survival_toggles_legal():
    game = SharedGame.new(23, max_rounds=20)
    actor = game.env.whose_turn()
    agent = OraclePlusAgent(
        actor,
        OracleConfig(
            leaf="networth",
            cash_gate=True,
            build_first=True,
            race_buy=True,
            lethal_jail=True,
            auction_kind="asu_delta",
        ),
        seed=0,
    )
    action = agent.choose_action(game.env)
    assert action in game.env.get_allowed_actions(actor)


def test_live_rent_zero_on_empty_board():
    from oracle.plus_steals import max_live_rent

    game = SharedGame.new(24, max_rounds=20)
    assert max_live_rent(game.env, 0) == 0.0


def test_incoming_trade_declines_completing_opponent():
    from monopoly_game_engine.actions import ActionType
    from monopoly_game_engine.env import TradeOffer, PHASE_OUT_OF_TURN
    from oracle.plus_steals import incoming_trade_action

    game = SharedGame.new(31, max_rounds=20)
    env = game.env
    brown = COLOR_GROUPS["brown"]
    env.properties[brown[0]].owner = 1
    env.properties[brown[1]].owner = 0
    if env.properties[brown[0]] not in env.players[1].properties:
        env.players[1].properties.append(env.properties[brown[0]])
    if env.properties[brown[1]] not in env.players[0].properties:
        env.players[0].properties.append(env.properties[brown[1]])
    env.pending_trades[1] = TradeOffer(
        1, 0, requested_prop=env.properties[brown[1]]
    )
    env.phase = PHASE_OUT_OF_TURN
    legal = [int(ActionType.ACCEPT_TRADE), int(ActionType.DECLINE_TRADE)]
    assert incoming_trade_action(env, 0, legal) == int(ActionType.DECLINE_TRADE)


def test_incoming_trade_accepts_completing_us():
    from monopoly_game_engine.actions import ActionType
    from monopoly_game_engine.env import TradeOffer
    from oracle.plus_steals import incoming_trade_action

    game = SharedGame.new(32, max_rounds=20)
    env = game.env
    brown = COLOR_GROUPS["brown"]
    env.properties[brown[0]].owner = 0
    env.properties[brown[1]].owner = 1
    if env.properties[brown[0]] not in env.players[0].properties:
        env.players[0].properties.append(env.properties[brown[0]])
    if env.properties[brown[1]] not in env.players[1].properties:
        env.players[1].properties.append(env.properties[brown[1]])
    env.pending_trades[1] = TradeOffer(
        1, 0, offered_prop=env.properties[brown[1]]
    )
    legal = [int(ActionType.ACCEPT_TRADE), int(ActionType.DECLINE_TRADE)]
    assert incoming_trade_action(env, 0, legal) == int(ActionType.ACCEPT_TRADE)


def test_cap_weight_ramps_after_midpoint():
    from oracle.plus_steals import cap_weight

    game = SharedGame.new(33, max_rounds=200)
    env = game.env
    env.round = 0
    assert cap_weight(env) == 0.0
    env.round = 100
    assert cap_weight(env) == 0.0
    env.round = 200
    assert cap_weight(env) == 1.0


def test_scrap_buy_late_when_cash_holds():
    from monopoly_game_engine.actions import ActionType
    from oracle.plus_steals import scrap_buy_action

    game = SharedGame.new(34, max_rounds=200)
    env = game.env
    env.round = 160
    env.players[0].position = 1
    env.players[0].cash = 1500
    legal = [int(ActionType.BUY_PROPERTY), int(ActionType.END_TURN)]
    assert scrap_buy_action(env, 0, legal) == int(ActionType.BUY_PROPERTY)
    env.round = 10
    assert scrap_buy_action(env, 0, legal) is None


def test_dead_mortgage_on_blocked_colour():
    from monopoly_game_engine.actions import OFFSETS
    from monopoly_game_engine.constants import PROPERTY_IDS
    from oracle.plus_steals import dead_mortgage_action

    game = SharedGame.new(35, max_rounds=80)
    env = game.env
    brown = COLOR_GROUPS["brown"]
    env.properties[brown[0]].owner = 0
    env.properties[brown[1]].owner = 1
    if env.properties[brown[0]] not in env.players[0].properties:
        env.players[0].properties.append(env.properties[brown[0]])
    if env.properties[brown[1]] not in env.players[1].properties:
        env.players[1].properties.append(env.properties[brown[1]])
    env.players[0].cash = 40
    idx = PROPERTY_IDS.index(brown[0])
    action = OFFSETS["mortgage"] + idx
    legal = [action, 1]
    assert dead_mortgage_action(env, 0, legal) == action


def test_phase_switch_mixes_late():
    from oracle.plus_agent import _own_score

    game = SharedGame.new(36, max_rounds=200)
    env = game.env
    env.round = 200
    cfg = resolve_plus_config(OracleConfig(leaf="asu"))
    assert cfg.phase_switch is True
    # Should not crash; late-game mix uses net worth.
    assert isinstance(_own_score(env, 0, cfg), float)


def test_cash_floor_empty_board_is_bail():
    from monopoly_game_engine.constants import JAIL_BAIL
    from oracle.plus_steals import cash_floor

    game = SharedGame.new(41, max_rounds=20)
    assert cash_floor(game.env, 0) == float(JAIL_BAIL)


def test_cash_floor_tracks_live_hotel():
    from oracle.plus_steals import cash_floor, max_live_rent

    game = SharedGame.new(42, max_rounds=20)
    env = game.env
    orange = COLOR_GROUPS["orange"]
    for sq in orange:
        env.properties[sq].owner = 1
        env.properties[sq].is_monopoly = True
        if env.properties[sq] not in env.players[1].properties:
            env.players[1].properties.append(env.properties[sq])
    env.properties[orange[0]].houses = 5
    live = max_live_rent(env, 0)
    assert live > 50
    assert cash_floor(env, 0) == live


def test_jail_sits_when_hotel_exists():
    from monopoly_game_engine.actions import ActionType
    from oracle.plus_steals import lethal_jail_action

    game = SharedGame.new(43, max_rounds=20)
    env = game.env
    orange = COLOR_GROUPS["orange"]
    for sq in orange:
        env.properties[sq].owner = 1
        env.properties[sq].is_monopoly = True
        if env.properties[sq] not in env.players[1].properties:
            env.players[1].properties.append(env.properties[sq])
    env.properties[orange[0]].houses = 5
    legal = [
        int(ActionType.PAY_BAIL),
        int(ActionType.ROLL_DICE),
        int(ActionType.END_TURN),
    ]
    action = lethal_jail_action(env, 0, legal)
    assert action in (int(ActionType.ROLL_DICE), int(ActionType.END_TURN))
    assert action != int(ActionType.PAY_BAIL)


def test_spend_floor_ignores_distant_hotel():
    from monopoly_game_engine.constants import JAIL_BAIL
    from oracle.plus_steals import cash_floor, spend_floor

    game = SharedGame.new(44, max_rounds=20)
    env = game.env
    env.players[0].position = 0
    orange = COLOR_GROUPS["orange"]
    for sq in orange:
        env.properties[sq].owner = 1
        env.properties[sq].is_monopoly = True
        if env.properties[sq] not in env.players[1].properties:
            env.players[1].properties.append(env.properties[sq])
    env.properties[orange[0]].houses = 5
    assert cash_floor(env, 0) > 50
    assert spend_floor(env, 0) == float(JAIL_BAIL)


def test_build_despite_distant_hotel():
    from monopoly_game_engine.actions import OFFSETS
    from monopoly_game_engine.constants import REAL_ESTATE_IDS
    from oracle.plus_steals import build_first_action

    game = SharedGame.new(45, max_rounds=20)
    env = game.env
    env.players[0].position = 0
    env.players[0].cash = 200
    brown = COLOR_GROUPS["brown"]
    orange = COLOR_GROUPS["orange"]
    for sq in brown:
        env.properties[sq].owner = 0
        env.properties[sq].is_monopoly = True
        if env.properties[sq] not in env.players[0].properties:
            env.players[0].properties.append(env.properties[sq])
    for sq in orange:
        env.properties[sq].owner = 1
        env.properties[sq].is_monopoly = True
        if env.properties[sq] not in env.players[1].properties:
            env.players[1].properties.append(env.properties[sq])
    env.properties[orange[0]].houses = 5
    idx = REAL_ESTATE_IDS.index(brown[0])
    action = OFFSETS["improve_house"] + idx
    assert build_first_action(env, 0, [action, 1]) == action


def test_jail_leaves_on_railroads_only():
    from monopoly_game_engine.actions import ActionType
    from oracle.plus_steals import lethal_jail_action

    game = SharedGame.new(46, max_rounds=20)
    env = game.env
    for sq in COLOR_GROUPS["railroad"]:
        env.properties[sq].owner = 1
        if env.properties[sq] not in env.players[1].properties:
            env.players[1].properties.append(env.properties[sq])
    legal = [
        int(ActionType.PAY_BAIL),
        int(ActionType.ROLL_DICE),
        int(ActionType.END_TURN),
    ]
    assert lethal_jail_action(env, 0, legal) is None


def test_race_buy_takes_uncontested_brown():
    from monopoly_game_engine.actions import ActionType
    from oracle.plus_steals import race_buy_action

    game = SharedGame.new(47, max_rounds=20)
    env = game.env
    env.players[0].position = COLOR_GROUPS["brown"][0]
    env.players[0].cash = 80
    legal = [int(ActionType.BUY_PROPERTY), int(ActionType.END_TURN)]
    assert race_buy_action(env, 0, legal) == int(ActionType.BUY_PROPERTY)


def test_race_buy_skips_uncontested_orange():
    from monopoly_game_engine.actions import ActionType
    from oracle.plus_steals import race_buy_action

    game = SharedGame.new(48, max_rounds=20)
    env = game.env
    env.players[0].position = COLOR_GROUPS["orange"][0]
    env.players[0].cash = 400
    legal = [int(ActionType.BUY_PROPERTY), int(ActionType.END_TURN)]
    assert race_buy_action(env, 0, legal) is None


def test_unowned_darkblue_books_at_solo_multiple():
    game = SharedGame.new(49, max_rounds=20)
    agent = OraclePlusAgent(0, OracleConfig(leaf="networth"), seed=0)
    boardwalk = COLOR_GROUPS["darkblue"][1]
    assert agent._property_worth(game.env, boardwalk) == 2.5 * 400


def test_complete_floor_empty_board_is_zero():
    from oracle.plus_steals import complete_floor

    game = SharedGame.new(50, max_rounds=20)
    assert complete_floor(game.env, 0) == 0.0


def test_active_colour_is_opening_race_on_empty_board():
    from oracle.plus_loop import OPENING_RACE, active_colour

    game = SharedGame.new(51, max_rounds=20)
    plan = active_colour(game.env, 0)
    assert plan in OPENING_RACE
    assert plan != "orange"
    assert plan != "darkblue"


def test_active_colour_skips_darkblue_when_pink_blocked():
    from oracle.plus_loop import active_colour

    game = SharedGame.new(62, max_rounds=20)
    env = game.env
    pink = COLOR_GROUPS["pink"]
    env.properties[pink[0]].owner = 1
    if env.properties[pink[0]] not in env.players[1].properties:
        env.players[1].properties.append(env.properties[pink[0]])
    plan = active_colour(env, 0)
    assert plan == "brown"


def test_plan_buy_takes_uncontested_opening_colour():
    from monopoly_game_engine.actions import ActionType
    from oracle.plus_loop import plan_buy_action

    game = SharedGame.new(52, max_rounds=20)
    env = game.env
    env.players[0].position = COLOR_GROUPS["lightblue"][0]
    env.players[0].cash = 400
    legal = [int(ActionType.BUY_PROPERTY), int(ActionType.END_TURN)]
    assert plan_buy_action(env, 0, legal) == int(ActionType.BUY_PROPERTY)


def test_plan_buy_skips_blocked_colour_without_denial():
    from monopoly_game_engine.actions import ActionType
    from oracle.plus_loop import plan_buy_action

    game = SharedGame.new(53, max_rounds=20)
    env = game.env
    orange = COLOR_GROUPS["orange"]
    env.properties[orange[0]].owner = 1
    if env.properties[orange[0]] not in env.players[1].properties:
        env.players[1].properties.append(env.properties[orange[0]])
    env.players[0].position = orange[1]
    env.players[0].cash = 400
    legal = [int(ActionType.BUY_PROPERTY), int(ActionType.END_TURN)]
    assert plan_buy_action(env, 0, legal, deny=False) is None


def test_plan_buy_contests_blocked_colour_with_denial():
    from monopoly_game_engine.actions import ActionType
    from oracle.plus_loop import plan_buy_action

    game = SharedGame.new(53, max_rounds=20)
    env = game.env
    orange = COLOR_GROUPS["orange"]
    env.properties[orange[0]].owner = 1
    if env.properties[orange[0]] not in env.players[1].properties:
        env.players[1].properties.append(env.properties[orange[0]])
    env.players[0].position = orange[1]
    env.players[0].cash = 400
    legal = [int(ActionType.BUY_PROPERTY), int(ActionType.END_TURN)]
    assert plan_buy_action(env, 0, legal) == int(ActionType.BUY_PROPERTY)


def test_plan_auction_finish_uses_cash_floor():
    from oracle.plus_loop import plan_auction_ceiling
    from oracle.plus_steals import complete_floor

    game = SharedGame.new(54, max_rounds=20)
    env = game.env
    brown = COLOR_GROUPS["brown"]
    env.properties[brown[0]].owner = 0
    if env.properties[brown[0]] not in env.players[0].properties:
        env.players[0].properties.append(env.properties[brown[0]])
    env.players[0].cash = 1500
    ceiling = plan_auction_ceiling(env, 0, brown[1])
    assert ceiling == 1500.0 - complete_floor(env, 0)
    assert ceiling > 400.0


def _give(env, pid: int, square: int) -> None:
    prop = env.properties[square]
    prop.owner = pid
    if prop not in env.players[pid].properties:
        env.players[pid].properties.append(prop)


def test_plan_buy_takes_open_backup_before_first_set():
    from monopoly_game_engine.actions import ActionType
    from oracle.plus_loop import plan_buy_action

    game = SharedGame.new(55, max_rounds=20)
    env = game.env
    env.players[0].position = COLOR_GROUPS["red"][0]
    env.players[0].cash = 400
    legal = [int(ActionType.BUY_PROPERTY), int(ActionType.END_TURN)]
    assert plan_buy_action(env, 0, legal) == int(ActionType.BUY_PROPERTY)


def test_plan_auction_backup_ceiling_is_positive_before_first_set():
    from oracle.plus_loop import plan_auction_ceiling

    game = SharedGame.new(56, max_rounds=20)
    env = game.env
    env.players[0].cash = 1500
    red = COLOR_GROUPS["red"][0]
    assert plan_auction_ceiling(env, 0, red) > 0.0


def test_plan_buy_takes_other_open_colour():
    from monopoly_game_engine.actions import ActionType
    from oracle.plus_loop import plan_buy_action

    game = SharedGame.new(61, max_rounds=20)
    env = game.env
    env.players[0].position = COLOR_GROUPS["orange"][0]
    env.players[0].cash = 400
    legal = [int(ActionType.BUY_PROPERTY), int(ActionType.END_TURN)]
    assert plan_buy_action(env, 0, legal) == int(ActionType.BUY_PROPERTY)


def test_plan_incoming_accepts_non_completing_weapon():
    from monopoly_game_engine.actions import ActionType
    from monopoly_game_engine.env import TradeOffer
    from oracle.plus_loop import plan_incoming_action

    game = SharedGame.new(57, max_rounds=20)
    env = game.env
    red = COLOR_GROUPS["red"]
    _give(env, 0, red[0])
    _give(env, 1, red[1])
    env.pending_trades[1] = TradeOffer(1, 0, offered_prop=env.properties[red[1]])
    legal = [int(ActionType.ACCEPT_TRADE), int(ActionType.DECLINE_TRADE)]
    assert plan_incoming_action(env, 0, legal) == int(ActionType.ACCEPT_TRADE)


def test_plan_incoming_declines_completing_opponent():
    from monopoly_game_engine.actions import ActionType
    from monopoly_game_engine.env import TradeOffer
    from oracle.plus_loop import plan_incoming_action

    game = SharedGame.new(58, max_rounds=20)
    env = game.env
    brown = COLOR_GROUPS["brown"]
    _give(env, 1, brown[0])
    _give(env, 0, brown[1])
    env.pending_trades[1] = TradeOffer(
        1, 0, requested_prop=env.properties[brown[1]]
    )
    legal = [int(ActionType.ACCEPT_TRADE), int(ActionType.DECLINE_TRADE)]
    assert plan_incoming_action(env, 0, legal, "orange") == int(
        ActionType.DECLINE_TRADE
    )


def test_plan_trade_takes_completing_orange():
    from monopoly_game_engine.actions import OFFSETS
    from oracle.plus_loop import plan_trade_action

    game = SharedGame.new(59, max_rounds=20)
    env = game.env
    orange = COLOR_GROUPS["orange"]
    brown = COLOR_GROUPS["brown"]
    _give(env, 0, orange[0])
    _give(env, 0, orange[1])
    _give(env, 0, brown[0])
    _give(env, 1, orange[2])
    _give(env, 1, brown[1])
    env.players[0].cash = 1500
    env.current_turn_idx = env.turn_order.index(0)
    legal = list(env.get_allowed_actions(0))
    action = plan_trade_action(env, 0, legal)
    assert action is not None
    assert action in legal
    assert OFFSETS["buy_trade"] <= action


def test_plan_trade_cash_buys_completing():
    from monopoly_game_engine.actions import OFFSETS
    from oracle.plus_loop import plan_trade_action

    game = SharedGame.new(60, max_rounds=20)
    env = game.env
    brown = COLOR_GROUPS["brown"]
    _give(env, 0, brown[0])
    _give(env, 1, brown[1])
    env.players[0].cash = 1500
    env.current_turn_idx = env.turn_order.index(0)
    legal = list(env.get_allowed_actions(0))
    action = plan_trade_action(env, 0, legal)
    assert action is not None
    assert OFFSETS["buy_trade"] <= action < OFFSETS["sell_trade"]

