"""oracle-jail-v1: jail-side buy, three-house cap, hijack trades."""

from __future__ import annotations

from monopoly_bench.engine import SharedGame, clone_env
from monopoly_game_engine.actions import OFFSETS, ActionType
from monopoly_game_engine.constants import COLOR_GROUPS, PROPERTIES
from monopoly_game_engine.env import TradeOffer
from oracle.eval_h2h import JAIL_FIELD_LINEUP, JAIL_ID, _is_oracle_policy, _parse_lineup
from oracle.jail import JailAgent
from oracle.jail.loop import (
    THREE_HOUSES,
    auction_ceiling,
    build_action,
    buy_action,
    debt_action,
    in_window,
    incoming_action,
    trade_action,
)


def _give(env, pid: int, square: int) -> None:
    prop = env.properties[square]
    prop.owner = pid
    if prop not in env.players[pid].properties:
        env.players[pid].properties.append(prop)


def _seat(env, pid: int) -> None:
    env.current_turn_idx = env.turn_order.index(pid)


def test_jail_id_is_oracle_policy():
    assert _is_oracle_policy(JAIL_ID)
    assert JAIL_FIELD_LINEUP[0] == JAIL_ID
    assert len(_parse_lineup(",".join(JAIL_FIELD_LINEUP))) == 4


def test_jail_agent_legal_and_non_mutating():
    game = SharedGame.new(70, max_rounds=20)
    before = clone_env(game.env)
    actor = game.env.whose_turn()
    action = JailAgent(actor, seed=0).choose_action(game.env)
    assert action in before.get_allowed_actions(actor)
    assert game.env.round == before.round


def test_dice_window_from_jail_covers_orange():
    orange = COLOR_GROUPS["orange"]
    assert in_window(10, orange)
    assert not in_window(10, COLOR_GROUPS["green"])
    assert in_window(0, {5, 7, 9})
    assert not in_window(0, {4, 10})


def test_buy_takes_jail_side_not_ghost():
    game = SharedGame.new(71, max_rounds=20)
    env = game.env
    env.players[0].cash = 1500
    env.players[0].position = COLOR_GROUPS["red"][0]
    legal = [int(ActionType.BUY_PROPERTY), int(ActionType.END_TURN)]
    assert buy_action(env, 0, legal, "orange") == int(ActionType.BUY_PROPERTY)

    env.players[0].position = COLOR_GROUPS["green"][0]
    assert buy_action(env, 0, legal, "orange") is None


def test_auction_ghost_ceiling_is_half_list():
    game = SharedGame.new(72, max_rounds=20)
    env = game.env
    env.players[0].cash = 1500
    green = COLOR_GROUPS["green"][0]
    ceiling = auction_ceiling(env, 0, green, "orange")
    assert ceiling == 0.5 * float(PROPERTIES[green]["price"])
    red = COLOR_GROUPS["red"][0]
    assert auction_ceiling(env, 0, red, "orange") == float(PROPERTIES[red]["price"])


def test_build_stops_at_three_houses():
    game = SharedGame.new(73, max_rounds=20)
    env = game.env
    pink = COLOR_GROUPS["pink"]
    for square in pink:
        _give(env, 0, square)
        env.properties[square].houses = THREE_HOUSES
    env._update_monopolies()
    env.players[0].cash = 1500
    env.players[1].position = 0
    env.players[2].position = 0
    env.players[3].position = 0
    _seat(env, 0)
    legal = list(env.get_allowed_actions(0))
    assert build_action(env, 0, legal, "pink") is None


def test_build_three_on_a_fresh_set():
    game = SharedGame.new(74, max_rounds=20)
    env = game.env
    pink = COLOR_GROUPS["pink"]
    for square in pink:
        _give(env, 0, square)
    env._update_monopolies()
    env.players[0].cash = 1500
    for opp in (1, 2, 3):
        env.players[opp].position = 5
        env.players[opp].in_jail = False
    _seat(env, 0)
    legal = list(env.get_allowed_actions(0))
    action = build_action(env, 0, legal, "pink")
    assert action is not None
    assert OFFSETS["improve_house"] <= action < OFFSETS["improve_hotel"]


def test_debt_mortgages_before_selling_houses():
    from monopoly_game_engine.constants import PROPERTY_IDS

    game = SharedGame.new(75, max_rounds=20)
    env = game.env
    brown = COLOR_GROUPS["brown"][0]
    _give(env, 0, brown)
    mort = OFFSETS["mortgage"] + PROPERTY_IDS.index(brown)
    sell_h = OFFSETS["sell_house"]
    legal = [sell_h, mort, int(ActionType.DECLARE_BANKRUPT)]
    assert debt_action(env, 0, legal, "orange") == mort


def test_hijack_exchanges_orange_for_completing_pink():
    game = SharedGame.new(76, max_rounds=20)
    env = game.env
    pink = COLOR_GROUPS["pink"]
    orange = COLOR_GROUPS["orange"]
    _give(env, 0, pink[0])
    _give(env, 0, pink[1])
    _give(env, 0, orange[0])
    _give(env, 1, pink[2])
    _give(env, 1, orange[1])
    _give(env, 1, orange[2])
    env.players[0].cash = 1000
    env.players[1].cash = 200
    _seat(env, 0)
    legal = list(env.get_allowed_actions(0))
    action = trade_action(env, 0, legal, "pink")
    assert action is not None
    assert action >= OFFSETS["exch_trade"]


def test_incoming_accepts_hijack_and_declines_funded_finish():
    game = SharedGame.new(77, max_rounds=20)
    env = game.env
    pink = COLOR_GROUPS["pink"]
    orange = COLOR_GROUPS["orange"]
    _give(env, 0, pink[0])
    _give(env, 0, pink[1])
    _give(env, 0, orange[0])
    _give(env, 1, pink[2])
    _give(env, 1, orange[1])
    _give(env, 1, orange[2])
    env.players[0].cash = 1000
    env.players[1].cash = 200
    env.pending_trades[1] = TradeOffer(
        1,
        0,
        offered_prop=env.properties[pink[2]],
        requested_prop=env.properties[orange[0]],
    )
    legal = [int(ActionType.ACCEPT_TRADE), int(ActionType.DECLINE_TRADE)]
    assert incoming_action(env, 0, legal, "pink") == int(ActionType.ACCEPT_TRADE)

    brown = COLOR_GROUPS["brown"]
    game = SharedGame.new(78, max_rounds=20)
    env = game.env
    _give(env, 1, brown[0])
    _give(env, 0, brown[1])
    env.players[1].cash = 2000
    env.pending_trades[1] = TradeOffer(
        1, 0, requested_prop=env.properties[brown[1]]
    )
    assert incoming_action(env, 0, legal, "orange") == int(ActionType.DECLINE_TRADE)


def test_incoming_railroad_complete_does_not_crash():
    from monopoly_game_engine.constants import RAILROAD_IDS

    game = SharedGame.new(79, max_rounds=20)
    env = game.env
    for square in RAILROAD_IDS[:3]:
        _give(env, 0, square)
    _give(env, 1, RAILROAD_IDS[3])
    env.pending_trades[1] = TradeOffer(
        1, 0, offered_prop=env.properties[RAILROAD_IDS[3]]
    )
    legal = [int(ActionType.ACCEPT_TRADE), int(ActionType.DECLINE_TRADE)]
    action = incoming_action(env, 0, legal, "orange")
    assert action in legal
