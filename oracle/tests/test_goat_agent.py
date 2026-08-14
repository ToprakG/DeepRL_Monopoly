"""toprakthegoat-v1: jail-v1 body plus race contest."""

from __future__ import annotations

from monopoly_bench.engine import SharedGame, clone_env
from monopoly_game_engine.actions import OFFSETS, ActionType
from monopoly_game_engine.constants import COLOR_GROUPS, PROPERTIES
from monopoly_game_engine.env import TradeOffer
from oracle.eval_h2h import GOAT_FIELD_LINEUP, GOAT_ID, _is_oracle_policy, _parse_lineup
from oracle.jail.loop import buy_action as jail_buy
from oracle.plus_steals import complete_floor
from toprakthegoat import GoatAgent
from toprakthegoat.loop import (
    BUILD_TARGET,
    DENY_FRAC,
    auction_ceiling,
    build_action,
    buy_action,
    deed_role,
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


def test_goat_id_is_oracle_policy():
    assert _is_oracle_policy(GOAT_ID)
    assert GOAT_FIELD_LINEUP[0] == GOAT_ID
    assert len(_parse_lineup(",".join(GOAT_FIELD_LINEUP))) == 4


def test_goat_agent_legal_and_non_mutating():
    game = SharedGame.new(80, max_rounds=20)
    before = clone_env(game.env)
    actor = game.env.whose_turn()
    action = GoatAgent(actor, seed=0).choose_action(game.env)
    assert action in before.get_allowed_actions(actor)
    assert game.env.round == before.round


def test_buys_open_brown_when_plan_is_elsewhere():
    game = SharedGame.new(81, max_rounds=20)
    env = game.env
    env.players[0].cash = 1500
    env.players[0].position = COLOR_GROUPS["brown"][0]
    legal = [int(ActionType.BUY_PROPERTY), int(ActionType.END_TURN)]
    assert buy_action(env, 0, legal, "orange") == int(ActionType.BUY_PROPERTY)
    assert jail_buy(env, 0, legal, "orange") is None


def test_contests_lightblue_when_opponent_has_one():
    game = SharedGame.new(82, max_rounds=20)
    env = game.env
    lightblue = COLOR_GROUPS["lightblue"]
    _give(env, 1, lightblue[0])
    env.players[0].cash = 1500
    env.players[0].position = lightblue[1]
    legal = [int(ActionType.BUY_PROPERTY), int(ActionType.END_TURN)]
    assert deed_role(env, 0, lightblue[1], "orange") == "contest"
    assert buy_action(env, 0, legal, "orange") == int(ActionType.BUY_PROPERTY)
    assert jail_buy(env, 0, legal, "orange") is None


def test_does_not_cash_buy_open_green():
    game = SharedGame.new(83, max_rounds=20)
    env = game.env
    env.players[0].cash = 1500
    env.players[0].position = COLOR_GROUPS["green"][0]
    legal = [int(ActionType.BUY_PROPERTY), int(ActionType.END_TURN)]
    assert buy_action(env, 0, legal, "orange") is None


def test_auction_brown_is_list_not_scrap():
    game = SharedGame.new(84, max_rounds=20)
    env = game.env
    env.players[0].cash = 1500
    brown = COLOR_GROUPS["brown"][0]
    ceiling = auction_ceiling(env, 0, brown, "orange")
    assert ceiling == float(PROPERTIES[brown]["price"])
    green = COLOR_GROUPS["green"][0]
    assert auction_ceiling(env, 0, green, "orange") == 0.5 * float(
        PROPERTIES[green]["price"]
    )


def test_builds_fourth_house_not_hotel():
    game = SharedGame.new(85, max_rounds=20)
    env = game.env
    pink = COLOR_GROUPS["pink"]
    for square in pink:
        _give(env, 0, square)
        env.properties[square].houses = 3
    env._update_monopolies()
    env.houses_available = 8
    env.hotels_available = 12
    env.players[0].cash = 1500
    env.players[1].position = 0
    env.players[2].position = 0
    env.players[3].position = 0
    _seat(env, 0)
    legal = list(env.get_allowed_actions(0))
    action = build_action(env, 0, legal, "pink")
    assert action is not None
    assert OFFSETS["improve_house"] <= action < OFFSETS["improve_hotel"]


def test_build_stops_at_four_and_never_hotels():
    game = SharedGame.new(85, max_rounds=20)
    env = game.env
    pink = COLOR_GROUPS["pink"]
    for square in pink:
        _give(env, 0, square)
        env.properties[square].houses = BUILD_TARGET
    env._update_monopolies()
    env.houses_available = 2
    env.hotels_available = 12
    env.players[0].cash = 1500
    env.players[1].position = 0
    env.players[2].position = 0
    env.players[3].position = 0
    _seat(env, 0)
    legal = list(env.get_allowed_actions(0))
    assert any(
        OFFSETS["improve_hotel"] <= a < OFFSETS["sell_house"] for a in legal
    )
    assert build_action(env, 0, legal, "pink") is None


def test_build_three_on_a_fresh_set():
    game = SharedGame.new(86, max_rounds=20)
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


def test_hijack_exchanges_orange_for_completing_pink():
    game = SharedGame.new(87, max_rounds=20)
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
    game = SharedGame.new(88, max_rounds=20)
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
    game = SharedGame.new(89, max_rounds=20)
    env = game.env
    _give(env, 1, brown[0])
    _give(env, 0, brown[1])
    env.players[1].cash = 2000
    env.pending_trades[1] = TradeOffer(
        1, 0, requested_prop=env.properties[brown[1]]
    )
    assert incoming_action(env, 0, legal, "orange") == int(ActionType.DECLINE_TRADE)


def test_after_brown_set_does_not_plan_darkblue():
    from toprakthegoat.loop import active_colour, colour_score

    game = SharedGame.new(93, max_rounds=20)
    env = game.env
    for square in COLOR_GROUPS["brown"]:
        _give(env, 0, square)
    for color in ("lightblue", "pink", "orange", "red", "yellow", "green"):
        _give(env, 1, COLOR_GROUPS[color][0])
    env._update_monopolies()
    assert colour_score(env, 0, "darkblue") == 0.0
    assert active_colour(env, 0) != "darkblue"


def test_open_darkblue_scores_before_a_set():
    from toprakthegoat.loop import colour_score

    game = SharedGame.new(96, max_rounds=20)
    env = game.env
    assert colour_score(env, 0, "darkblue") > 0.0


def test_after_brown_declines_cash_for_orange_blocker():
    game = SharedGame.new(94, max_rounds=20)
    env = game.env
    brown = COLOR_GROUPS["brown"]
    orange = COLOR_GROUPS["orange"]
    for square in brown:
        _give(env, 0, square)
    _give(env, 0, orange[0])
    _give(env, 1, orange[1])
    _give(env, 1, orange[2])
    env._update_monopolies()
    env.players[0].cash = 1000
    env.players[1].cash = 200
    env.pending_trades[1] = TradeOffer(
        1, 0, requested_prop=env.properties[orange[0]], cash_offered=225
    )
    legal = [int(ActionType.ACCEPT_TRADE), int(ActionType.DECLINE_TRADE)]
    assert incoming_action(env, 0, legal, "orange") == int(ActionType.DECLINE_TRADE)


def test_after_brown_does_not_sell_completing_pink():
    game = SharedGame.new(95, max_rounds=20)
    env = game.env
    brown = COLOR_GROUPS["brown"]
    pink = COLOR_GROUPS["pink"]
    for square in brown:
        _give(env, 0, square)
        env.properties[square].houses = 4
    _give(env, 0, pink[0])
    _give(env, 1, pink[1])
    _give(env, 1, pink[2])
    env._update_monopolies()
    env.players[0].cash = 1000
    env.players[1].cash = 2000
    _seat(env, 0)
    legal = list(env.get_allowed_actions(0))
    action = trade_action(env, 0, legal, "orange")
    assert action is None or action < OFFSETS["sell_trade"] or action >= OFFSETS["exch_trade"]


def test_one_away_orange_auctions_through_list():
    game = SharedGame.new(90, max_rounds=20)
    env = game.env
    orange = COLOR_GROUPS["orange"]
    _give(env, 1, orange[0])
    _give(env, 1, orange[1])
    env.players[0].cash = 1200
    leftover = 1200.0 - complete_floor(env, 0)
    ceiling = auction_ceiling(env, 0, orange[2], "lightblue")
    assert ceiling == leftover
    assert ceiling > DENY_FRAC * float(PROPERTIES[orange[2]]["price"])


def test_one_away_brown_stays_at_deny_frac():
    game = SharedGame.new(91, max_rounds=20)
    env = game.env
    brown = COLOR_GROUPS["brown"]
    _give(env, 1, brown[0])
    env.players[0].cash = 1200
    price = float(PROPERTIES[brown[1]]["price"])
    leftover = 1200.0 - complete_floor(env, 0)
    ceiling = auction_ceiling(env, 0, brown[1], "orange")
    assert ceiling == min(DENY_FRAC * price, leftover)


def test_open_orange_auction_stays_at_list():
    game = SharedGame.new(92, max_rounds=20)
    env = game.env
    orange = COLOR_GROUPS["orange"][0]
    env.players[0].cash = 1200
    assert auction_ceiling(env, 0, orange, "lightblue") == float(PROPERTIES[orange]["price"])
