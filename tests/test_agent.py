"""Tournament contract for repo-root ``agent.py``."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from toprakthegoat.bootstrap import install_engine_namespace

ROOT = Path(__file__).resolve().parents[1]
install_engine_namespace(ROOT)

from monopoly_game_engine.actions import ActionType
from monopoly_game_engine.constants import COLOR_GROUPS, NUM_PLAYERS
from monopoly_game_engine.env import PHASE_AUCTION, MonopolyEnv, TradeOffer
from monopoly_game_engine.state import STATE_DIM, build_state_vector

from toprakthegoat.agent import heuristic_action
from toprakthegoat.hydrate import hydrate_env


def _give(env: MonopolyEnv, pid: int, square: int) -> None:
    prop = env.properties[square]
    prop.owner = pid
    if prop not in env.players[pid].properties:
        env.players[pid].properties.append(prop)


def _seat(env: MonopolyEnv, pid: int) -> None:
    env.current_turn_idx = env.turn_order.index(pid)


def _rich_env() -> MonopolyEnv:
    env = MonopolyEnv(agent_ids=[0], max_rounds=200)
    env.turn_order = [0, 1, 2, 3]
    _seat(env, 0)
    env.players[0].cash = 1500
    env.players[0].position = 19
    env.players[1].cash = 800
    env.players[1].in_jail = True
    _give(env, 0, COLOR_GROUPS["brown"][0])
    _give(env, 0, COLOR_GROUPS["brown"][1])
    _give(env, 1, COLOR_GROUPS["orange"][0])
    env.properties[COLOR_GROUPS["brown"][0]].houses = 4
    env.properties[COLOR_GROUPS["brown"][1]].houses = 4
    env._update_monopolies()
    env.houses_available = 24
    env.round = 12
    return env


def test_hydrate_roundtrip_board():
    env = _rich_env()
    vector = build_state_vector(env.players, env.properties, 0, env)
    assert len(vector) == STATE_DIM
    shadow = hydrate_env(vector, 0, [int(ActionType.END_TURN)])
    assert shadow.players[0].cash == 1500
    assert shadow.players[0].position == 19
    assert shadow.players[1].in_jail
    assert shadow.properties[COLOR_GROUPS["brown"][0]].owner == 0
    assert shadow.properties[COLOR_GROUPS["brown"][0]].houses == 4
    assert shadow.properties[COLOR_GROUPS["orange"][0]].owner == 1
    assert shadow.properties[COLOR_GROUPS["brown"][0]].is_monopoly
    assert shadow.round == 12
    assert shadow.houses_available == 24


def test_hydrate_recovers_clipped_cash_from_board():
    env = _rich_env()
    env.players[0].cash = 7200
    vector = build_state_vector(env.players, env.properties, 0, env)
    assert float(vector[1]) >= 0.999
    board = {"players": [{"player_id": 0, "cash": 7200}], "round": 12}
    shadow = hydrate_env(vector, 0, [int(ActionType.END_TURN)], board)
    assert shadow.players[0].cash == 7200
    assert shadow.round == 12


def test_hydrate_auction_and_trade():
    env = _rich_env()
    env.phase = PHASE_AUCTION
    env.auction_property_id = COLOR_GROUPS["lightblue"][0]
    env.auction_high_bid = 100
    env.auction_high_bidder = 2
    env.auction_bidders = [0, 1, 2]
    env.auction_current_pid = 0
    offered = env.properties[COLOR_GROUPS["orange"][0]]
    env.pending_trades[1] = TradeOffer(1, 0, offered, None, 50, 0)
    vector = build_state_vector(env.players, env.properties, 0, env)
    shadow = hydrate_env(vector, 0, [80, 81, 82, 83, 84])
    assert shadow.phase == PHASE_AUCTION
    assert shadow.auction_property_id == COLOR_GROUPS["lightblue"][0]
    assert shadow.auction_high_bid == 100
    assert shadow.auction_high_bidder == 2
    incoming = shadow._incoming_trade(0)
    assert incoming is not None
    assert incoming.from_player == 1
    assert incoming.offered_prop.square_id == COLOR_GROUPS["orange"][0]


def test_tournament_choose_action_matches_heuristic():
    import agent as submit

    env = _rich_env()
    env.phase = "post_roll"
    env.has_rolled = True
    env.players[0].position = COLOR_GROUPS["lightblue"][1]
    _seat(env, 0)
    legal = [int(ActionType.BUY_PROPERTY), int(ActionType.END_TURN)]
    vector = build_state_vector(env.players, env.properties, 0, env)
    state = SimpleNamespace(
        vector=vector,
        board={
            "players": [
                {"player_id": i, "cash": env.players[i].cash} for i in range(NUM_PLAYERS)
            ]
        },
        actions=legal,
        decision_seed=7,
        ruleset_version="ppo-plus-v2",
        schema_version=1,
    )
    expected = heuristic_action(env, 0, legal)
    got = submit.choose_action(state, 0, legal)
    assert got == expected
    assert got in legal


def test_numpy_action_ids_and_seat_are_accepted():
    import numpy as np
    import agent as submit

    env = _rich_env()
    env.phase = "post_roll"
    env.has_rolled = True
    env.players[0].position = COLOR_GROUPS["lightblue"][1]
    _seat(env, 0)
    legal = [int(ActionType.BUY_PROPERTY), int(ActionType.END_TURN)]
    vector = build_state_vector(env.players, env.properties, 0, env)
    state = SimpleNamespace(vector=vector, board=None, actions=legal)
    got = submit.choose_action(state, np.int64(0), np.array(legal, dtype=np.int64))
    assert got in legal
    assert submit._as_int_list(np.array(legal, dtype=np.int64)) == legal


def test_tournament_never_raises_on_garbage():
    import agent as submit

    legal = [int(ActionType.END_TURN), int(ActionType.ROLL_DICE)]
    assert submit.choose_action(None, 0, legal) in legal
    assert submit.choose_action({}, 3, legal) in legal
    assert submit.choose_action(SimpleNamespace(), 0, legal) in legal
    assert submit.choose_action(state=None, player_id=1, allowed_actions=legal) in legal
    assert submit.choose_action("broken", 0, []) == int(ActionType.END_TURN)


def test_submit_import_does_not_pull_torch():
    script = (
        "import sys; import agent; "
        "assert 'torch' not in sys.modules; "
        "assert 'monopoly_bench' not in sys.modules; "
        "assert 'oracle' not in sys.modules"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    subprocess.check_call(
        [sys.executable, "-c", script],
        cwd=str(ROOT),
        env=env,
    )
