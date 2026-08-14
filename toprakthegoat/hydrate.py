"""Rebuild a MonopolyEnv from the tournament's 300-float observation.

The live engine is not handed to submitted agents. ``build_state_vector`` in
``monopoly_game_engine.state`` is the layout we invert. Cash above $5,000 is
clipped in the vector; ``board`` is read only to recover clipped cash and the
exact round.
"""

from __future__ import annotations

from typing import Any, Sequence
from numbers import Integral, Real

from monopoly_game_engine.actions import ActionType
from monopoly_game_engine.constants import MAX_JAIL_TURNS, NUM_PLAYERS, PROPERTY_IDS
from monopoly_game_engine.env import (
    PHASE_AUCTION,
    PHASE_OUT_OF_TURN,
    PHASE_POST_ROLL,
    PHASE_PRE_ROLL,
    MonopolyEnv,
    TradeOffer,
)
from monopoly_game_engine.state import BASE_STATE_DIM, PHASES, STATE_DIM

CASH_SCALE = 5000.0
MONEY_SCALE = 2000.0
DEFAULT_MAX_ROUNDS = 200
N_PROP = len(PROPERTY_IDS)

_idx = 0
V_PLAYERS = _idx
_idx += NUM_PLAYERS * 4
V_PROPS = _idx
_idx += N_PROP * 8
assert _idx == BASE_STATE_DIM
V_PHASE = _idx
_idx += len(PHASES)
V_WHOSE = _idx
_idx += NUM_PLAYERS
V_ACTIVE = _idx
_idx += NUM_PLAYERS
V_HAS_ROLLED = _idx
_idx += 1
V_DOUBLES = _idx
_idx += 1
V_DICE = _idx
_idx += 2
V_HOUSES = _idx
V_HOTELS = _idx + 1
_idx += 2
V_BANKRUPT = _idx
_idx += NUM_PLAYERS
V_JAIL_TURNS = _idx
_idx += NUM_PLAYERS
V_TURN_ORDER = _idx
_idx += NUM_PLAYERS
V_DEBT_AMOUNT = _idx
_idx += 1
V_DEBT_CREDITOR = _idx
_idx += NUM_PLAYERS + 1
V_AUCTION_PROP = _idx
_idx += 1
V_AUCTION_BID = _idx
_idx += 1
V_ROUND = _idx
_idx += 1
V_AUCTION_LEADER = _idx
_idx += NUM_PLAYERS + 1
V_AUCTION_BIDDERS = _idx
_idx += NUM_PLAYERS
V_EXTRA_ROLL = _idx
_idx += 1
V_TRADE_SENDER = _idx
_idx += NUM_PLAYERS + 1
V_TRADE_OFFERED = _idx
_idx += 1
V_TRADE_REQUESTED = _idx
_idx += 1
V_TRADE_CASH_OFFERED = _idx
V_TRADE_CASH_REQUESTED = _idx + 1
_idx += 2
V_OUTGOING_TO = _idx
_idx += 1
V_PENDING_COUNT = _idx
_idx += 1
assert _idx == STATE_DIM
del _idx

_PHASE_NAMES = tuple(PHASES)
_CASH_KEYS = ("cash", "money", "balance", "funds")
_PLAYERS_KEYS = ("players", "seats", "player_states", "playerStates")
_ROUND_KEYS = ("round", "turn", "round_index", "roundIndex")
_MAX_ROUND_KEYS = ("max_rounds", "maxRounds", "round_limit", "roundLimit")


def _one_hot(vector: Sequence[float], start: int, width: int) -> int | None:
    for i in range(width):
        if float(vector[start + i]) > 0.5:
            return i
    return None


def _field(obj: Any, names: Sequence[str]) -> Any:
    if obj is None:
        return None
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _deed_from_slot(env: MonopolyEnv, slot: float):
    if float(slot) <= 0:
        return None
    index = int(round(float(slot) * (N_PROP + 1))) - 1
    if 0 <= index < N_PROP:
        return env.properties[PROPERTY_IDS[index]]
    return None


def _rel_seat(slot: float, order: list[int]) -> int | None:
    if float(slot) <= 0:
        return None
    relative = int(round(float(slot) * (NUM_PLAYERS + 1))) - 1
    if 0 <= relative < NUM_PLAYERS:
        return order[relative]
    return None


def enrich_from_board(env: MonopolyEnv, board: Any) -> None:
    """Fill values the vector clips, mainly cash above $5,000 and the round."""

    if board is None:
        return
    try:
        rounds = _field(board, _MAX_ROUND_KEYS)
        if isinstance(rounds, Real) and not isinstance(rounds, bool) and rounds > 0:
            env.max_rounds = int(rounds)
        current = _field(board, _ROUND_KEYS)
        if isinstance(current, Real) and not isinstance(current, bool):
            env.round = int(current)
        seats = _field(board, _PLAYERS_KEYS)
        if not isinstance(seats, (list, tuple)):
            return
        for entry in seats:
            seat_id = _field(entry, ("player_id", "playerId", "id", "seat", "index"))
            if not isinstance(seat_id, Real) or isinstance(seat_id, bool):
                continue
            seat_id = int(seat_id)
            if not 0 <= seat_id < NUM_PLAYERS:
                continue
            cash = _field(entry, _CASH_KEYS)
            if isinstance(cash, Real) and not isinstance(cash, bool):
                env.players[seat_id].cash = float(cash)
            position = _field(entry, ("position", "pos", "square", "location"))
            if isinstance(position, Real) and not isinstance(position, bool):
                env.players[seat_id].position = int(position)
    except Exception:
        return


def hydrate_env(
    vector: Sequence[float],
    player_id: int,
    allowed: Sequence[int],
    board: Any = None,
    *,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
) -> MonopolyEnv:
    """Invert ``build_state_vector`` into a duck-typed engine snapshot."""

    env = MonopolyEnv(agent_ids=[int(player_id)], max_rounds=max_rounds)
    acting = int(player_id)
    order = [acting] + [i for i in range(NUM_PLAYERS) if i != acting]
    allowed_ids = [int(a) for a in allowed]

    for relative, seat_id in enumerate(order):
        base = V_PLAYERS + relative * 4
        seat = env.players[seat_id]
        seat.position = int(round(float(vector[base]) * 39.0))
        seat.cash = int(round(float(vector[base + 1]) * CASH_SCALE))
        seat.in_jail = float(vector[base + 2]) > 0.5
        seat.gooj_card = float(vector[base + 3]) > 0.5
        seat.bankrupt = float(vector[V_BANKRUPT + relative]) > 0.5
        seat.jail_turns = int(
            round(float(vector[V_JAIL_TURNS + relative]) * max(MAX_JAIL_TURNS, 1))
        )
        seat.properties = []

    for i, square in enumerate(PROPERTY_IDS):
        base = V_PROPS + i * 8
        prop = env.properties[square]
        owner = _one_hot(vector, base, NUM_PLAYERS)
        prop.owner = owner
        prop.mortgaged = float(vector[base + 5]) > 0.5
        prop.houses = int(round(float(vector[base + 7]) * 5.0))
        if owner is not None:
            env.players[owner].properties.append(prop)

    env._update_monopolies()

    phase = _one_hot(vector, V_PHASE, len(_PHASE_NAMES))
    env.phase = _PHASE_NAMES[phase] if phase is not None else PHASE_PRE_ROLL
    whose = _one_hot(vector, V_WHOSE, NUM_PLAYERS)
    if whose is not None:
        acting = order[whose]
    env.has_rolled = float(vector[V_HAS_ROLLED]) > 0.5
    env.consecutive_doubles = int(round(float(vector[V_DOUBLES]) * 3.0))
    env.last_dice = (
        int(round(float(vector[V_DICE]) * 6.0)),
        int(round(float(vector[V_DICE + 1]) * 6.0)),
    )
    env.houses_available = int(round(float(vector[V_HOUSES]) * 32.0))
    env.hotels_available = int(round(float(vector[V_HOTELS]) * 12.0))
    env.round = int(round(float(vector[V_ROUND]) * max(env.max_rounds, 1)))
    env.extra_roll_pending = float(vector[V_EXTRA_ROLL]) > 0.5

    turn_order = []
    for slot in range(NUM_PLAYERS):
        relative = int(round(float(vector[V_TURN_ORDER + slot]) * (NUM_PLAYERS - 1)))
        turn_order.append(order[min(max(relative, 0), NUM_PLAYERS - 1)])
    if sorted(turn_order) == list(range(NUM_PLAYERS)):
        env.turn_order = turn_order
    if acting in env.turn_order:
        env.current_turn_idx = env.turn_order.index(acting)

    env.debt_amount = float(vector[V_DEBT_AMOUNT]) * MONEY_SCALE
    creditor_slot = _one_hot(vector, V_DEBT_CREDITOR, NUM_PLAYERS + 1)
    if creditor_slot is not None and creditor_slot > 0:
        env.debt_creditor = order[creditor_slot - 1]
    env.debt_player = None
    env.player_needs_funds = None
    end_turn = int(ActionType.END_TURN)
    roll_dice = int(ActionType.ROLL_DICE)
    if (
        env.debt_amount > 0
        and env.phase == PHASE_POST_ROLL
        and end_turn not in allowed_ids
        and roll_dice not in allowed_ids
    ):
        env.debt_player = acting
        env.player_needs_funds = acting

    auction_slot = float(vector[V_AUCTION_PROP])
    if auction_slot > 0:
        index = int(round(auction_slot * (N_PROP + 1))) - 1
        if 0 <= index < N_PROP:
            env.auction_property_id = PROPERTY_IDS[index]
    env.auction_high_bid = int(round(float(vector[V_AUCTION_BID]) * MONEY_SCALE))
    leader_slot = _one_hot(vector, V_AUCTION_LEADER, NUM_PLAYERS + 1)
    if leader_slot is not None and leader_slot > 0:
        env.auction_high_bidder = order[leader_slot - 1]
    env.auction_bidders = [
        order[i]
        for i in range(NUM_PLAYERS)
        if float(vector[V_AUCTION_BIDDERS + i]) > 0.5
    ]
    env.auction_current_pid = acting if env.phase == PHASE_AUCTION else None
    if env.phase == PHASE_OUT_OF_TURN:
        env.out_of_turn_pids = [acting]

    env.pending_trades = {}
    sender_slot = _one_hot(vector, V_TRADE_SENDER, NUM_PLAYERS + 1)
    if sender_slot is not None and sender_slot > 0:
        sender = order[sender_slot - 1]
        env.pending_trades[sender] = TradeOffer(
            sender,
            acting,
            _deed_from_slot(env, float(vector[V_TRADE_OFFERED])),
            _deed_from_slot(env, float(vector[V_TRADE_REQUESTED])),
            float(vector[V_TRADE_CASH_OFFERED]) * MONEY_SCALE,
            float(vector[V_TRADE_CASH_REQUESTED]) * MONEY_SCALE,
        )
    outgoing_to = _rel_seat(float(vector[V_OUTGOING_TO]), order)
    if outgoing_to is not None and acting not in env.pending_trades:
        env.pending_trades[acting] = TradeOffer(acting, outgoing_to)

    env.get_allowed_actions = lambda _pid=None, ids=allowed_ids: list(ids)
    enrich_from_board(env, board)
    return env
