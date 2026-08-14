"""Seat-invariant observations -- the frame our policies are trained in.

WHY THIS FILE EXISTS
====================
The engine's 300-float observation (`engine/state.py:build_state_vector`) is
NOT seat-invariant.  Two independent bugs break it, and fixing only one leaves
it broken:

  (a) the 28x5 deed-owner one-hot is written at the owner's ABSOLUTE seat index
      (engine/state.py:152-155).  "Seat 2 owns Boardwalk" lights the same bit no
      matter who is looking.

  (b) opponents are ordered `[me] + [other seats ASCENDING BY ABSOLUTE ID]`
      (engine/state.py:139 and :170; mirrored at monopoly_bench/engine.py:85-88).
      That is a permutation, but not a meaningful one: from seat 0 the player in
      relative slot 1 is the one who moves next, while from seat 1 relative slot
      1 is seat 0 -- who moves LAST.  The network cannot tell "the player about
      to act" from "the player who just acted".

MEASURED (tools below, reproduced in tests/test_obs.py): take one real mid-game
position, relabel the seats (a pure renaming -- the game is identical), and ask
each seat for its observation.  Upstream changes up to 58 of 300 dims.  Even a
relabelling that leaves the ACTOR's seat id untouched changes 36 dims, because
of bug (a).  After the fix in this file: 0 dims, at every seat, in every
position tested.

One network therefore cannot play four seats against the unmodified engine.
This module is the fix, and it is deliberately a WRAPPER: `engine/` stays a
byte-pinned vendored copy (see engine/README.md) and is never forked.

THE FRAME
=========
Our frame orders players by TURN ORDER RELATIVE TO THE ACTOR:

    order[0] = me
    order[1] = the player who acts after me
    order[2] = two seats after me
    order[3] = three seats after me   (= the player who acts before me)

taken from `env.turn_order`, which the engine SHUFFLES at reset
(engine/env.py:101-102) -- so `(player_id + k) % 4` is NOT the same thing and is
only used as a fallback when no turn order is available.

Every player-indexed field -- deed owners included -- is stored at its relative
slot.  The board itself (which square is which property) is absolute, because
the board does not move.

TWO WAYS IN
===========
    build_obs(game, player_id)          # build our frame straight from the game
    to_our_frame(canonical, player_id)  # convert an observation the ENGINE built

They agree exactly (asserted in tests/test_obs.py).  `to_our_frame` is what
makes a policy trained here tournament-portable: at play time the organizer runs
their UNMODIFIED engine, hands us a canonical 300-vector, and we permute it into
the frame the weights expect.  `to_canonical` is the inverse.

ACTIONS NEED THE SAME TREATMENT
===============================
The trade actions encode their target as an index into
`[i for i in range(4) if i != pid]` -- ascending absolute again
(engine/env.py:851-854 and :882-885).  A policy that observes opponents in
turn-relative order must also EMIT trade targets in turn-relative order, or it
proposes deals to the wrong player.  `action_to_canonical` /
`action_to_our_frame` / `legal_mask_our_frame` handle that.

FULL INDEX MAP OF THE 300-VECTOR
================================
Identical layout to upstream; only the contents of the REL-marked fields move.
`r` below is a RELATIVE player slot 0..3 (0 = me).  "REL" = permuted by this
module; "abs" = board- or game-global, untouched.

  idx    size  content
  ----   ----  -------------------------------------------------------------
    0     16   player block, 4 slots x 4 features, stride 4          REL
               +0 position/39   +1 min(cash/5000,1)
               +2 in_jail       +3 has get-out-of-jail-free card
               slot r starts at 4*r
   16    224   deed block, 28 properties x 8 features, stride 8
               property k (k = index into constants.PROPERTY_IDS) at 16+8k:
               +0..+3  owner one-hot by RELATIVE slot                REL
                       (upstream indexes this one by ABSOLUTE SEAT -- it is
                       the only player-indexed block that does, and that is
                       bug (a).  Every other block below is relative in both
                       frames and only the ORDER differs.)
               +4      unused -- upstream sizes the one-hot 5 but only
                       ever writes 0..3; all-zero means the bank owns it
               +5      mortgaged                                     abs
               +6      is_monopoly (owner holds the full colour group)
               +7      houses/5 (0 for railroads and utilities; 5 = hotel)
  240      4   phase one-hot: pre_roll, post_roll, out_of_turn, auction
  244      4   whose_turn() one-hot                                  REL
  248      4   active_player_id() one-hot                            REL
  252      1   has_rolled
  253      1   min(consecutive_doubles/3, 1)
  254      2   last_dice[0]/6, last_dice[1]/6
  256      1   houses_available/32
  257      1   hotels_available/12
  258      4   bankrupt flag per player                              REL
  262      4   jail_turns/MAX_JAIL_TURNS per player                  REL
  266      4   turn-order map.  CONSTANT [0, 1/3, 2/3, 1] in our frame
               -- see "the dead block" below.
  270      1   min(debt_amount/2000, 1)
  271      5   debt creditor: +0 = nobody, +1+r = relative player r   REL
  276      1   auction property: 0 = none, else (1+PROPERTY_IDS.index)/29
  277      1   min(auction_high_bid/2000, 1)
  278      1   min(round/max_rounds, 1)
  279      5   auction high bidder: +0 = nobody, +1+r                 REL
  284      4   auction bidders, multi-hot                             REL
  288      1   extra_roll_pending
  289      5   incoming trade sender: +0 = nobody, +1+r               REL
  294      1   incoming offer: property offered TO me, encoded as 276
  295      1   incoming offer: property requested FROM me, same encoding
  296      2   incoming cash_offered/2000, cash_requested/2000
  298      1   my outgoing offer's recipient, (1+r)/5; 0.0 = no offer  REL
               (scalar-coded, not one-hot -- recoded, not permuted)
  299      1   min(len(env.pending_trades)/4, 1)
  ----   ----
  total  300

THE DEAD BLOCK (266..269)
=========================
Upstream stores, at ABSOLUTE turn slot t, the relative id of the player in that
slot.  Once players are ordered by turn order relative to the actor, that map is
`[0, 1/3, 2/3, 1]` by construction -- constant, and therefore carries no
information.  That is not a bug: the information it used to carry was precisely
the seat-identity information that has to be destroyed for seat invariance.  It
is kept as four constant dims so the layout stays byte-compatible with the
engine's and so `to_our_frame` / `to_canonical` remain exact inverses.

Consequence: `to_canonical` cannot recover `env.turn_order` from our vector
(nothing invariant could).  Pass it:

    canonical = to_canonical(ours, pid, turn_order=env.turn_order)

`to_our_frame` does not need it -- it decodes the turn order out of the
canonical vector's own 266..269 block.

Depends on: mono.types (unwrap, legal_actions).  numpy + stdlib only, no torch.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable, Sequence

import numpy as np

from engine.actions import ACTION_SPACE_SIZE, OFFSETS
from engine.constants import (
    MAX_JAIL_TURNS,
    NUM_PLAYERS,
    PROPERTY_IDS,
    TRADE_CASH_LEVELS,
)
from engine.state import BASE_STATE_DIM, PHASES, STATE_DIM

import engine_shim  # noqa: F401

def unwrap(game):
    return getattr(game, 'env', game)

def legal_actions(env, pid):
    return env.get_allowed_actions(pid)


__all__ = [
    "STATE_DIM",
    "NUM_PLAYERS",
    "ACTION_SPACE_SIZE",
    "PLAYER_BLOCK",
    "DEED_BLOCK",
    "TURN_ORDER_BLOCK",
    "OUTGOING_TRADE_DIM",
    "seat_order",
    "seat_order_from_turn_order",
    "canonical_seat_order",
    "turn_order_of",
    "build_obs",
    "to_our_frame",
    "to_canonical",
    "obs_index_perm",
    "action_to_our_frame",
    "action_to_canonical",
    "action_perm_to_canonical",
    "action_perm_to_our_frame",
    "legal_actions_our_frame",
    "legal_mask_our_frame",
    "relative_of",
    "absolute_of",
]


# --------------------------------------------------------------------------
# Layout constants -- every magic number in this file is named here.
# --------------------------------------------------------------------------

PLAYER_BLOCK = 0            # 4 slots x 4 features
PLAYER_FEATURES = 4
DEED_BLOCK = 16             # 28 properties x 8 features
DEED_STRIDE = 8
OWNER_ONEHOT = 5            # first 4 slots used; 5th is upstream padding
NUM_PROPERTIES = len(PROPERTY_IDS)

TAIL = BASE_STATE_DIM       # 240
PHASE_BLOCK = TAIL                    # 4
WHOSE_TURN_BLOCK = TAIL + 4           # 4   REL
ACTIVE_PLAYER_BLOCK = TAIL + 8        # 4   REL
HAS_ROLLED_DIM = TAIL + 12
DOUBLES_DIM = TAIL + 13
DICE_BLOCK = TAIL + 14                # 2
SUPPLY_BLOCK = TAIL + 16              # 2
BANKRUPT_BLOCK = TAIL + 18            # 4   REL
JAIL_TURNS_BLOCK = TAIL + 22          # 4   REL
TURN_ORDER_BLOCK = TAIL + 26          # 4   REL, constant in our frame (266)
DEBT_AMOUNT_DIM = TAIL + 30
DEBT_CREDITOR_BLOCK = TAIL + 31       # 5   REL, offset 1 (271)
AUCTION_PROP_DIM = TAIL + 36
AUCTION_BID_DIM = TAIL + 37
ROUND_DIM = TAIL + 38
AUCTION_LEADER_BLOCK = TAIL + 39      # 5   REL, offset 1 (279)
AUCTION_BIDDERS_BLOCK = TAIL + 44     # 4   REL (284)
EXTRA_ROLL_DIM = TAIL + 48
TRADE_SENDER_BLOCK = TAIL + 49        # 5   REL, offset 1 (289)
TRADE_OFFERED_DIM = TAIL + 54
TRADE_REQUESTED_DIM = TAIL + 55
TRADE_CASH_BLOCK = TAIL + 56          # 2
OUTGOING_TRADE_DIM = TAIL + 58        # REL, scalar-coded (298)
PENDING_TRADES_DIM = TAIL + 59

# 4-wide blocks indexed directly by relative player slot.
_REL4_BLOCKS = (
    WHOSE_TURN_BLOCK,
    ACTIVE_PLAYER_BLOCK,
    BANKRUPT_BLOCK,
    JAIL_TURNS_BLOCK,
    AUCTION_BIDDERS_BLOCK,
)
# 5-wide blocks whose slot 0 means "nobody" and whose slots 1..4 are players.
_REL5_BLOCKS = (
    DEBT_CREDITOR_BLOCK,
    AUCTION_LEADER_BLOCK,
    TRADE_SENDER_BLOCK,
)

_PROP_ENCODE_DIV = float(NUM_PROPERTIES + 1)   # 29
_PLAYER_ENCODE_DIV = float(NUM_PLAYERS + 1)    # 5
_TURN_SLOT_DIV = float(max(NUM_PLAYERS - 1, 1))  # 3

_IDENTITY_TURN_ORDER = tuple(range(NUM_PLAYERS))


# --------------------------------------------------------------------------
# Seat orders
# --------------------------------------------------------------------------


def canonical_seat_order(player_id: int) -> tuple[int, ...]:
    """The engine's own relative->absolute map: `[me] + others ASCENDING`.

    This is bug (b).  It is reproduced faithfully here because it is the frame
    the organizer's engine actually emits, and we have to convert out of it.
    """
    player_id = int(player_id)
    return (player_id,) + tuple(i for i in range(NUM_PLAYERS) if i != player_id)


def seat_order_from_turn_order(
    player_id: int, turn_order: Sequence[int] | None = None
) -> tuple[int, ...]:
    """OUR relative->absolute map: turn order rotated to start at `player_id`.

    `turn_order` is `env.turn_order` -- the engine shuffles it at reset, so it
    is generally NOT `[0, 1, 2, 3]`.  When it is unavailable we fall back to
    `(player_id + k) % NUM_PLAYERS`, which is right only for an unshuffled
    table; every code path that has a game should pass the real thing.
    """
    player_id = int(player_id)
    if turn_order is None:
        turn_order = _IDENTITY_TURN_ORDER
    order = tuple(int(p) for p in turn_order)
    if sorted(order) != list(range(NUM_PLAYERS)):
        raise ValueError(f"turn_order is not a permutation of 0..{NUM_PLAYERS - 1}: {order}")
    k = order.index(player_id)
    return tuple(order[(k + i) % NUM_PLAYERS] for i in range(NUM_PLAYERS))


def turn_order_of(game_or_env: Any) -> tuple[int, ...]:
    """`env.turn_order` as a tuple, falling back to identity if absent."""
    env = unwrap(game_or_env)
    order = getattr(env, "turn_order", None)
    if order is None:
        return _IDENTITY_TURN_ORDER
    return tuple(int(p) for p in order)


def seat_order(game_or_env: Any, player_id: int) -> tuple[int, ...]:
    """OUR relative->absolute seat map for `player_id` in this game."""
    return seat_order_from_turn_order(player_id, turn_order_of(game_or_env))


def relative_of(order: Sequence[int], absolute_seat: int) -> int:
    """Relative slot of an absolute seat, given a relative->absolute `order`."""
    return tuple(order).index(int(absolute_seat))


def absolute_of(order: Sequence[int], relative_slot: int) -> int:
    """Absolute seat of a relative slot."""
    return int(order[int(relative_slot)])


# --------------------------------------------------------------------------
# build_obs -- our frame, straight from the game
# --------------------------------------------------------------------------


def build_obs(game_or_env: Any, player_id: int) -> np.ndarray:
    """The fully actor-relative 300-float observation for `player_id`.

    Mirrors `engine/state.py:build_state_vector` field for field -- same layout,
    same normalisation, same dtype -- and differs in exactly two places:

      * `order` is turn-order-relative instead of absolute-ascending, and
      * the deed-owner one-hot is written at the owner's RELATIVE slot.

    Accepts a `monopoly_bench.SharedGame` or a bare `MonopolyEnv`.  Read-only:
    it never mutates the game.
    """
    env = unwrap(game_or_env)
    players = env.players
    properties = env.properties
    player_id = int(player_id)
    order = seat_order(env, player_id)
    rel = {seat: i for i, seat in enumerate(order)}

    state = np.zeros(STATE_DIM, dtype=np.float32)

    # -- player block (16) ------------------------------------------------
    idx = PLAYER_BLOCK
    for pid in order:
        p = players[pid]
        state[idx] = p.position / 39.0
        state[idx + 1] = min(p.cash / 5000.0, 1.0)
        state[idx + 2] = float(p.in_jail)
        state[idx + 3] = float(p.gooj_card)
        idx += PLAYER_FEATURES

    # -- deed block (224) -------------------------------------------------
    for sq in PROPERTY_IDS:
        prop = properties[sq]
        if prop.owner is not None:
            state[idx + rel[prop.owner]] = 1.0        # <-- RELATIVE, bug (a) fixed
        idx += OWNER_ONEHOT
        state[idx] = float(prop.mortgaged)
        idx += 1
        state[idx] = float(prop.is_monopoly)
        idx += 1
        if prop.is_real_estate:
            state[idx] = prop.houses / 5.0
        idx += 1

    assert idx == BASE_STATE_DIM, f"deed block ended at {idx}, expected {BASE_STATE_DIM}"

    # -- tail (60) --------------------------------------------------------
    if env.phase in PHASES:
        state[idx + PHASES.index(env.phase)] = 1.0
    idx += len(PHASES)

    state[idx + rel[env.whose_turn()]] = 1.0
    idx += NUM_PLAYERS

    state[idx + rel[env.active_player_id()]] = 1.0
    idx += NUM_PLAYERS

    state[idx] = float(bool(env.has_rolled))
    idx += 1
    state[idx] = min(getattr(env, "consecutive_doubles", 0) / 3.0, 1.0)
    idx += 1

    dice = getattr(env, "last_dice", (0, 0))
    state[idx : idx + 2] = [die / 6.0 for die in dice]
    idx += 2

    state[idx] = getattr(env, "houses_available", 0) / 32.0
    state[idx + 1] = getattr(env, "hotels_available", 0) / 12.0
    idx += 2

    for r, pid in enumerate(order):
        state[idx + r] = float(players[pid].bankrupt)
    idx += NUM_PLAYERS

    for r, pid in enumerate(order):
        state[idx + r] = min(players[pid].jail_turns / max(MAX_JAIL_TURNS, 1), 1.0)
    idx += NUM_PLAYERS

    # Turn-order map, now indexed by RELATIVE turn slot.  By construction the
    # player in relative turn slot r IS relative player r, so this is the
    # constant [0, 1/3, 2/3, 1].  Written the long way so the derivation stays
    # visible and so a future ruleset with a non-cyclic turn order still works.
    turn_order = turn_order_of(env)
    my_slot = turn_order.index(player_id)
    for r in range(NUM_PLAYERS):
        state[idx + r] = rel[turn_order[(my_slot + r) % NUM_PLAYERS]] / _TURN_SLOT_DIV
    idx += NUM_PLAYERS

    state[idx] = min(getattr(env, "debt_amount", 0) / 2000.0, 1.0)
    idx += 1

    creditor = getattr(env, "debt_creditor", None)
    state[idx if creditor is None else idx + 1 + rel[creditor]] = 1.0
    idx += NUM_PLAYERS + 1

    auction_property = getattr(env, "auction_property_id", None)
    state[idx] = (
        0.0
        if auction_property is None
        else (1 + PROPERTY_IDS.index(auction_property)) / _PROP_ENCODE_DIV
    )
    idx += 1

    state[idx] = min(getattr(env, "auction_high_bid", 0) / 2000.0, 1.0)
    idx += 1

    max_rounds = max(getattr(env, "max_rounds", 1), 1)
    state[idx] = min(getattr(env, "round", 0) / max_rounds, 1.0)
    idx += 1

    leader = getattr(env, "auction_high_bidder", None)
    state[idx if leader is None else idx + 1 + rel[leader]] = 1.0
    idx += NUM_PLAYERS + 1

    for pid in getattr(env, "auction_bidders", ()):
        state[idx + rel[pid]] = 1.0
    idx += NUM_PLAYERS

    state[idx] = float(bool(env.extra_roll_pending))
    idx += 1

    incoming = env._incoming_trade(player_id)
    sender = None if incoming is None else incoming.from_player
    state[idx if sender is None else idx + 1 + rel[sender]] = 1.0
    idx += NUM_PLAYERS + 1

    state[idx] = (
        0.0
        if incoming is None or incoming.offered_prop is None
        else (1 + PROPERTY_IDS.index(incoming.offered_prop.square_id)) / _PROP_ENCODE_DIV
    )
    idx += 1
    state[idx] = (
        0.0
        if incoming is None or incoming.requested_prop is None
        else (1 + PROPERTY_IDS.index(incoming.requested_prop.square_id)) / _PROP_ENCODE_DIV
    )
    idx += 1
    if incoming is not None:
        state[idx] = min(incoming.cash_offered / 2000.0, 1.0)
        state[idx + 1] = min(incoming.cash_requested / 2000.0, 1.0)
    idx += 2

    outgoing = getattr(env, "pending_trades", {}).get(player_id)
    state[idx] = (
        0.0 if outgoing is None else (1 + rel[outgoing.to_player]) / _PLAYER_ENCODE_DIV
    )
    idx += 1
    state[idx] = min(len(getattr(env, "pending_trades", {})) / NUM_PLAYERS, 1.0)
    idx += 1

    assert idx == STATE_DIM, f"state vector ended at {idx}, expected {STATE_DIM}"
    return state


# --------------------------------------------------------------------------
# The frame adapter: canonical <-> ours
# --------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _rel_maps(player_id: int, turn_order: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """(canonical order, our order, our_rel->canon_rel, canon_rel->our_rel)."""
    canon = canonical_seat_order(player_id)
    ours = seat_order_from_turn_order(player_id, turn_order)
    c_of_r = tuple(canon.index(ours[r]) for r in range(NUM_PLAYERS))
    r_of_c = tuple(ours.index(canon[c]) for c in range(NUM_PLAYERS))
    return canon, ours, c_of_r, r_of_c


@lru_cache(maxsize=64)
def obs_index_perm(player_id: int, turn_order: tuple[int, ...] | None = None) -> np.ndarray:
    """Gather-index array `src` with `ours[i] = canonical[src[i]]`.

    Covers the 296 dims that are a pure permutation.  Dims 266..269
    (`TURN_ORDER_BLOCK`) and 298 (`OUTGOING_TRADE_DIM`) are scalar-CODED rather
    than one-hot, so they are value-recoded instead; `src` leaves them at
    identity and `to_our_frame` / `to_canonical` fix them up.  Use those two
    functions unless you know exactly what you are doing.
    """
    if turn_order is None:
        turn_order = _IDENTITY_TURN_ORDER
    _, our_order, c_of_r, _ = _rel_maps(int(player_id), tuple(turn_order))

    src = np.arange(STATE_DIM, dtype=np.int64)

    for r in range(NUM_PLAYERS):                       # player block
        for f in range(PLAYER_FEATURES):
            src[PLAYER_BLOCK + PLAYER_FEATURES * r + f] = (
                PLAYER_BLOCK + PLAYER_FEATURES * c_of_r[r] + f
            )

    # Deed-owner one-hots.  NOTE the asymmetry, and it is the whole of bug (a):
    # every OTHER player-indexed block in the canonical vector is indexed by
    # CANONICAL RELATIVE slot, but this one is indexed by ABSOLUTE SEAT.  So it
    # gathers through `our_order` (relative -> absolute), not through `c_of_r`
    # (our relative -> canonical relative).  Getting this wrong is silent: the
    # round-trip still closes, it just disagrees with `build_obs`, which is why
    # tests/test_obs.py::test_to_our_frame_matches_build_obs exists.
    for k in range(NUM_PROPERTIES):
        base = DEED_BLOCK + DEED_STRIDE * k
        for r in range(NUM_PLAYERS):
            src[base + r] = base + our_order[r]

    for base in _REL4_BLOCKS:
        for r in range(NUM_PLAYERS):
            src[base + r] = base + c_of_r[r]

    for base in _REL5_BLOCKS:
        for r in range(NUM_PLAYERS):
            src[base + 1 + r] = base + 1 + c_of_r[r]

    src.setflags(write=False)   # cached: never let a caller mutate it
    return src


def _decode_turn_order_from_canonical(canonical: np.ndarray, player_id: int) -> tuple[int, ...]:
    """Read `env.turn_order` back out of a canonical vector's 266..269 block."""
    canon = canonical_seat_order(player_id)
    slots = canonical[TURN_ORDER_BLOCK : TURN_ORDER_BLOCK + NUM_PLAYERS]
    decoded = []
    for v in slots:
        j = int(round(float(v) * _TURN_SLOT_DIV))
        if not 0 <= j < NUM_PLAYERS:
            raise ValueError(
                f"cannot decode turn order from dims {TURN_ORDER_BLOCK}..."
                f"{TURN_ORDER_BLOCK + 3} (got {list(map(float, slots))}); "
                "pass turn_order= explicitly"
            )
        decoded.append(canon[j])
    if sorted(decoded) != list(range(NUM_PLAYERS)):
        raise ValueError(
            f"decoded turn order {decoded} is not a permutation; pass turn_order= explicitly"
        )
    return tuple(decoded)


def to_our_frame(
    canonical_300: np.ndarray,
    player_id: int,
    turn_order: Sequence[int] | None = None,
) -> np.ndarray:
    """Convert an observation the ENGINE built into our seat-invariant frame.

    This is the tournament path: the organizer runs their unmodified engine,
    which emits `build_state_vector(..., agent_id=player_id, env)`; we permute
    it into the frame the trained weights were fitted in.

    `turn_order` defaults to being decoded from the vector's own 266..269 block,
    which is exact.  Pass `env.turn_order` if you have it (cheaper, and it
    works on vectors built with `env=None`, whose tail is all zeros).
    """
    canonical = np.asarray(canonical_300, dtype=np.float32).reshape(-1)
    if canonical.shape[0] != STATE_DIM:
        raise ValueError(f"expected a {STATE_DIM}-vector, got shape {canonical.shape}")
    player_id = int(player_id)
    if turn_order is None:
        turn_order = _decode_turn_order_from_canonical(canonical, player_id)
    turn_order = tuple(int(p) for p in turn_order)
    _, _, _, r_of_c = _rel_maps(player_id, turn_order)

    ours = canonical[obs_index_perm(player_id, turn_order)].astype(np.float32)

    # 266..269: constant by construction in our frame.
    for r in range(NUM_PLAYERS):
        ours[TURN_ORDER_BLOCK + r] = np.float32(r / _TURN_SLOT_DIV)

    # 298: scalar-coded relative player id -> recode, do not permute.
    ours[OUTGOING_TRADE_DIM] = _recode_player_scalar(
        canonical[OUTGOING_TRADE_DIM], r_of_c
    )
    return ours


def to_canonical(
    our_vec: np.ndarray,
    player_id: int,
    turn_order: Sequence[int] | None = None,
) -> np.ndarray:
    """Inverse of `to_our_frame`.

    `turn_order` CANNOT be recovered from our frame -- seat identity is exactly
    what seat-invariance destroys (see "the dead block" in the module docstring)
    -- so pass `env.turn_order`.  It defaults to `[0..3]`, which is correct only
    for an unshuffled table.
    """
    ours = np.asarray(our_vec, dtype=np.float32).reshape(-1)
    if ours.shape[0] != STATE_DIM:
        raise ValueError(f"expected a {STATE_DIM}-vector, got shape {ours.shape}")
    player_id = int(player_id)
    turn_order = tuple(int(p) for p in (turn_order if turn_order is not None else _IDENTITY_TURN_ORDER))
    canon_order, _, c_of_r, _ = _rel_maps(player_id, turn_order)

    src = obs_index_perm(player_id, turn_order)
    canonical = np.empty(STATE_DIM, dtype=np.float32)
    canonical[src] = ours                     # invert the gather

    # 266..269: absolute turn slot t holds the CANONICAL relative id there.
    for t in range(NUM_PLAYERS):
        canonical[TURN_ORDER_BLOCK + t] = np.float32(
            canon_order.index(turn_order[t]) / _TURN_SLOT_DIV
        )

    canonical[OUTGOING_TRADE_DIM] = _recode_player_scalar(ours[OUTGOING_TRADE_DIM], c_of_r)
    return canonical


def _recode_player_scalar(value: Any, mapping: Sequence[int]) -> np.float32:
    """Recode a `(1 + relative_id) / 5` scalar through a relative-slot mapping.

    0.0 means "no such player" and passes through untouched.
    """
    v = float(value)
    if v == 0.0:
        return np.float32(0.0)
    idx = int(round(v * _PLAYER_ENCODE_DIV)) - 1
    if not 0 <= idx < NUM_PLAYERS:
        raise ValueError(f"dim {OUTGOING_TRADE_DIM} decodes to relative player {idx}")
    return np.float32((1 + mapping[idx]) / _PLAYER_ENCODE_DIV)


# --------------------------------------------------------------------------
# The action space needs the same frame change
# --------------------------------------------------------------------------
#
# Three action blocks address another player by an index into
# `others = [i for i in range(4) if i != pid]` (ascending absolute):
#
#     buy_trade   252 = 3 players x 28 properties x 3 cash levels, stride  84
#     sell_trade  252 = same layout,                               stride  84
#     exch_trade 2268 = 3 players x 28 offered x 27 requested,     stride 756
#
# engine/env.py:851-854 (`_make_trade_offer`) and :882-885
# (`_make_exchange_offer`).  Everything else in the 2958 is board- or
# self-addressed and is identical in both frames.

_TRADE_BLOCKS = (
    (OFFSETS["buy_trade"], NUM_PROPERTIES * len(TRADE_CASH_LEVELS)),
    (OFFSETS["sell_trade"], NUM_PROPERTIES * len(TRADE_CASH_LEVELS)),
    (OFFSETS["exch_trade"], NUM_PROPERTIES * (NUM_PROPERTIES - 1)),
)


@lru_cache(maxsize=64)
def _other_maps(player_id: int, turn_order: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(our other-index -> canonical other-index, and its inverse).

    "other index" is 0..2, the trade-target encoding used inside the action id.
    """
    canon_others = tuple(i for i in range(NUM_PLAYERS) if i != int(player_id))
    our_others = seat_order_from_turn_order(player_id, turn_order)[1:]
    to_canon = tuple(canon_others.index(s) for s in our_others)
    to_ours = tuple(our_others.index(s) for s in canon_others)
    return to_canon, to_ours


@lru_cache(maxsize=64)
def action_perm_to_canonical(
    player_id: int, turn_order: tuple[int, ...] | None = None
) -> np.ndarray:
    """Array `P` with `canonical_action = P[our_action]`, length 2958."""
    if turn_order is None:
        turn_order = _IDENTITY_TURN_ORDER
    to_canon, _ = _other_maps(int(player_id), tuple(turn_order))
    perm = np.arange(ACTION_SPACE_SIZE, dtype=np.int64)
    for base, stride in _TRADE_BLOCKS:
        for r in range(NUM_PLAYERS - 1):
            c = to_canon[r]
            perm[base + r * stride : base + (r + 1) * stride] = np.arange(
                base + c * stride, base + (c + 1) * stride, dtype=np.int64
            )
    perm.setflags(write=False)  # cached: never let a caller mutate it
    return perm


@lru_cache(maxsize=64)
def action_perm_to_our_frame(
    player_id: int, turn_order: tuple[int, ...] | None = None
) -> np.ndarray:
    """Array `Q` with `our_action = Q[canonical_action]`, length 2958."""
    perm = action_perm_to_canonical(player_id, turn_order)
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(ACTION_SPACE_SIZE, dtype=np.int64)
    inverse.setflags(write=False)
    return inverse


def action_to_canonical(
    action: int, player_id: int, turn_order: Sequence[int] | None = None
) -> int:
    """Translate an action index our policy emitted into the engine's frame.

    ALWAYS call this before `env.step` when the policy was trained on
    `build_obs`.  Skipping it silently sends trade offers to the wrong seat.
    """
    key = None if turn_order is None else tuple(int(p) for p in turn_order)
    return int(action_perm_to_canonical(int(player_id), key)[int(action)])


def action_to_our_frame(
    action: int, player_id: int, turn_order: Sequence[int] | None = None
) -> int:
    """Translate an engine-frame action index into our policy's frame."""
    key = None if turn_order is None else tuple(int(p) for p in turn_order)
    return int(action_perm_to_our_frame(int(player_id), key)[int(action)])


def legal_actions_our_frame(game_or_env: Any, player_id: int | None = None) -> tuple[int, ...]:
    """`env.get_allowed_actions` translated into our action frame."""
    env = unwrap(game_or_env)
    pid = env.whose_turn() if player_id is None else int(player_id)
    order = turn_order_of(env)
    q = action_perm_to_our_frame(pid, order)
    return tuple(int(q[a]) for a in legal_actions(env, pid))


def legal_mask_our_frame(game_or_env: Any, player_id: int | None = None) -> np.ndarray:
    """Boolean (2958,) mask over OUR action indices.  Feed this to the policy."""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
    mask[list(legal_actions_our_frame(game_or_env, player_id))] = True
    return mask
