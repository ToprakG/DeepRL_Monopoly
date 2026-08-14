"""Phase 1 probe harness: controlled state construction + opaque teacher query.

Design contract (see DECISIONS.md D0.3)
--------------------------------------
The teacher is queried **only** as ``decide(env) -> action`` and only the
selected action id is recorded. No value component, safety margin, auction
ceiling or rejection reason is ever read, and no ASU helper other than the two
policy classes is imported. Every probe therefore produces evidence of the
same kind a competitor would have: a state in, an action out.

State construction
------------------
Probes need *controlled* states, not sampled ones, so ``blank_board`` resets a
seeded game to a known canonical position and the ``set_*`` helpers move one
variable at a time. Every constructed state is validated with
``env.get_allowed_actions`` before being shown to the teacher: if the decision
under test is not actually legal, the probe row is marked invalid rather than
silently recording a default action.
"""

from __future__ import annotations

import copy
import csv
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ASU_FROZEN_TEACHER import ASURolloutV1, ASUValueV1  # noqa: E402
from ASU_FROZEN_TEACHER.evaluate import _new_seeded_game  # noqa: E402
from monopoly_game_engine.actions import (  # noqa: E402
    ActionType,
    AuctionAction,
    action_to_description,
)
from monopoly_game_engine.constants import (  # noqa: E402
    COLOR_GROUPS,
    PROPERTIES,
    PROPERTY_IDS,
    REAL_ESTATE_IDS,
)

PROBE_DIR = Path(__file__).resolve().parent / "probes"

PHASE_PRE_ROLL = "pre_roll"
PHASE_POST_ROLL = "post_roll"
PHASE_OUT_OF_TURN = "out_of_turn"
PHASE_AUCTION = "auction"


# --------------------------------------------------------------------------
# teacher access — the only two entry points this package is allowed to use
# --------------------------------------------------------------------------
_VALUE_CACHE: Dict[int, ASUValueV1] = {}
_ROLLOUT_CACHE: Dict[int, ASURolloutV1] = {}


def ask_value(env, pid: int) -> int:
    """Opaque query of ASUValueV1. Returns the selected action id only."""
    agent = _VALUE_CACHE.get(pid)
    if agent is None:
        agent = _VALUE_CACHE[pid] = ASUValueV1(pid)
    return int(agent.choose_action(env))


def ask_rollout(env, pid: int) -> int:
    """Opaque query of ASURolloutV1. Returns the selected action id only."""
    agent = _ROLLOUT_CACHE.get(pid)
    if agent is None:
        agent = _ROLLOUT_CACHE[pid] = ASURolloutV1(pid)
    return int(agent.choose_action(env))


def describe(action: int) -> str:
    return action_to_description(int(action))


# --------------------------------------------------------------------------
# controlled state construction
# --------------------------------------------------------------------------
def blank_board(seed: int = 0, cash: int = 1500):
    """A seeded game reset to a canonical, fully-unowned position.

    All deeds return to the bank, all players sit on Go with equal cash, no
    houses, no jail, no debt, no pending trades or auctions.
    """
    game = _new_seeded_game(seed)
    env = game.env

    for prop in env.properties.values():
        prop.owner = None
        prop.mortgaged = False
        prop.houses = 0
        prop.is_monopoly = False

    for player in env.players:
        player.cash = cash
        player.position = 0
        player.in_jail = False
        player.jail_turns = 0
        player.gooj_card = False
        player.bankrupt = False
        player.properties = []

    env.phase = PHASE_PRE_ROLL
    env.has_rolled = False
    env.current_turn_idx = 0
    env.turn_order = list(range(len(env.players)))
    env.debt_player = None
    env.debt_amount = 0
    env.debt_creditor = None
    env.consecutive_doubles = 0
    env.extra_roll_pending = False
    env.last_dice = (0, 0)
    env.out_of_turn_pids = []
    env.pending_trades = {}
    env.auction_property_id = None
    env.auction_high_bid = 0
    env.auction_high_bidder = None
    env.auction_bidders = []
    env.auction_current_pid = None
    env.houses_available = 32
    env.hotels_available = 12
    env.round = 1
    env.done = False
    env._update_monopolies()
    return game, env


def give(env, pid: int, squares: Iterable[int], houses: int = 0,
         mortgaged: bool = False) -> None:
    """Assign deeds to a player, optionally developed or mortgaged."""
    for sq in squares:
        prop = env.properties[sq]
        if prop.owner is not None:
            env.players[prop.owner].properties.remove(prop)
        prop.owner = pid
        prop.mortgaged = mortgaged
        env.players[pid].properties.append(prop)
    env._update_monopolies()
    # houses only after monopoly flags settle — the engine gates building on them
    for sq in squares:
        prop = env.properties[sq]
        if houses and prop.is_real_estate:
            prop.houses = houses
            env.houses_available -= houses if houses < 5 else 0
            env.hotels_available -= 1 if houses == 5 else 0


def set_turn(env, pid: int) -> None:
    env.current_turn_idx = env.turn_order.index(pid)


def set_buy_decision(env, pid: int, square: int, cash: int) -> None:
    """Put `pid` on an unowned `square` facing the buy/skip choice.

    Mirrors the engine's post-roll branch: phase post_roll, has_rolled True,
    no outstanding debt, deed unowned, cash >= price => BUY_PROPERTY legal.
    """
    set_turn(env, pid)
    env.phase = PHASE_POST_ROLL
    env.has_rolled = True
    env.debt_player = None
    env.players[pid].cash = cash
    env.players[pid].position = square
    env.players[pid].in_jail = False


def set_pre_roll(env, pid: int, cash: Optional[int] = None) -> None:
    """Pre-roll decision point: build / mortgage / unmortgage / trade / jail."""
    set_turn(env, pid)
    env.phase = PHASE_PRE_ROLL
    env.has_rolled = False
    env.debt_player = None
    if cash is not None:
        env.players[pid].cash = cash


def set_auction(env, pid: int, square: int, high_bid: int = 0,
                high_bidder: Optional[int] = None,
                bidders: Optional[Sequence[int]] = None) -> None:
    """Put `pid` on the clock in an auction for `square`."""
    env.phase = PHASE_AUCTION
    env.auction_property_id = square
    env.auction_high_bid = high_bid
    env.auction_high_bidder = high_bidder
    env.auction_bidders = list(
        bidders if bidders is not None else range(len(env.players))
    )
    env.auction_current_pid = pid


def set_jail(env, pid: int, gooj: bool = False, jail_turns: int = 0,
             cash: int = 1500) -> None:
    set_turn(env, pid)
    env.phase = PHASE_PRE_ROLL
    env.has_rolled = False
    env.debt_player = None
    p = env.players[pid]
    p.in_jail = True
    p.jail_turns = jail_turns
    p.gooj_card = gooj
    p.cash = cash
    p.position = 10


# --------------------------------------------------------------------------
# validation + CSV
# --------------------------------------------------------------------------
def legal(env, pid: int) -> List[int]:
    return [int(a) for a in env.get_allowed_actions(pid)]


def requires(env, pid: int, *action_ids: int) -> bool:
    """True iff every listed action is currently legal for `pid`.

    Probes call this before querying the teacher so that a malformed state
    is reported as invalid instead of producing a misleading "skip".
    """
    allowed = set(legal(env, pid))
    return all(int(a) in allowed for a in action_ids)


class ProbeWriter:
    """CSV sink. One file per experiment, seed recorded on every row."""

    def __init__(self, name: str, fieldnames: Sequence[str]):
        PROBE_DIR.mkdir(parents=True, exist_ok=True)
        self.path = PROBE_DIR / f"{name}.csv"
        self._fh = self.path.open("w", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=list(fieldnames))
        self._w.writeheader()
        self.rows = 0

    def write(self, **row) -> None:
        self._w.writerow(row)
        self.rows += 1

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def deed_price(square: int) -> int:
    return PROPERTIES[square]["price"]


def deed_color(square: int) -> str:
    return PROPERTIES[square]["color"]


def group_of(square: int) -> List[int]:
    return list(COLOR_GROUPS[deed_color(square)])


def bisect_flip(predicate, lo: int, hi: int) -> Optional[int]:
    """Smallest x in [lo, hi] with predicate(x) true, or None.

    UNSAFE ON ITS OWN — assumes predicate is monotone in x. Prefer
    `scan_flip`, which verifies monotonicity, unless the probe checks it
    separately (as p01 does with its own coarse scan).

    p03 was invalidated by using this on a non-monotone predicate: seat 0 held
    a buildable monopoly, so "chooses unmortgage" went
    false -> false(builds instead) -> true as cash rose. Bisection returned a
    plausible number in some rows and silently returned None in others, where
    a linear scan showed a clear flip.
    """
    if predicate(hi) is not True:
        return None
    if predicate(lo) is True:
        return lo
    while lo < hi:
        mid = (lo + hi) // 2
        if predicate(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def scan_flip(predicate, lo: int, hi: int, step: int):
    """Locate a flip by linear scan and report whether it is monotone.

    Returns ``(flip, monotone, points)``. ``flip`` is the first x where the
    predicate holds; ``monotone`` is False if it ever reverts afterwards,
    which means the "threshold" framing does not apply to that decision and
    any single number reported for it would be misleading.

    Costlier than `bisect_flip`, and that is the point: the threshold is only
    meaningful if the response is monotone, so the scan that establishes the
    threshold should be the same scan that proves it exists.
    """
    scan = [(x, bool(predicate(x))) for x in range(lo, hi + 1, step)]
    first = next((i for i, (_, b) in enumerate(scan) if b), None)
    if first is None:
        return None, True, len(scan)
    monotone = all(b for _, b in scan[first:])
    return scan[first][0], monotone, len(scan)


__all__ = [
    "ActionType", "AuctionAction", "COLOR_GROUPS", "PROPERTIES",
    "PROPERTY_IDS", "REAL_ESTATE_IDS", "ProbeWriter", "ask_rollout",
    "ask_value", "bisect_flip", "blank_board", "deed_color", "deed_price",
    "describe", "give", "group_of", "legal", "requires", "set_auction",
    "scan_flip", "set_buy_decision", "set_jail", "set_pre_roll", "set_turn",
    "copy",
]
