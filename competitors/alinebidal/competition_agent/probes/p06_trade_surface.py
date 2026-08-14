"""Experiment 6 — the trade accept/decline surface.

Question
--------
Given an incoming offer, where is the boundary between accept and decline, and
how does it move with the cash sweetener and with what the deed is worth to
each side?

Method
------
Seat 0 receives an offer in out_of_turn, where `ACCEPT_TRADE` and
`DECLINE_TRADE` are both legal. The offer is swept over a grid:

  * which deed is offered — one that completes seat 0's group (high value to
    seat 0), one that is merely useful, and one that is worthless to it;
  * which deed is requested — including the completing piece of seat 0's own
    near-monopoly, which it should be most reluctant to give up;
  * the cash sweetener, from -400 (seat 0 pays) to +400 (seat 0 is paid), in
    $25 steps.

Monotonicity in the sweetener is verified with `scan_flip` rather than
assumed, after the p03 failure (SPEC D6).

This also settles the provisional trade exclusion in the Phase 4 gate
(DECISIONS D1.1): the earlier divergence probe had only 14 trade states.

Output: probes/p06_trade_surface.csv
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from competition_agent.proc import managed_pool  # noqa: E402
from competition_agent.probe_harness import (  # noqa: E402
    PROPERTIES, ActionType, ProbeWriter, ask_rollout, ask_value, blank_board,
    describe, give, legal, scan_flip,
)

SEED = 20250811
ACCEPT = int(ActionType.ACCEPT_TRADE)
DECLINE = int(ActionType.DECLINE_TRADE)

# seat 0 holds two oranges (16, 18) + spares; rival holds the completing 19.
OFFERED = [
    (19, "completes our orange group"),
    (5, "a railroad, no group value"),
    (1, "cheap brown, near worthless"),
]
REQUESTED = [
    (25, "our spare railroad"),
    (16, "our own orange piece"),
]
CASH = list(range(-400, 401, 25))
ROLLOUT_CASH = list(range(-400, 401, 100))   # 9 points per cell


def _env(offered_sq, requested_sq, cash_to_us):
    from monopoly_game_engine.env import TradeOffer

    _, env = blank_board(seed=SEED)
    give(env, 0, [16, 18, 25, 37])
    give(env, 1, [19, 5, 1, 21, 23])
    env.players[0].cash = 1200
    env.players[1].cash = 1200
    env.phase = "out_of_turn"
    env.current_turn_idx = 1
    env.out_of_turn_pids = [0, 2, 3]
    env.pending_trades[1] = TradeOffer(
        from_player=1, to_player=0,
        offered_prop=env.properties[offered_sq],
        requested_prop=env.properties[requested_sq],
        cash_offered=max(0, cash_to_us),
        cash_requested=max(0, -cash_to_us),
    )
    return env


def _accepts(offered_sq, requested_sq, cash_to_us):
    env = _env(offered_sq, requested_sq, cash_to_us)
    allowed = legal(env, 0)
    if ACCEPT not in allowed or DECLINE not in allowed:
        return False
    return ask_value(env, 0) == ACCEPT


def _job(item):
    offered_sq, off_label, requested_sq, req_label = item
    flip, monotone, points = scan_flip(
        lambda c: _accepts(offered_sq, requested_sq, c), -400, 400, 25
    )
    # Rollout agreement, on a coarser grid: rollout costs ~9 s per trade state
    # (p08), so the full $25 grid would be ~30 min of pure lookahead for a
    # sample whose only job is to widen the Phase 4 trade evidence.
    agree = total = 0
    for c in ROLLOUT_CASH:
        env = _env(offered_sq, requested_sq, c)
        if ACCEPT not in legal(env, 0):
            continue
        total += 1
        agree += (ask_value(env, 0) == ask_rollout(env, 0))

    return {
        "offered_sq": offered_sq, "offered_desc": off_label,
        "requested_sq": requested_sq, "requested_desc": req_label,
        "offered_price": PROPERTIES[offered_sq]["price"],
        "requested_price": PROPERTIES[requested_sq]["price"],
        "accept_from_cash": flip if flip is not None else "",
        "monotone": monotone,
        "never_accepts": flip is None,
        "grid_points": points,
        "rollout_states": total, "rollout_agree": agree,
        "seed": SEED,
    }


def main() -> int:
    items = [(o, ol, r, rl) for o, ol in OFFERED for r, rl in REQUESTED]
    with managed_pool(6) as pool:
        rows = pool.map(_job, items)

    with ProbeWriter("p06_trade_surface", list(rows[0].keys())) as out:
        for r in rows:
            out.write(**r)

    print(f"{'offered':<32}{'requested':<26}{'accept from':>12}"
          f"{'monotone':>10}{'roll agree':>12}")
    print("-" * 94)
    for r in rows:
        acc = ("never" if r["never_accepts"]
               else f"${r['accept_from_cash']:+d}")
        print(f"{str(r['offered_sq'])+' '+r['offered_desc']:<32}"
              f"{str(r['requested_sq'])+' '+r['requested_desc']:<26}"
              f"{acc:>12}{str(r['monotone']):>10}"
              f"{str(r['rollout_agree'])+'/'+str(r['rollout_states']):>12}")
    tot_a = sum(r["rollout_agree"] for r in rows)
    tot_n = sum(r["rollout_states"] for r in rows)
    print(f"\nrollout/value agreement on trade states: {tot_a}/{tot_n}")
    print(f"wrote {out.path} ({out.rows} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
