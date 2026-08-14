"""Experiment 7 — jail policy.

Question
--------
In jail, does the teacher buy its way out or sit? Does holding a
get-out-of-jail-free card change the answer, and does the choice depend on how
dangerous the board outside has become?

Method
------
Seat 0 is placed in jail in pre_roll, where the legal menu is END_TURN (sit),
PAY_BAIL when it can afford $50, and USE_GOOJ_CARD when it holds one. Four
variables are swept independently:

  * card held or not;
  * jail turn 0, 1, 2, 3 (the engine's forced-exit boundary is 3);
  * cash, across the safety floor;
  * outside danger — a rival's green monopoly developed 0..5, which is what
    makes leaving jail expensive. Sitting in jail is shelter: a player in jail
    cannot land on anyone's property.

The classic Monopoly result is that jail is *good* late, when the board is
dangerous, and bad early, when you want to buy deeds. If the teacher tracks
that, the pay/sit boundary should move with rival development.

Output: probes/p07_jail_policy.csv
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from competition_agent.proc import managed_pool  # noqa: E402
from competition_agent.probe_harness import (  # noqa: E402
    COLOR_GROUPS, ActionType, ProbeWriter, ask_value, blank_board, describe,
    give, legal, set_jail,
)

SEED = 20250811
CASHES = [50, 100, 200, 260, 400, 800, 1500]
DEVS = [0, 2, 4, 5]
TURNS = [0, 1, 2, 3]


def _env(card, turns, cash, dev):
    _, env = blank_board(seed=SEED)
    give(env, 0, [5, 15])                      # something to own, no monopoly
    give(env, 1, COLOR_GROUPS["green"], houses=dev)
    for opid in (1, 2, 3):
        env.players[opid].position = 20
    set_jail(env, 0, gooj=card, jail_turns=turns, cash=cash)
    return env


def _job(item):
    card, turns, cash, dev = item
    env = _env(card, turns, cash, dev)
    allowed = legal(env, 0)
    a = ask_value(env, 0)
    d = describe(a)
    return {
        "has_card": card, "jail_turns": turns, "cash": cash,
        "rival_houses": dev,
        "pay_bail_legal": int(ActionType.PAY_BAIL) in allowed,
        "card_legal": int(ActionType.USE_GOOJ_CARD) in allowed,
        "n_legal": len(allowed),
        "choice": d,
        "leaves_jail": d in ("PAY_BAIL", "USE_GOOJ_CARD"),
        "seed": SEED,
    }


def main() -> int:
    items = [(c, t, cash, d)
             for c in (False, True)
             for t in TURNS
             for cash in CASHES
             for d in DEVS]
    with managed_pool(10) as pool:
        rows = pool.map(_job, items)

    with ProbeWriter("p07_jail_policy", list(rows[0].keys())) as out:
        for r in rows:
            out.write(**r)

    # summary: choice by (card, rival development), collapsed over cash/turns
    print("choice counts by card / rival development")
    print(f"{'card':>6}{'rivalH':>8}   " + "  ".join(
        f"{k:<16}" for k in ("END_TURN", "PAY_BAIL", "USE_GOOJ_CARD")))
    print("-" * 74)
    for card in (False, True):
        for dev in DEVS:
            sub = [r for r in rows
                   if r["has_card"] == card and r["rival_houses"] == dev]
            counts = {k: sum(1 for r in sub if r["choice"] == k)
                      for k in ("END_TURN", "PAY_BAIL", "USE_GOOJ_CARD")}
            print(f"{str(card):>6}{dev:>8}   " + "  ".join(
                f"{counts[k]:<16}" for k in counts))
    print(f"\nleaves jail overall: "
          f"{sum(1 for r in rows if r['leaves_jail'])}/{len(rows)}")
    print(f"wrote {out.path} ({out.rows} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
