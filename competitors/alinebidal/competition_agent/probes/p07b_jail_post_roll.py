"""Experiment 7b — jail policy at the real decision point.

Why this exists
---------------
p07 swept the jail decision in `pre_roll` and found the teacher never leaves
jail (0/224). That number is real but it does not mean what it looks like.
In `pre_roll`, `END_TURN` does not mean "sit in jail" — the engine's comment is
"end pre-roll, go to post-roll" — so the teacher was simply deferring. The
binding choice happens in `post_roll` with `has_rolled` False, where the menu
is USE_GOOJ_CARD / PAY_BAIL / ROLL_DICE and there is no way to defer.

p07's rows are retained as evidence about *pre-roll deferral*; this probe
measures jail policy proper. Both phases are swept here so the two are
directly comparable on identical states.

Sweep: card held or not x jail turn 0-3 x cash across the safety floor x rival
green development 0-5 (the danger outside jail).

Output: probes/p07b_jail_post_roll.csv
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
LEAVE = {"PAY_BAIL", "USE_GOOJ_CARD"}


def _env(card, turns, cash, dev, phase, holdings):
    _, env = blank_board(seed=SEED)
    if holdings:
        give(env, 0, [5, 15])
    give(env, 1, COLOR_GROUPS["green"], houses=dev)
    for opid in (1, 2, 3):
        env.players[opid].position = 20
    set_jail(env, 0, gooj=card, jail_turns=turns, cash=cash)
    if phase == "post_roll":
        env.phase = "post_roll"
        env.has_rolled = False
    return env


def _job(item):
    card, turns, cash, dev, phase, holdings = item
    env = _env(card, turns, cash, dev, phase, holdings)
    allowed = legal(env, 0)
    d = describe(ask_value(env, 0))
    return {
        "phase": phase, "holdings": holdings, "has_card": card,
        "jail_turns": turns, "cash": cash, "rival_houses": dev,
        "n_legal": len(allowed),
        "bail_legal": int(ActionType.PAY_BAIL) in allowed,
        "card_legal": int(ActionType.USE_GOOJ_CARD) in allowed,
        "choice": d,
        "leaves_jail": d in LEAVE,
        "seed": SEED,
    }


def main() -> int:
    items = [(c, t, cash, d, phase, hold)
             for phase in ("post_roll", "pre_roll")
             for hold in (True, False)
             for c in (False, True)
             for t in TURNS
             for cash in CASHES
             for d in DEVS]
    with managed_pool(10) as pool:
        rows = pool.map(_job, items)

    with ProbeWriter("p07b_jail_post_roll", list(rows[0].keys())) as out:
        for r in rows:
            out.write(**r)

    print(f"{'phase':<11}{'holds':>7}{'card':>7}   "
          f"{'ROLL_DICE':<11}{'PAY_BAIL':<11}{'USE_GOOJ':<11}"
          f"{'END_TURN':<11}{'other':<8} leaves")
    print("-" * 92)
    for phase in ("post_roll", "pre_roll"):
        for hold in (True, False):
            for card in (False, True):
                sub = [r for r in rows if r["phase"] == phase
                       and r["holdings"] == hold and r["has_card"] == card]
                c = {k: sum(1 for r in sub if r["choice"] == k)
                     for k in ("ROLL_DICE", "PAY_BAIL", "USE_GOOJ_CARD",
                               "END_TURN")}
                other = len(sub) - sum(c.values())
                lv = sum(1 for r in sub if r["leaves_jail"])
                print(f"{phase:<11}{str(hold):>7}{str(card):>7}   "
                      f"{c['ROLL_DICE']:<11}{c['PAY_BAIL']:<11}"
                      f"{c['USE_GOOJ_CARD']:<11}{c['END_TURN']:<11}"
                      f"{other:<8} {lv}/{len(sub)}")

    post = [r for r in rows if r["phase"] == "post_roll"]
    print(f"\npost_roll leaves jail: "
          f"{sum(1 for r in post if r['leaves_jail'])}/{len(post)}")
    print(f"wrote {out.path} ({out.rows} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
