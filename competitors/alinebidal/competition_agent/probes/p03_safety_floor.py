"""Experiment 3 — the protected cash cushion, and whether it scales.

Question
--------
A2 located a $200 floor on the *buy* decision. Is the same floor applied to
discretionary spending generally (building, unmortgaging), and does the
cushion grow when opponents are developed enough to charge large rent?

Method
------
Seat 0 holds a completed orange monopoly and faces a build decision in
pre_roll; cash is swept to find the smallest cash at which it will improve.
Two things are varied independently:

  * opponent development — rival holds the green monopoly with H houses each,
    H in 0..5 (5 = hotel). Green hotel rent is $1,275-$1,400, so H moves the
    worst rent seat 0 could be charged by roughly an order of magnitude.
  * reachability — seat 0 is parked either 6 squares before the green group
    (reachable with one roll, high 2d6 mass) or on the far side of the board
    (not reachable next turn). If the cushion tracks *reachable* danger rather
    than danger in the abstract, only the near position should move the flip.

The same sweep is repeated for unmortgaging (cost = 1.1x mortgage value),
to test whether the cushion is a property of discretionary spending in
general or specific to building.

Output: probes/p03_safety_floor.csv
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from competition_agent.proc import managed_pool  # noqa: E402
from competition_agent.probe_harness import (  # noqa: E402
    COLOR_GROUPS, PROPERTIES, ProbeWriter, ask_value,
    bisect_flip, blank_board, describe, give, legal, set_pre_roll,
)

SEED = 20250811
OWN_GROUP = COLOR_GROUPS["orange"]        # 16, 18, 19 — house price $100
RIVAL_GROUP = COLOR_GROUPS["green"]       # 31, 32, 34 — hotel rent $1275-1400
NEAR_POS = 25                             # 6 before Pacific (31): reachable
FAR_POS = 1                               # 30 squares away: not reachable
MAX_CASH = 3000


def _build_env(dev: int, pos: int, mortgaged_target: int | None):
    _, env = blank_board(seed=SEED)
    give(env, 0, OWN_GROUP)
    if mortgaged_target is not None:
        env.properties[mortgaged_target].mortgaged = True
    give(env, 1, RIVAL_GROUP, houses=dev)
    env.players[0].position = pos
    for opid in (1, 2, 3):
        env.players[opid].position = 20
    return env


def _acts(env, pid, kind, target):
    """The action id under test, or None if it is not currently legal."""
    allowed = set(legal(env, pid))
    if kind == "build":
        from monopoly_game_engine.constants import REAL_ESTATE_IDS
        from monopoly_game_engine.actions import OFFSETS
        cand = [OFFSETS["improve_house"] + REAL_ESTATE_IDS.index(s)
                for s in OWN_GROUP]
    else:
        from monopoly_game_engine.constants import PROPERTY_IDS
        from monopoly_game_engine.actions import OFFSETS
        cand = [OFFSETS["unmortgage"] + PROPERTY_IDS.index(target)]
    live = [c for c in cand if c in allowed]
    return live or None


def _spends_at(dev, pos, kind, target, cash):
    env = _build_env(dev, pos, target if kind == "unmortgage" else None)
    set_pre_roll(env, 0, cash=cash)
    cand = _acts(env, 0, kind, target)
    if cand is None:
        return False
    return ask_value(env, 0) in cand


def _job(item):
    kind, dev, pos, label = item
    target = 16 if kind == "unmortgage" else None
    flip = bisect_flip(
        lambda c: _spends_at(dev, pos, kind, target, c), 0, MAX_CASH
    )
    # what does it do just below and just above the flip?
    below = above = ""
    if flip is not None:
        for delta, slot in ((-1, "below"), (0, "above")):
            env = _build_env(dev, pos, target if kind == "unmortgage" else None)
            set_pre_roll(env, 0, cash=max(0, flip + delta))
            d = describe(ask_value(env, 0))
            if slot == "below":
                below = d
            else:
                above = d
    cost = 100 if kind == "build" else int(PROPERTIES[16]["mortgage"] * 1.1)
    return {
        "decision": kind,
        "rival_houses": dev,
        "position": label,
        "our_pos": pos,
        "spend_cost": cost,
        "flip_cash": flip if flip is not None else "",
        "cushion_after_spend": (flip - cost) if flip is not None else "",
        "action_below_flip": below,
        "action_at_flip": above,
        "seed": SEED,
    }


def main() -> int:
    items = []
    for kind in ("build", "unmortgage"):
        for dev in range(6):
            for pos, label in ((NEAR_POS, "near"), (FAR_POS, "far")):
                items.append((kind, dev, pos, label))

    with managed_pool(10) as pool:
        rows = pool.map(_job, items)

    fields = list(rows[0].keys())
    with ProbeWriter("p03_safety_floor", fields) as out:
        for r in rows:
            out.write(**r)

    print(f"{'decision':<11}{'rivalH':>7}{'pos':>6}{'cost':>6}"
          f"{'flip':>7}{'cushion':>9}   at-flip action")
    print("-" * 74)
    for r in rows:
        print(f"{r['decision']:<11}{r['rival_houses']:>7}{r['position']:>6}"
              f"{r['spend_cost']:>6}{str(r['flip_cash']):>7}"
              f"{str(r['cushion_after_spend']):>9}   {r['action_at_flip']}")
    print(f"\nwrote {out.path} ({out.rows} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
