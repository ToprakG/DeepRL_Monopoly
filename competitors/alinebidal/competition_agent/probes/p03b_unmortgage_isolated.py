"""Experiment 3b — the unmortgage cushion, with building removed.

Why this exists
---------------
p03's unmortgage sweep is invalid and its rows must not be used. It assumed
the "chooses unmortgage" predicate is monotone in cash, but seat 0 also held a
buildable monopoly, so the real response at rival development 2 was

    END_TURN  ->  improve_house  ->  unmortgage

as cash rises. Building outranks unmortgaging over a middle cash band, which
makes the predicate non-monotone: `bisect_flip` returned a number in some rows
and silently returned *nothing* at rival development 4-5, where a linear scan
shows a clear flip at $1,100.

This probe removes the competing action instead of assuming it away. Seat 0
holds two railroads, one of them mortgaged: no colour group is complete, so no
`improve_*` action is ever legal and unmortgaging is the only discretionary
spend available. Monotonicity is then verified explicitly with a linear scan
before any threshold is reported — a lesson applied to `bisect_flip` generally.

Output: probes/p03b_unmortgage_isolated.csv
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from competition_agent.proc import managed_pool  # noqa: E402
from competition_agent.probe_harness import (  # noqa: E402
    COLOR_GROUPS, PROPERTIES, PROPERTY_IDS, ProbeWriter, ask_value,
    blank_board, describe, give, legal, set_pre_roll,
)
from monopoly_game_engine.actions import OFFSETS  # noqa: E402

SEED = 20250811
HELD = [5, 15]                      # two railroads: never a buildable group
TARGET = 5                          # the mortgaged one; unmortgage = 1.1 x 100
RIVAL_GROUP = COLOR_GROUPS["green"]
NEAR_POS, FAR_POS = 25, 1
STEP, MAX_CASH = 5, 2000

UNMORTGAGE = OFFSETS["unmortgage"] + PROPERTY_IDS.index(TARGET)


def _env(dev: int, pos: int, cash: int):
    _, env = blank_board(seed=SEED)
    give(env, 0, HELD)
    env.properties[TARGET].mortgaged = True
    give(env, 1, RIVAL_GROUP, houses=dev)
    env.players[0].position = pos
    for opid in (1, 2, 3):
        env.players[opid].position = 20
    set_pre_roll(env, 0, cash=cash)
    return env


def _job(item):
    dev, pos, label = item
    scan = []
    for cash in range(0, MAX_CASH + 1, STEP):
        env = _env(dev, pos, cash)
        if UNMORTGAGE not in legal(env, 0):
            scan.append((cash, False))
            continue
        scan.append((cash, ask_value(env, 0) == UNMORTGAGE))

    first = next((i for i, (_, b) in enumerate(scan) if b), None)
    monotone = first is not None and all(b for _, b in scan[first:])
    flip = scan[first][0] if first is not None else None

    # confirm no build action was ever available (the p03 failure mode)
    env = _env(dev, pos, MAX_CASH)
    build_legal = any(
        OFFSETS["improve_house"] <= a < OFFSETS["sell_house"]
        for a in legal(env, 0)
    )
    at_flip = describe(ask_value(_env(dev, pos, flip), 0)) if flip else ""

    cost = int(PROPERTIES[TARGET]["mortgage"] * 1.1)
    return {
        "rival_houses": dev, "position": label, "our_pos": pos,
        "unmortgage_cost": cost,
        "flip_cash": flip if flip is not None else "",
        "cushion_after_spend": (flip - cost) if flip is not None else "",
        "monotone": monotone, "build_action_available": build_legal,
        "action_at_flip": at_flip, "scan_points": len(scan), "seed": SEED,
    }


def main() -> int:
    items = [(dev, pos, label)
             for dev in range(6)
             for pos, label in ((NEAR_POS, "near"), (FAR_POS, "far"))]
    with managed_pool(10) as pool:
        rows = pool.map(_job, items)

    with ProbeWriter("p03b_unmortgage_isolated", list(rows[0].keys())) as out:
        for r in rows:
            out.write(**r)

    print(f"{'rivalH':>7}{'pos':>6}{'cost':>6}{'flip':>7}{'cushion':>9}"
          f"{'monotone':>10}{'build?':>8}   at-flip")
    print("-" * 76)
    for r in rows:
        print(f"{r['rival_houses']:>7}{r['position']:>6}"
              f"{r['unmortgage_cost']:>6}{str(r['flip_cash']):>7}"
              f"{str(r['cushion_after_spend']):>9}{str(r['monotone']):>10}"
              f"{str(r['build_action_available']):>8}   {r['action_at_flip']}")
    print(f"\nwrote {out.path} ({out.rows} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
