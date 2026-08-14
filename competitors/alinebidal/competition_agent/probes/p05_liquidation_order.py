"""Experiment 5 — liquidation order under debt.

Question
--------
When rent is owed that cash cannot cover, in what order does the teacher
raise money? Houses before mortgages? Cheapest asset first? Does it protect
its monopoly?

Method
------
Seat 0 owes a fixed debt in post_roll with `debt_player` set, which is the
engine's forced-rescue branch: the only legal actions are `sell_house`,
`sell_hotel`, `mortgage`, `sell_prop`, or `DECLARE_BANKRUPT` when nothing is
left. Each chosen action is applied through `game.step` and the sequence is
recorded until the debt clears or bankruptcy is declared.

Seat 0's holdings are deliberately mixed so the ordering has something to
choose between: a developed monopoly (orange, houses), a spare undeveloped
monopoly-less deed set (two railroads), and a lone high-value deed
(Boardwalk). Debt size is swept so that shallow debts, which need one or two
sales, are distinguished from deep debts that force everything out.

Output: probes/p05_liquidation_order.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from competition_agent.probe_harness import (  # noqa: E402
    COLOR_GROUPS, ActionType, ProbeWriter, ask_value, blank_board, describe,
    give, set_turn,
)

SEED = 20250811
DEBTS = [50, 150, 300, 600, 1000, 1600, 2500]
MAX_STEPS = 30
BANKRUPT = int(ActionType.DECLARE_BANKRUPT)


def run(debt: int, houses: int):
    game, env = blank_board(seed=SEED)
    give(env, 0, COLOR_GROUPS["orange"], houses=houses)   # developed monopoly
    give(env, 0, [5, 15])                                  # two railroads
    give(env, 0, [39])                                     # lone Boardwalk
    give(env, 1, COLOR_GROUPS["green"], houses=4)          # the creditor
    set_turn(env, 0)
    env.phase = "post_roll"
    env.has_rolled = True
    env.players[0].cash = 0
    env.players[0].position = 34
    env.debt_player = 0
    env.debt_amount = debt
    env.debt_creditor = 1

    seq = []
    for _ in range(MAX_STEPS):
        a = ask_value(env, 0)
        d = describe(a)
        seq.append(d)
        if a == BANKRUPT:
            break
        game.step(a)
        if env.debt_player != 0 or env.done:
            break

    kinds = [s.split("(")[0] for s in seq]
    return {
        "debt": debt,
        "start_houses_per_orange": houses,
        "n_actions": len(seq),
        "first_action": seq[0] if seq else "",
        "first_kind": kinds[0] if kinds else "",
        "went_bankrupt": "DECLARE_BANKRUPT" in kinds,
        "sold_house_before_mortgage": (
            ("sell_house" in kinds and "mortgage" in kinds
             and kinds.index("sell_house") < kinds.index("mortgage"))
            if ("sell_house" in kinds and "mortgage" in kinds) else ""
        ),
        "kind_order": " -> ".join(kinds),
        "sequence": " -> ".join(seq),
        "seed": SEED,
    }


def main() -> int:
    rows = [run(d, h) for h in (0, 2, 4) for d in DEBTS]
    with ProbeWriter("p05_liquidation_order", list(rows[0].keys())) as out:
        for r in rows:
            out.write(**r)

    print(f"{'houses':>7}{'debt':>7}{'n':>4}  {'first action':<24} sequence")
    print("-" * 110)
    for r in rows:
        print(f"{r['start_houses_per_orange']:>7}{r['debt']:>7}"
              f"{r['n_actions']:>4}  {r['first_action']:<24} "
              f"{r['sequence'][:64]}")
    print(f"\nwrote {out.path} ({out.rows} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
