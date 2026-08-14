"""Experiment 4 — development order and its tie-break.

Question
--------
Given a completed monopoly and ample cash, which deed does the teacher improve
first, to what level, and what breaks a tie between deeds that are identical
on paper?

Method
------
Seat 0 holds one completed colour group with $5,000 and no other holdings.
The teacher is asked repeatedly and each chosen improvement is applied through
`game.step`, so the engine's own even-building rule and bank inventory are
respected. The resulting sequence of squares is the development order.

Two groups contain deeds that are identical in price and rent — orange 16/18
($180, rent 14/70/200/550/750/950) and red 21/23 ($220), and pink 11/13
($140) — so any consistent preference between them isolates the tie-break
independently of value.

The sweep is repeated with the bank's house supply constrained, to see whether
scarcity changes the order or only the reachable level.

Output: probes/p04_development_order.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from competition_agent.probe_harness import (  # noqa: E402
    COLOR_GROUPS, PROPERTIES, ActionType, ProbeWriter, ask_value, blank_board,
    describe, give, set_pre_roll,
)

SEED = 20250811
GROUPS = ["brown", "lightblue", "pink", "orange", "red", "yellow",
          "green", "darkblue"]
MAX_STEPS = 40
END = int(ActionType.END_TURN)


def run(color: str, cash: int, houses_avail: int, hotels_avail: int):
    game, env = blank_board(seed=SEED)
    squares = COLOR_GROUPS[color]
    give(env, 0, squares)
    env.houses_available = houses_avail
    env.hotels_available = hotels_avail
    set_pre_roll(env, 0, cash=cash)

    seq, spent_start = [], env.players[0].cash
    for _ in range(MAX_STEPS):
        a = ask_value(env, 0)
        d = describe(a)
        if a == END or not d.startswith("improve"):
            break
        seq.append(d)
        game.step(a)

    return {
        "color": color,
        "group": "|".join(str(s) for s in squares),
        "prices": "|".join(str(PROPERTIES[s]["price"]) for s in squares),
        "cash": cash,
        "houses_available": houses_avail,
        "hotels_available": hotels_avail,
        "first_built": seq[0] if seq else "",
        "n_improvements": len(seq),
        "final_houses": "|".join(
            str(env.properties[s].houses) for s in squares
        ),
        "spent": spent_start - env.players[0].cash,
        "sequence": " -> ".join(seq),
        "stopped_on": describe(ask_value(env, 0)),
        "seed": SEED,
    }


def main() -> int:
    rows = []
    for color in GROUPS:
        rows.append(run(color, 5000, 32, 12))          # unconstrained
        rows.append(run(color, 5000, 3, 0))            # house-scarce, no hotels
        rows.append(run(color, 400, 32, 12))           # cash-constrained

    with ProbeWriter("p04_development_order", list(rows[0].keys())) as out:
        for r in rows:
            out.write(**r)

    for r in rows:
        tag = ("unconstrained" if r["houses_available"] == 32 and
               r["cash"] == 5000 else
               "house-scarce" if r["houses_available"] == 3 else "low-cash")
        print(f"{r['color']:<10} {tag:<14} group={r['group']:<10} "
              f"prices={r['prices']:<12} first={r['first_built']:<22} "
              f"final={r['final_houses']:<8} n={r['n_improvements']:<3} "
              f"stop={r['stopped_on']}")
    print(f"\nwrote {out.path} ({out.rows} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
