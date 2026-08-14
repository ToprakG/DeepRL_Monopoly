"""Experiment 1b — is the buy-threshold residual an expected-rent term?

Motivation
----------
p01 found flip_cash = price + 200 for 21/28 deeds, but 7 deeds flipped
*below* price + 200, all of them squares 5..15 — exactly the deeds an
opponent sitting on Go can reach with one 2d6 roll. Hypothesis: the buy gate
is not `cash_after >= 200` but `cash_after + E[next-round net rent] >= 200`,
so a deed that will collect rent soon needs less cash behind it.

Decisive test
-------------
Hold the deed fixed and move the *opponent* instead. Boardwalk (sq 39) showed
a residual of exactly 0 with all opponents on Go — 39 squares away, i.e.
unreachable next round. If the residual is a rent term, walking one opponent
towards Boardwalk must make the flip point fall, peaking where 2d6 landing
probability peaks (6-8 squares back), and it must scale with the deed's rent.

A flat response across all opponent positions would falsify the hypothesis
and leave the residual unexplained.

Output: probes/p01b_rent_residual.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from competition_agent.probe_harness import (  # noqa: E402
    PROPERTIES, ActionType, ProbeWriter, ask_value, bisect_flip, blank_board,
    deed_price, legal, set_buy_decision,
)

SEED = 20250811
BUY = int(ActionType.BUY_PROPERTY)

# deeds whose p01 residual was 0 (no opponent could reach them from Go)
TARGETS = [39, 37, 34, 26, 24]
# how far *behind* the deed to park the opponent
GAPS = list(range(1, 15))


def buys_at(env, sq: int, cash: int) -> bool:
    set_buy_decision(env, 0, sq, cash)
    if BUY not in legal(env, 0):
        return False
    return ask_value(env, 0) == BUY


def main() -> int:
    fields = [
        "square", "name", "price", "base_rent", "opponent_gap",
        "opponent_pos", "flip_cash", "residual", "seed",
    ]
    with ProbeWriter("p01b_rent_residual", fields) as out:
        for sq in TARGETS:
            price = deed_price(sq)
            for gap in GAPS:
                _, env = blank_board(seed=SEED)
                opp_pos = (sq - gap) % 40
                # park all three opponents at the same distance to amplify
                for opid in (1, 2, 3):
                    env.players[opid].position = opp_pos

                flip = bisect_flip(
                    lambda c: buys_at(env, sq, c), price, price + 2500
                )
                residual = (price + 200 - flip) if flip is not None else ""
                out.write(
                    square=sq, name=PROPERTIES[sq]["name"], price=price,
                    base_rent=PROPERTIES[sq]["rent"][0], opponent_gap=gap,
                    opponent_pos=opp_pos, flip_cash=flip, residual=residual,
                    seed=SEED,
                )
                print(f"  sq {sq:>2} gap {gap:>2} (opp@{opp_pos:>2})  "
                      f"flip {flip}  residual {residual}", flush=True)
            print()
    print(f"wrote {out.path} ({out.rows} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
