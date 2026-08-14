"""Experiment 1 — buy threshold curve.

Question
--------
For each of the 28 deeds, on an otherwise empty board, what is the smallest
cash holding at which the teacher buys rather than skips? How does that flip
point relate to the deed's list price?

Method
------
Canonical blank board, all deeds with the bank, all opponents on Go with
$1500. Seat 0 is placed on the target square in post_roll with `has_rolled`
set, so exactly two actions are legal: BUY_PROPERTY and END_TURN. Cash is the
only variable swept. A coarse linear scan first checks that the buy response
is monotone in cash (a non-monotone response would falsify the whole "flip
point" framing); a bisection then locates the exact flip.

Output: probes/p01_buy_threshold.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from competition_agent.probe_harness import (  # noqa: E402
    PROPERTIES, PROPERTY_IDS, ActionType, ProbeWriter, ask_value, blank_board,
    bisect_flip, deed_color, deed_price, legal, set_buy_decision,
)

SEED = 20250811
BUY = int(ActionType.BUY_PROPERTY)
COARSE_STEP = 25
SWEEP_HEADROOM = 2500


def buys_at(env, pid: int, square: int, cash: int) -> bool:
    set_buy_decision(env, pid, square, cash)
    allowed = legal(env, pid)
    if BUY not in allowed:
        return False
    return ask_value(env, pid) == BUY


def main() -> int:
    fields = [
        "square", "name", "color", "price", "mortgage", "base_rent",
        "flip_cash", "flip_minus_price", "flip_over_price",
        "monotone", "coarse_points", "seed",
    ]
    with ProbeWriter("p01_buy_threshold", fields) as out:
        for sq in PROPERTY_IDS:
            price = deed_price(sq)
            _, env = blank_board(seed=SEED)

            lo, hi = price, price + SWEEP_HEADROOM
            coarse = [(c, buys_at(env, 0, sq, c))
                      for c in range(lo, hi + 1, COARSE_STEP)]

            # monotone <=> once buying starts it never stops
            first_true = next((i for i, (_, b) in enumerate(coarse) if b), None)
            monotone = (
                first_true is not None
                and all(b for _, b in coarse[first_true:])
            )

            flip = bisect_flip(lambda c: buys_at(env, 0, sq, c), lo, hi)

            out.write(
                square=sq,
                name=PROPERTIES[sq]["name"],
                color=deed_color(sq),
                price=price,
                mortgage=PROPERTIES[sq]["mortgage"],
                base_rent=PROPERTIES[sq]["rent"][0],
                flip_cash=flip if flip is not None else "",
                flip_minus_price=(flip - price) if flip is not None else "",
                flip_over_price=(
                    round(flip / price, 4) if flip is not None else ""
                ),
                monotone=monotone,
                coarse_points=len(coarse),
                seed=SEED,
            )
            print(f"  sq {sq:>2} {PROPERTIES[sq]['name'][:22]:<22} "
                  f"price {price:>3}  flip {flip}  "
                  f"delta {(flip - price) if flip else '-'}  "
                  f"monotone {monotone}", flush=True)
    print(f"\nwrote {out.path} ({out.rows} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
