"""Experiment 2 — auction ceiling curve and the group-completion premium.

Question
--------
What is the most the teacher will pay for each deed at auction, and how does
that ceiling move when it already holds part of the deed's colour group?

Method
------
In the auction phase the legal actions are PASS and the four bid increments
(+1, +10, +50, +100), where an increment is legal only if
`high_bid + increment <= cash`. That gives a clean way to locate the ceiling
without reading any internal value: sweep the standing `high_bid` B and find
the smallest B at which the teacher passes. Bidding +1 costs B+1, so the
teacher passes exactly when B+1 exceeds its ceiling; the smallest passing B is
therefore the ceiling itself.

Cash is held at $5,000 so the safety gates do not bind and the ceiling
reflects deed value alone — the interaction with safety is Experiment 3's job.

The sweep is repeated with 0, 1 and 2 of the group's other deeds already
owned, which exposes the group-completion premium and the discount applied per
still-missing deed.

Output: probes/p02_auction_ceiling.csv
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from competition_agent.proc import managed_pool  # noqa: E402
from competition_agent.probe_harness import (  # noqa: E402
    COLOR_GROUPS, PROPERTIES, PROPERTY_IDS, AuctionAction, ProbeWriter,
    ask_value, blank_board, deed_color, deed_price, describe, give, legal,
    set_auction,
)

SEED = 20250811
PASS = int(AuctionAction.PASS)
CASH = 5000
MAX_BID = 4000


def _passes_at(env, sq, bid, owned):
    """True if the teacher declines to raise when the standing bid is `bid`."""
    _, env = blank_board(seed=SEED)
    env.players[0].cash = CASH
    if owned:
        give(env, 0, owned)
    set_auction(env, 0, sq, high_bid=bid, high_bidder=1, bidders=[0, 1, 2, 3])
    allowed = legal(env, 0)
    if len(allowed) < 2:            # only PASS legal (cash-bound): not a value signal
        return None
    return ask_value(env, 0) == PASS


def _ceiling(sq, owned):
    """Smallest standing bid at which the teacher passes = its ceiling."""
    lo, hi = 0, MAX_BID
    if _passes_at(None, sq, lo, owned) is True:
        return 0
    if _passes_at(None, sq, hi, owned) is not True:
        return None                 # never passes below MAX_BID
    while lo < hi:
        mid = (lo + hi) // 2
        r = _passes_at(None, sq, mid, owned)
        if r is None:
            return None
        if r:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _job(item):
    sq, n_owned = item
    group = [g for g in COLOR_GROUPS[deed_color(sq)] if g != sq]
    if n_owned > len(group):
        return None
    owned = group[:n_owned]
    missing_after = len(group) - n_owned      # still missing once sq is won
    ceil = _ceiling(sq, owned)

    # what does it actually bid from a standing bid of zero?
    _, env = blank_board(seed=SEED)
    env.players[0].cash = CASH
    if owned:
        give(env, 0, owned)
    set_auction(env, 0, sq, high_bid=0, high_bidder=1, bidders=[0, 1, 2, 3])
    opening = ask_value(env, 0)

    price = deed_price(sq)
    return {
        "square": sq,
        "name": PROPERTIES[sq]["name"],
        "color": deed_color(sq),
        "price": price,
        "group_size": len(group) + 1,
        "n_owned_of_group": n_owned,
        "missing_after_win": missing_after,
        "ceiling": ceil if ceil is not None else "",
        "ceiling_over_price": (
            round(ceil / price, 4) if ceil is not None else ""
        ),
        "ceiling_minus_price": (ceil - price) if ceil is not None else "",
        "opening_action": describe(opening),
        "seed": SEED,
    }


def main() -> int:
    items = [(sq, n) for sq in PROPERTY_IDS for n in (0, 1, 2, 3)]
    with managed_pool(10) as pool:
        rows = [r for r in pool.map(_job, items) if r is not None]

    fields = list(rows[0].keys())
    with ProbeWriter("p02_auction_ceiling", fields) as out:
        for r in rows:
            out.write(**r)

    print(f"{'sq':>3} {'name':<22} {'col':<10} {'price':>5} "
          f"{'own':>4} {'miss':>5} {'ceiling':>8} {'ceil/price':>11}  opening")
    print("-" * 96)
    for r in sorted(rows, key=lambda r: (r["color"], r["square"],
                                         r["n_owned_of_group"])):
        print(f"{r['square']:>3} {r['name'][:22]:<22} {r['color']:<10} "
              f"{r['price']:>5} {r['n_owned_of_group']:>4} "
              f"{r['missing_after_win']:>5} {str(r['ceiling']):>8} "
              f"{str(r['ceiling_over_price']):>11}  {r['opening_action']}")

    print(f"\nwrote {out.path} ({out.rows} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
