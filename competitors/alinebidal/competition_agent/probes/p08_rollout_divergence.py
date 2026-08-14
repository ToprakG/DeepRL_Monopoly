"""Experiment 8 (pulled forward) — when does ASURolloutV1 disagree with ASUValueV1?

Motivation
----------
On ordinary seed-3 play the two variants agreed 10/10 while rollout cost ~460x
more. If lookahead almost never changes the choice, Phase 4 is not worth
building. Ordinary play is the wrong place to look, though: most decisions are
lopsided, so any sane lookahead agrees. This probe therefore constructs states
*near decision boundaries*, where a one-step value and a truncated rollout have
the best chance of disagreeing.

Note on "seeds": both variants are deterministic given a state — the rollout
uses fixed common-random-number streams (published as seeds 0..7). Repeating a
state cannot produce a new answer, so the agreement rate is estimated over a
*population of constructed states*, not over RNG draws.

Categories (as scoped in the follow-up brief)
  a) buy       — cash pinned at the p01 flip point +/- a few dollars
  b) build     — monopoly held, cash swept across the build safety floor
  c) auction   — high bid swept across the bid/pass ceiling
  d) trade     — incoming offer swept across the accept/decline boundary

Output: probes/p08_rollout_divergence.csv
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from competition_agent.proc import managed_pool  # noqa: E402
from competition_agent.probe_harness import (  # noqa: E402
    COLOR_GROUPS, PROPERTY_IDS, ActionType, ProbeWriter, ask_rollout,
    ask_value, blank_board, deed_price, describe, legal, set_auction,
    set_buy_decision, set_pre_roll,
)

SEED = 20250811
BUY = int(ActionType.BUY_PROPERTY)

# flip points measured in p01 (flip = price + 200 on an empty board)
BUY_OFFSETS = [-1, 0, +1, +25]
BUILD_GROUPS = ["brown", "lightblue", "orange", "red", "yellow", "darkblue"]
BUILD_CASH = [150, 200, 220, 260, 300, 400, 600, 900]
AUCTION_DEEDS = [1, 6, 16, 24, 31, 37, 39]
AUCTION_BIDS = [0, 40, 80, 120, 160, 200, 260, 320]
TRADE_CASH_REQ = [0, 50, 100, 150, 200, 300, 400]


def _states():
    """Yield (category, label, builder) for every boundary state."""
    for sq in PROPERTY_IDS:
        for off in BUY_OFFSETS:
            yield ("buy", f"sq{sq}_off{off}", ("buy", sq, off))
    for color in BUILD_GROUPS:
        for cash in BUILD_CASH:
            yield ("build", f"{color}_cash{cash}", ("build", color, cash))
    for sq in AUCTION_DEEDS:
        for bid in AUCTION_BIDS:
            yield ("auction", f"sq{sq}_bid{bid}", ("auction", sq, bid))
    for req in TRADE_CASH_REQ:
        for spare in (5, 15):
            yield ("trade", f"req{req}_rr{spare}", ("trade", req, spare))


def _build(kind, a, b):
    """Materialise one boundary state; returns (env, pid) or None if illegal."""
    from monopoly_game_engine.env import TradeOffer

    _, env = blank_board(seed=SEED)

    if kind == "buy":
        sq, off = a, b
        set_buy_decision(env, 0, sq, deed_price(sq) + 200 + off)
        return env, 0

    if kind == "build":
        color, cash = a, b
        squares = COLOR_GROUPS[color]
        for s in squares:
            prop = env.properties[s]
            prop.owner = 0
            env.players[0].properties.append(prop)
        env._update_monopolies()
        set_pre_roll(env, 0, cash=cash)
        return env, 0

    if kind == "auction":
        sq, bid = a, b
        env.players[0].cash = 800
        set_auction(env, 0, sq, high_bid=bid, high_bidder=1,
                    bidders=[0, 1, 2, 3])
        return env, 0

    if kind == "trade":
        req, spare = a, b
        for s in (16, 18):
            env.properties[s].owner = 0
            env.players[0].properties.append(env.properties[s])
        env.properties[spare].owner = 0
        env.players[0].properties.append(env.properties[spare])
        env.properties[19].owner = 1
        env.players[1].properties.append(env.properties[19])
        env._update_monopolies()
        env.players[0].cash = 800
        env.phase = "out_of_turn"
        env.current_turn_idx = 1
        env.out_of_turn_pids = [0, 2, 3]
        env.pending_trades[1] = TradeOffer(
            from_player=1, to_player=0,
            offered_prop=env.properties[19],
            requested_prop=env.properties[spare],
            cash_offered=0, cash_requested=req,
        )
        return env, 0

    raise ValueError(kind)


def _job(item):
    category, label, spec = item
    env, pid = _build(*spec)
    allowed = legal(env, pid)
    if len(allowed) < 2:
        return None                       # no real choice; not informative

    t0 = time.perf_counter()
    v = ask_value(env, pid)
    t_val = time.perf_counter() - t0

    t0 = time.perf_counter()
    r = ask_rollout(env, pid)
    t_roll = time.perf_counter() - t0

    return {
        "category": category, "label": label, "n_legal": len(allowed),
        "value_action": v, "value_desc": describe(v),
        "rollout_action": r, "rollout_desc": describe(r),
        "agree": v == r,
        "value_seconds": round(t_val, 5),
        "rollout_seconds": round(t_roll, 4),
        "seed": SEED,
    }


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return p, max(0.0, c - m), min(1.0, c + m)


def main() -> int:
    items = list(_states())
    print(f"constructed {len(items)} boundary states")

    with managed_pool(10) as pool:
        rows = [r for r in pool.map(_job, items) if r is not None]

    fields = list(rows[0].keys())
    with ProbeWriter("p08_rollout_divergence", fields) as out:
        for r in rows:
            out.write(**r)

    print(f"\nusable states: {len(rows)}")
    cats = sorted({r["category"] for r in rows})
    print(f"\n{'category':<10} {'n':>5} {'diverge':>8} {'rate':>8} "
          f"{'value s':>9} {'rollout s':>10} {'ratio':>7}")
    print("-" * 62)
    tot_d = 0
    for c in cats:
        sub = [r for r in rows if r["category"] == c]
        d = sum(1 for r in sub if not r["agree"])
        tot_d += d
        vs = sum(r["value_seconds"] for r in sub) / len(sub)
        rs = sum(r["rollout_seconds"] for r in sub) / len(sub)
        print(f"{c:<10} {len(sub):>5} {d:>8} {d/len(sub)*100:>7.1f}% "
              f"{vs:>9.4f} {rs:>10.3f} {rs/max(vs,1e-9):>6.0f}x")
    p, lo, hi = wilson(tot_d, len(rows))
    print("-" * 62)
    print(f"{'ALL':<10} {len(rows):>5} {tot_d:>8} {p*100:>7.1f}%")
    print(f"\noverall divergence rate {p*100:.1f}%  "
          f"95% Wilson CI [{lo*100:.1f}%, {hi*100:.1f}%]")
    print(f"null 'rollout never changes the decision' is "
          f"{'REJECTED' if tot_d > 0 else 'NOT rejected'} "
          f"({tot_d} divergence(s) observed; both policies are deterministic "
          f"given a state, so a single divergence falsifies the null)")
    print(f"\nwrote {out.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
