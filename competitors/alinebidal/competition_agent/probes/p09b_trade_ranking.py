"""Experiment 9b — calibrate deed valuation on RANKING, across diverse boards.

What went wrong twice
---------------------
`deed_value` is accurate enough for *threshold* decisions, where it is only
compared against a cash gate: buy agrees 98.4%, auction 90.5%. It is not
accurate enough to *rank two deeds against each other*, which is what a trade
proposal needs — `exch_trade` agrees 0.3% over 722 held-out decisions.

Two earlier attempts regressed because each was designed from a narrow sample:

  * p08's trade cell (14 states, one board) reported 0% rollout divergence.
    Experiment 6, on a wider population, measured 92.6% — D1.3.
  * p09 (80 states, one board shape) showed 36/36 proposals were "offer the
    cheapest spare for the group-completing deed". Held-out play refuted it
    flatly: agreement on the requested deed alone was 27/189.

**Board diversity is therefore a design requirement of this probe, not a nice
to have.** A preference measured on one board configuration has twice now
failed to generalise, and both times the narrow result looked unambiguous.

Design
------
Sample boards from a seeded generator that randomises, independently:

  * which deeds seat 0 holds and which each opponent holds (deed-level, so
    part-groups, whole groups and scattered holdings all occur);
  * every player's board position;
  * development level on completed groups, and bank house/hotel stock;
  * mortgage flags;
  * every player's cash across the safety-gate range.

Each board is then reduced so the exchange menu is small and readable — seat 0
and one rival keep only a handful of tradeable deeds — and the teacher is
asked once. Its pick is a revealed preference over the whole candidate set,
not a yes/no on one pair.

Recorded per observation: the full candidate list, the teacher's choice, and
**the rank our current `deed_value` model assigns to that choice**. Top-1
accuracy and mean rank are the calibration targets; the per-deed residuals are
what a corrected valuation gets fitted to.

Diversity is reported, not assumed: the summary prints how many distinct
colour groups, holding sizes, development levels and position spreads the
sample actually covered, so a future reader can see this is not one board.

Output: probes/p09b_trade_ranking.csv
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from competition_agent.proc import managed_pool  # noqa: E402
from competition_agent.probe_harness import (  # noqa: E402
    COLOR_GROUPS, PROPERTIES, PROPERTY_IDS, ProbeWriter, ask_value,
    blank_board, describe, legal, set_pre_roll,
)
from competition_agent.spec_model import deed_value  # noqa: E402
from monopoly_game_engine.actions import OFFSETS  # noqa: E402

SEED = 770001
N_BOARDS = 400
MAX_OWN = 5
MAX_RIVAL = 5


def _decode(action: int, n: int, others):
    loc = action - OFFSETS["exch_trade"]
    p_idx = loc // (n * (n - 1))
    rem = loc % (n * (n - 1))
    off_idx = rem // (n - 1)
    req_raw = rem % (n - 1)
    req_idx = req_raw if req_raw < off_idx else req_raw + 1
    if p_idx >= len(others):
        return None
    return others[p_idx], PROPERTY_IDS[off_idx], PROPERTY_IDS[req_idx]


def build_board(rng: random.Random):
    """One randomly configured board. Diversity lives here."""
    _, env = blank_board(seed=SEED)

    deeds = list(PROPERTY_IDS)
    rng.shuffle(deeds)

    n_own = rng.randint(2, MAX_OWN)
    n_rival = rng.randint(2, MAX_RIVAL)
    ours = deeds[:n_own]
    theirs = deeds[n_own:n_own + n_rival]
    rival = rng.choice([1, 2, 3])

    # a third party sometimes holds stock too, so the board is not two-sided
    third = deeds[n_own + n_rival:n_own + n_rival + rng.randint(0, 4)]
    third_pid = rng.choice([p for p in (1, 2, 3) if p != rival])

    for sq in ours:
        env.properties[sq].owner = 0
        env.players[0].properties.append(env.properties[sq])
    for sq in theirs:
        env.properties[sq].owner = rival
        env.players[rival].properties.append(env.properties[sq])
    for sq in third:
        env.properties[sq].owner = third_pid
        env.players[third_pid].properties.append(env.properties[sq])
    env._update_monopolies()

    # development on any group that came out complete
    dev_level = 0
    for squares in COLOR_GROUPS.values():
        if all(env.properties[s].is_monopoly for s in squares):
            if rng.random() < 0.5:
                h = rng.randint(1, 4)
                dev_level = max(dev_level, h)
                for s in squares:
                    if env.properties[s].is_real_estate:
                        env.properties[s].houses = h

    # mortgages
    for sq in ours + theirs:
        if rng.random() < 0.15 and env.properties[sq].houses == 0:
            env.properties[sq].mortgaged = True

    for p in env.players:
        p.position = rng.randrange(40)
        p.cash = rng.choice([150, 250, 400, 700, 1200, 2000, 3500])

    env.houses_available = rng.choice([32, 32, 16, 6, 2])
    env.hotels_available = rng.choice([12, 12, 5, 1, 0])
    set_pre_roll(env, 0, cash=env.players[0].cash)
    return env, rival, dev_level


def _job(k: int):
    rng = random.Random(SEED + k)
    env, rival, dev = build_board(rng)
    n = len(PROPERTY_IDS)
    others = [i for i in range(len(env.players)) if i != 0]

    lg = legal(env, 0)
    cands = [a for a in lg
             if OFFSETS["exch_trade"] <= a < OFFSETS["auction"]]
    if len(cands) < 2:
        return None                    # no ranking to reveal

    chosen = ask_value(env, 0)
    proposed = chosen in cands

    # our model's ranking over the same candidate set
    ourv, theirv = {}, {}

    def v(player, sq, cache):
        key = (player, sq)
        if key not in cache:
            cache[key] = deed_value(env, player, sq)
        return cache[key]

    scored = []
    for a in cands:
        dec = _decode(a, n, others)
        if dec is None:
            continue
        tgt, off, req = dec
        gain = v(0, req, ourv) - v(0, off, ourv)
        scored.append((gain, a, tgt, off, req))
    scored.sort(key=lambda t: (-t[0], t[1]))
    order = [s[1] for s in scored]

    rank = order.index(chosen) + 1 if chosen in order else ""
    top = scored[0] if scored else None
    dec = _decode(chosen, n, others) if proposed else None

    owned_groups = sum(
        1 for squares in COLOR_GROUPS.values()
        if all(env.properties[s].owner == 0 for s in squares))

    return {
        "board": k,
        "n_candidates": len(cands),
        "our_deeds": len(env.players[0].properties),
        "rival_deeds": len(env.players[rival].properties),
        "our_full_groups": owned_groups,
        "dev_level": dev,
        "houses_avail": env.houses_available,
        "our_cash": env.players[0].cash,
        "proposed": proposed,
        "teacher_action": describe(chosen),
        "teacher_target": "" if not dec else dec[0],
        "teacher_offer": "" if not dec else dec[1],
        "teacher_request": "" if not dec else dec[2],
        "teacher_offer_color": "" if not dec else PROPERTIES[dec[1]]["color"],
        "teacher_req_color": "" if not dec else PROPERTIES[dec[2]]["color"],
        "model_rank_of_teacher_choice": rank,
        "model_top_action": "" if not top else describe(top[1]),
        "model_top_offer": "" if not top else top[3],
        "model_top_request": "" if not top else top[4],
        "model_top1_correct": bool(top and proposed and top[1] == chosen),
        "seed": SEED + k,
    }


def main() -> int:
    with managed_pool(10) as pool:
        rows = [r for r in pool.map(_job, range(N_BOARDS)) if r]

    with ProbeWriter("p09b_trade_ranking", list(rows[0].keys())) as out:
        for r in rows:
            out.write(**r)

    prop = [r for r in rows if r["proposed"]]
    top1 = sum(1 for r in prop if r["model_top1_correct"])
    ranks = [int(r["model_rank_of_teacher_choice"]) for r in prop
             if r["model_rank_of_teacher_choice"]]

    print(f"boards with a real ranking choice: {len(rows)}")
    print(f"  teacher proposed a trade:        {len(prop)}")
    print(f"  teacher ended turn instead:      {len(rows) - len(prop)}")
    print(f"\nmodel top-1 accuracy: {top1}/{len(prop)} = "
          f"{100*top1/max(len(prop),1):.1f}%")
    if ranks:
        ranks.sort()
        print(f"model rank of the teacher's pick: "
              f"median {ranks[len(ranks)//2]}, mean {sum(ranks)/len(ranks):.1f}, "
              f"worst {ranks[-1]}")
        for k in (1, 3, 5, 10):
            hit = sum(1 for r in ranks if r <= k)
            print(f"  teacher's pick in our top-{k:<2}: "
                  f"{hit}/{len(ranks)} = {100*hit/len(ranks):.1f}%")

    print(f"\n=== diversity actually achieved (not assumed) ===")
    def spread(field):
        vals = sorted({r[field] for r in rows})
        return f"{len(vals)} distinct: {vals[:9]}{'...' if len(vals) > 9 else ''}"
    for f in ("our_deeds", "rival_deeds", "our_full_groups", "dev_level",
              "houses_avail", "our_cash", "n_candidates"):
        print(f"  {f:<20} {spread(f)}")
    colors = sorted({r["teacher_offer_color"] for r in prop if r["teacher_offer_color"]})
    print(f"  offered colours      {len(colors)}: {colors}")
    colors = sorted({r["teacher_req_color"] for r in prop if r["teacher_req_color"]})
    print(f"  requested colours    {len(colors)}: {colors}")
    print(f"\nwrote {out.path} ({out.rows} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
