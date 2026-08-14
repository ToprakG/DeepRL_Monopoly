"""Phase 2 acceptance: does the spec policy pick what the teacher picks?

Two measurements, because they answer different questions:

  held-out  — replay fresh seeded games driven by the TEACHER, and at every
              seat-0 decision ask the clone what it would have done. This is
              off-policy: the state distribution is the teacher's own, so it
              measures rule fidelity without letting the clone's mistakes
              compound.

  on-policy — let the CLONE drive, and ask the teacher what it would have done
              at each state the clone reaches. Harder and more honest: a clone
              that drifts into states it handles badly is penalised here and
              not off-policy.

Seeds used here are disjoint from every probe seed (probes used 20250811),
so this is genuinely held out.

Disagreements are logged by action family, because an aggregate percentage
does not tell you which rule to fix.
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ASU_FROZEN_TEACHER import ASUValueV1  # noqa: E402
from ASU_FROZEN_TEACHER.evaluate import _ScriptedAdapter, _new_seeded_game  # noqa: E402
from monopoly_game_engine.actions import OFFSETS, action_to_description  # noqa: E402
from monopoly_game_engine.agents_fixed import FP_AGENT_CLASSES  # noqa: E402

from competition_agent.proc import managed_pool  # noqa: E402
from competition_agent.spec_policy import SpecPolicy  # noqa: E402

PROBE_DIR = Path(__file__).resolve().parent / "probes"


def family(action: int) -> str:
    """Which action family an id belongs to — the unit we debug in.

    Binary actions are reported individually rather than lumped together:
    PAY_BAIL and ROLL_DICE are different decisions with different rules, and
    collapsing them hides which rule is failing.
    """
    best = "binary"
    for name, start in sorted(OFFSETS.items(), key=lambda kv: kv[1]):
        if action >= start:
            best = name
    if best == "binary":
        return action_to_description(action)
    return best


# Every family the policy can be asked about, so the report shows the ones
# that never came up as explicit zero rows instead of silently omitting them.
ALL_FAMILIES = [
    "DO_NOTHING", "END_TURN", "ROLL_DICE", "BUY_PROPERTY", "USE_GOOJ_CARD",
    "PAY_BAIL", "DECLARE_BANKRUPT", "ACCEPT_TRADE", "DECLINE_TRADE",
    "mortgage", "unmortgage", "improve_house", "improve_hotel",
    "sell_house", "sell_hotel", "sell_prop",
    "buy_trade", "sell_trade", "exch_trade", "auction",
]

# Semantic grouping, so "development order" and "liquidation order" can be
# read as single numbers even though each spans several action families.
CATEGORIES = [
    ("turn flow",        ["DO_NOTHING", "END_TURN", "ROLL_DICE"]),
    ("buy",              ["BUY_PROPERTY"]),
    ("auction",          ["auction"]),
    ("jail",             ["USE_GOOJ_CARD", "PAY_BAIL"]),
    ("development order", ["improve_house", "improve_hotel"]),
    ("liquidation order", ["mortgage", "sell_house", "sell_hotel",
                           "sell_prop", "DECLARE_BANKRUPT"]),
    ("unmortgage",       ["unmortgage"]),
    ("trade reply",      ["ACCEPT_TRADE", "DECLINE_TRADE"]),
    ("trade proposal",   ["buy_trade", "sell_trade", "exch_trade"]),
]


def _run_seed(args):
    seed, mode, max_steps = args
    game = _new_seeded_game(seed)
    env = game.env
    teacher = ASUValueV1(0)
    clone = SpecPolicy(0)
    opp = {i: _ScriptedAdapter(FP_AGENT_CLASSES[i - 1](i), i) for i in (1, 2, 3)}

    rows, steps = [], 0
    while steps < max_steps and not env.done:
        actor = env.whose_turn()
        if actor == 0:
            t = int(teacher.choose_action(env))
            try:
                c = int(clone.choose_action(env))
            except Exception as exc:                       # noqa: BLE001
                c = -1
                rows.append((seed, "ERROR", f"{type(exc).__name__}: {exc}",
                             "", False))
                env.done = True
                break
            legal = {int(a) for a in env.get_allowed_actions(0)}
            rows.append((
                seed, family(t), action_to_description(t),
                action_to_description(c), t == c,
            ))
            action = t if mode == "heldout" else c
            if action not in legal:            # clone emitted an illegal move
                rows[-1] = (seed, "ILLEGAL", action_to_description(t),
                            action_to_description(c), False)
                action = t
        else:
            action = int(opp[actor].choose_action(env))
        game.step(action)
        steps += 1
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["heldout", "onpolicy"],
                    default="heldout")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seed-base", type=int, default=900000,
                    help="disjoint from probe seeds")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--target", type=int, default=1000,
                    help="stop once this many decisions are collected")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    jobs = [(args.seed_base + k, args.mode, args.max_steps)
            for k in range(args.seeds)]
    with managed_pool(args.workers) as pool:
        batches = pool.map(_run_seed, jobs)

    rows = [r for b in batches for r in b][: args.target * 4]
    if not rows:
        print("no decisions collected")
        return 1

    total = len(rows)
    agree = sum(1 for r in rows if r[4])
    by_fam = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        by_fam[r[1]][0] += int(r[4])
        by_fam[r[1]][1] += 1

    out = PROBE_DIR / f"agreement_{args.mode}.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "family", "teacher", "clone", "agree"])
        w.writerows(rows)

    print(f"mode={args.mode}  seeds={args.seed_base}.."
          f"{args.seed_base + args.seeds - 1}  decisions={total}")

    print(f"\n=== per action family (every family, including unseen) ===")
    print(f"{'family':<20}{'agree':>8}{'n':>8}{'rate':>9}   {'share':>7}")
    print("-" * 56)
    listed = set()
    for fam in ALL_FAMILIES:
        a, n = by_fam.get(fam, [0, 0])
        listed.add(fam)
        rate = f"{100*a/n:>7.1f}%" if n else "      —"
        share = f"{100*n/total:>6.1f}%" if n else "     —"
        print(f"{fam:<20}{a:>8}{n:>8}{rate}   {share}")
    extra = [f for f in by_fam if f not in listed]
    for fam in sorted(extra):
        a, n = by_fam[fam]
        print(f"{fam+' *':<20}{a:>8}{n:>8}{100*a/n:>7.1f}%   "
              f"{100*n/total:>6.1f}%")
    if extra:
        print("  * not a normal action family (ERROR / ILLEGAL markers)")

    print(f"\n=== by decision category ===")
    print(f"{'category':<20}{'agree':>8}{'n':>8}{'rate':>9}   {'share':>7}")
    print("-" * 56)
    for label, fams in CATEGORIES:
        a = sum(by_fam.get(f, [0, 0])[0] for f in fams)
        n = sum(by_fam.get(f, [0, 0])[1] for f in fams)
        rate = f"{100*a/n:>7.1f}%" if n else "      —"
        share = f"{100*n/total:>6.1f}%" if n else "     —"
        print(f"{label:<20}{a:>8}{n:>8}{rate}   {share}")
    print("-" * 56)
    print(f"{'TOTAL':<20}{agree:>8}{total:>8}{100*agree/total:>7.1f}%   "
          f"{100.0:>6.1f}%")

    # What a family costs us: its share of all disagreements.
    print(f"\n=== disagreement budget (where the missing % lives) ===")
    losses = sorted(((n - a, f) for f, (a, n) in by_fam.items() if n - a),
                    reverse=True)
    miss = total - agree
    for lost, fam in losses:
        print(f"{fam:<20}{lost:>8} lost   {100*lost/miss:>5.1f}% of all "
              f"disagreement   {100*lost/total:>5.1f}pp of total")

    target = 90 if args.mode == "heldout" else 85
    verdict = "PASS" if 100 * agree / total >= target else "FAIL"
    print(f"\nPhase 2 target for {args.mode}: >={target}%  -> {verdict}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
