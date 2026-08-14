"""Part B — mine the ranker's errors instead of guessing features.

Every previous feature added to the trade scorer was chosen because it sounded
relevant, and D2.6 records what that cost: the monopoly term decided the whole
ordering and decided it wrong. So this does not propose features. It replays
the hand-fitted ranker over the harvested corpus, isolates the states where it
puts something else on top of the teacher's actual choice, and reports what
separates the two picks.

Three outputs, in increasing order of usefulness:

1. **Where the teacher's pick lands in our ranking.** If it is usually rank 2
   the scorer is nearly right and the fix is a tie-break; if it is rank 30 the
   scorer is blind to whatever drives the choice.

2. **Score decomposition.** The scorer is linear in six weighted terms, so the
   gap between our pick and the teacher's decomposes exactly into six numbers.
   The term with the largest mean contribution to the gap is the one actively
   steering us wrong — as opposed to merely being absent.

3. **Side-by-side distributions on quantities the scorer does NOT use.** A
   feature worth adding has to separate the two picks on the disagreements. If
   it does not, it cannot fix them, however plausible it sounds.

The corpus is the same one Candidate D trained on, so a feature surfaced here
is testable by the ranker without new data collection.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from competition_agent.spec_policy import TRADE_W  # noqa: E402

PROBES = Path(__file__).resolve().parent / "probes"


def terms(c) -> dict:
    """The six weighted terms of the fitted scorer, evaluated per candidate.

    Mirrors `SpecPolicy._propose_trade` exactly; `harvest_trades.deed_facts`
    records the same quantities under different names, so this is a replay and
    not a reimplementation.
    """
    r, o = c["req"], c["off"]
    return {
        "d_rent": TRADE_W["d_rent"] * (r["rent_if_ours"] - o["rent_if_ours"]),
        "d_price": TRADE_W["d_price"] * ((r["price"] - o["price"]) / 100.0),
        "completes": TRADE_W["completes"] * (
            1.0 if r["ours_in_group"] == r["group_size"] - 1 else 0.0),
        "d_ours": TRADE_W["d_ours"] * (r["ours_in_group"] - o["ours_in_group"]),
        "off_mort": TRADE_W["off_mort"] * (1.0 if o["mortgaged"] else 0.0),
        "d_houses": TRADE_W["d_houses"] * (r["houses"] - o["houses"]),
    }


def score(c) -> float:
    return sum(terms(c).values())


def opener(p: Path):
    if p.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(p, "rb"))
    return p.open()


def records(paths):
    for p in paths:
        with opener(p) as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


# --------------------------------------------------------------------------
# quantities the scorer does NOT use — candidate features for the fix
# --------------------------------------------------------------------------
def unseen(c) -> dict:
    r, o = c["req"], c["off"]
    return {
        # Does the deed we hand over complete a group for someone else? The
        # scorer cannot express this at all: it only counts OUR holdings.
        "off_completes_theirs": float(
            o["theirs_in_group"] == o["group_size"] - 1),
        "off_theirs_in_group": float(o["theirs_in_group"]),
        "req_theirs_in_group": float(r["theirs_in_group"]),
        # How close the group we are asking about is to being contested.
        "req_group_size": float(r["group_size"]),
        "off_group_size": float(o["group_size"]),
        # Absolute rent levels, as opposed to the difference the scorer takes.
        "req_base_rent": float(r["base_rent"]),
        "off_base_rent": float(o["base_rent"]),
        "req_rent_if_ours": float(r["rent_if_ours"]),
        "off_rent_if_ours": float(o["rent_if_ours"]),
        # Mortgage status of the deed we ask for (the scorer only sees the
        # offered side's mortgage flag).
        "req_mortgaged": float(r["mortgaged"]),
        # Development already on the board.
        "req_houses": float(r["houses"]),
        "off_houses": float(o["houses"]),
        "req_is_monopoly": float(r["is_monopoly"]),
        "off_is_monopoly": float(o["is_monopoly"]),
        # Prices in absolute terms.
        "req_price": float(r["price"]),
        "off_price": float(o["price"]),
        # Does the deed we give away break a group we were assembling?
        "off_breaks_ours": float(o["ours_in_group"] >= 2),
        "req_completes_ours": float(
            r["ours_in_group"] == r["group_size"] - 1),
        # Balance of the exchange. The scorer has a `d_price` term but its
        # fitted weight is -0.0155 per $100, so in practice it is blind to
        # whether the two sides are comparable. A proposal the counterparty
        # will decline is worth nothing, and the scorer has no notion of that.
        "price_gap_req_minus_off": float(r["price"] - o["price"]),
        "abs_price_gap": float(abs(r["price"] - o["price"])),
        "rent_gap_req_minus_off": float(r["rent_if_ours"] - o["rent_if_ours"]),
        "abs_rent_gap": float(abs(r["rent_if_ours"] - o["rent_if_ours"])),
        # A swap where both sides move a group forward, as opposed to a
        # one-sided grab.
        "mutual_swap": float(o["theirs_in_group"] == o["group_size"] - 1
                             and r["ours_in_group"] == r["group_size"] - 1),
    }


def mean_sd(xs):
    n = len(xs)
    if not n:
        return 0.0, 0.0
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / n
    return m, math.sqrt(v)


def cohen_d(a, b):
    ma, sa = mean_sd(a)
    mb, sb = mean_sd(b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    sp = math.sqrt(((na - 1) * sa * sa + (nb - 1) * sb * sb) / (na + nb - 2))
    return (mb - ma) / sp if sp > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default=None,
                    help="harvest file(s); glob allowed. Defaults to the "
                         "Candidate D corpus, then any sharded corpus.")
    ap.add_argument("--top", type=int, default=12,
                    help="how many separating features to print")
    args = ap.parse_args()

    if args.src:
        paths = sorted(Path().glob(args.src)) or [Path(args.src)]
    else:
        single = PROBES / "trade_harvest.jsonl"
        shards = sorted((PROBES / "trade_shards").glob("*.jsonl.gz"))
        paths = shards if shards else [single]
    paths = [p for p in paths if p.exists()]
    if not paths:
        print("no harvest found")
        return 1
    print(f"source: {len(paths)} file(s), "
          f"{sum(p.stat().st_size for p in paths) / 1e6:.0f} MB")

    n_states = n_agree = 0
    rank_hist = Counter()
    gap_terms = Counter()
    ours_vals, theirs_vals, pop_vals = {}, {}, {}
    margin = []
    n_cands_agree, n_cands_dis = [], []
    seeds = set()

    for rec in records(paths):
        if not rec.get("proposed"):
            continue
        cands = rec["cands"]
        if len(cands) < 2:
            continue
        chosen = rec["chosen"]
        scored = sorted(((score(c), i) for i, c in enumerate(cands)),
                        reverse=True)
        tgt = next((i for i, c in enumerate(cands) if c["a"] == chosen), None)
        if tgt is None:
            continue
        n_states += 1
        seeds.add(rec["seed"])
        our_i = scored[0][1]
        rank = next(k for k, (_, i) in enumerate(scored, 1) if i == tgt)
        rank_hist[min(rank, 21)] += 1
        if our_i == tgt:
            n_agree += 1
            n_cands_agree.append(len(cands))
            continue

        n_cands_dis.append(len(cands))
        margin.append(scored[0][0] - score(cands[tgt]))
        to, tt = terms(cands[our_i]), terms(cands[tgt])
        for k in to:
            gap_terms[k] += to[k] - tt[k]
        uo, ut = unseen(cands[our_i]), unseen(cands[tgt])
        for k in uo:
            ours_vals.setdefault(k, []).append(uo[k])
            theirs_vals.setdefault(k, []).append(ut[k])
        # The population mean over every candidate in the same states. Without
        # it the table cannot say which of the two picks is the unusual one:
        # our pick is an argmax and is extreme by construction, so a large
        # difference could be our scorer selecting an outlier rather than the
        # teacher preferring one.
        for c in cands:
            uc = unseen(c)
            for k in uc:
                pop_vals.setdefault(k, []).append(uc[k])

    n_dis = n_states - n_agree
    print(f"\nproposals replayed : {n_states}   game seeds {len(seeds)}")
    print(f"fitted ranker top-1: {n_agree} "
          f"({100 * n_agree / max(n_states, 1):.2f}%)")
    print(f"disagreements      : {n_dis}")
    ma, _ = mean_sd(n_cands_agree)
    md, _ = mean_sd(n_cands_dis)
    print(f"mean candidates    : {ma:.1f} when we agree, "
          f"{md:.1f} when we do not")

    # ---- 1. where the teacher's pick sits in our ranking ----------------
    print("\n=== 1. rank of the teacher's pick under the fitted scorer ===")
    cum = 0
    for r in sorted(rank_hist):
        cum += rank_hist[r]
        label = f"{r}" if r < 21 else "21+"
        print(f"  rank {label:>3}  {rank_hist[r]:>5}  "
              f"({100 * rank_hist[r] / n_states:5.2f}%)   "
              f"cumulative {100 * cum / n_states:5.2f}%")

    # ---- 2. which term steers us wrong ---------------------------------
    print("\n=== 2. score gap decomposition (our pick - teacher's pick) ===")
    print("    positive = this term is why we preferred ours")
    mm, _ = mean_sd(margin)
    print(f"    mean total gap {mm:.3f}  over {n_dis} disagreements\n")
    for k, v in sorted(gap_terms.items(), key=lambda kv: -abs(kv[1])):
        per = v / max(n_dis, 1)
        print(f"  {k:<12}{per:+8.3f}   "
              f"{100 * per / mm if mm else 0:+6.1f}% of the gap")

    # ---- 3. what separates the two picks, among unused quantities ------
    print("\n=== 3. quantities the scorer does not use ===")
    print(f"  {'feature':<26}{'our pick':>11}{'teacher':>11}"
          f"{'all cands':>11}{'diff':>9}{'Cohen d':>9}")
    print("  " + "-" * 78)
    rows = []
    for k in ours_vals:
        a, b = ours_vals[k], theirs_vals[k]
        ma_, _ = mean_sd(a)
        mb_, _ = mean_sd(b)
        mp_, _ = mean_sd(pop_vals.get(k, []))
        rows.append((abs(cohen_d(a, b)), k, ma_, mb_, mp_, cohen_d(a, b)))
    for _, k, ma_, mb_, mp_, d in sorted(rows, reverse=True)[:args.top]:
        print(f"  {k:<26}{ma_:>11.2f}{mb_:>11.2f}{mp_:>11.2f}"
              f"{mb_ - ma_:>+9.2f}{d:>+9.3f}")
    print("\n  Cohen d is teacher-minus-ours in pooled SD units. |d| < 0.2 is "
          "noise;\n  a feature that does not separate the picks cannot fix "
          "the disagreements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
