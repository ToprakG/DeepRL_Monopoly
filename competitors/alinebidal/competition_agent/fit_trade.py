"""Fit the two trade defects on p09b, with an honest train/held-out split.

Defect 1 (gate)    — the teacher proposes on only 118/400 boards where a legal
                     exchange exists. Ours fires nearly always.
Defect 2 (ranking) — our top-1 over the candidate set is 11%.

Protocol
--------
The split is by **board**, fixed before any search, 60/40. Every weight is
chosen on train only; the held-out number is computed once per reported
configuration and never optimised against. This matters because the search
space is small and 118 positive boards is not many — it would be very easy to
tune to the whole file and report a number that means nothing.

Held-out *agreement in real play* stays a third, independent check: this file
never touches it.

Features are the components `deed_value` already computes, differenced between
the requested and the offered deed:

    price      list price
    rent       projected rent to us, opponents' real positions (SPEC A4/A6)
    mono       marginal group value (SPEC B5)
    mortgaged  1 if the deed is mortgaged

score(candidate) = w_price*dprice + w_rent*drent + w_mono*dmono + w_mort*dmort

Usage:  python3 competition_agent/fit_trade.py [--iters 4000]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from competition_agent.proc import managed_pool  # noqa: E402
from competition_agent.probe_harness import ask_value, legal  # noqa: E402
from competition_agent.probes.p09b_trade_ranking import (  # noqa: E402
    SEED, N_BOARDS, _decode, build_board,
)
from competition_agent.spec_model import (  # noqa: E402
    SHORT_TURNS, marginal_monopoly_value, multi_turn_landings, rent_for,
)
from monopoly_game_engine.actions import OFFSETS  # noqa: E402
from monopoly_game_engine.constants import PROPERTIES, PROPERTY_IDS  # noqa: E402

CACHE = Path(__file__).resolve().parent / "probes" / "fit_trade_features.json"
FEATURES = ("price", "rent", "mono", "mortgaged")


def _deed_features(env, pid, sq):
    prop = env.properties[sq]
    saved = prop.owner
    prop.owner = pid
    try:
        rent = 0.0
        for opp in env.players:
            if opp.player_id == pid or opp.bankrupt:
                continue
            for land, p in multi_turn_landings(opp.position, SHORT_TURNS):
                if land == sq:
                    rent += p * rent_for(env, sq)
    finally:
        prop.owner = saved
    return {
        "price": float(PROPERTIES[sq]["price"]),
        "rent": rent,
        "mono": marginal_monopoly_value(env, pid, sq),
        "mortgaged": 1.0 if prop.mortgaged else 0.0,
    }


def _extract(k: int):
    rng = random.Random(SEED + k)
    env, rival, dev = build_board(rng)
    n = len(PROPERTY_IDS)
    others = [i for i in range(len(env.players)) if i != 0]
    cands = [a for a in legal(env, 0)
             if OFFSETS["exch_trade"] <= a < OFFSETS["auction"]]
    if len(cands) < 2:
        return None

    chosen = int(ask_value(env, 0))
    cache = {}

    def feat(sq):
        if sq not in cache:
            cache[sq] = _deed_features(env, 0, sq)
        return cache[sq]

    rows = []
    for a in cands:
        dec = _decode(a, n, others)
        if dec is None:
            continue
        _, off, req = dec
        fo, fr = feat(off), feat(req)
        rows.append({
            "action": a,
            "d": {f: fr[f] - fo[f] for f in FEATURES},
        })
    if not rows:
        return None
    return {"board": k, "chosen": chosen,
            "proposed": chosen in cands, "cands": rows}


def score(d, w):
    return sum(w[f] * d[f] for f in FEATURES)


def top1(boards, w):
    hit = 0
    for b in boards:
        best = max(b["cands"], key=lambda c: (score(c["d"], w), -c["action"]))
        hit += best["action"] == b["chosen"]
    return hit / len(boards) if boards else 0.0


def gate_acc(boards, w, thresh):
    """Predict 'propose' iff the best candidate scores above `thresh`."""
    ok = 0
    for b in boards:
        best = max(score(c["d"], w) for c in b["cands"])
        ok += (best > thresh) == b["proposed"]
    return ok / len(boards) if boards else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    if CACHE.exists() and not args.rebuild:
        data = json.loads(CACHE.read_text())
    else:
        with managed_pool(10) as pool:
            data = [d for d in pool.map(_extract, range(N_BOARDS)) if d]
        CACHE.write_text(json.dumps(data))
    print(f"boards with candidates: {len(data)}")

    # ---- split fixed BEFORE any search, by board id -----------------------
    rng = random.Random(20250811)
    ids = sorted(d["board"] for d in data)
    rng.shuffle(ids)
    cut = int(0.6 * len(ids))
    train_ids, held_ids = set(ids[:cut]), set(ids[cut:])
    train = [d for d in data if d["board"] in train_ids]
    held = [d for d in data if d["board"] in held_ids]
    tr_pos = [d for d in train if d["proposed"]]
    hd_pos = [d for d in held if d["proposed"]]
    print(f"train {len(train)} boards ({len(tr_pos)} proposals)   "
          f"held-out {len(held)} boards ({len(hd_pos)} proposals)")

    baseline = {"price": 1.0, "rent": 1.0, "mono": 1.0, "mortgaged": 0.0}
    print(f"\nbaseline (current deed_value weights)")
    print(f"  train  top-1 {100*top1(tr_pos, baseline):5.1f}%   "
          f"held-out top-1 {100*top1(hd_pos, baseline):5.1f}%")

    # ---- defect 2: ranking weights, searched on TRAIN only ----------------
    best_w, best_s = dict(baseline), top1(tr_pos, baseline)
    srng = random.Random(7)
    for i in range(args.iters):
        if i < args.iters // 2:
            w = {f: srng.uniform(-2, 3) for f in FEATURES}
        else:
            w = {f: best_w[f] + srng.gauss(0, 0.25) for f in FEATURES}
        s = top1(tr_pos, w)
        if s > best_s:
            best_w, best_s = w, s

    print(f"\ndefect 2 — ranking weights fitted on train")
    print("  " + "  ".join(f"{f}={best_w[f]:+.3f}" for f in FEATURES))
    print(f"  train  top-1 {100*best_s:5.1f}%   "
          f"held-out top-1 {100*top1(hd_pos, best_w):5.1f}%")

    # ---- defect 1: propose/don't gate, threshold fitted on TRAIN ----------
    scores = sorted({round(max(score(c["d"], best_w) for c in b["cands"]), 3)
                     for b in train})
    best_t, best_g = 0.0, 0.0
    for t in scores:
        g = gate_acc(train, best_w, t)
        if g > best_g:
            best_t, best_g = t, g
    always = sum(1 for b in train if b["proposed"]) / len(train)
    print(f"\ndefect 1 — propose/don't gate")
    print(f"  always-propose baseline : train {100*always:5.1f}%")
    print(f"  never-propose  baseline : train {100*(1-always):5.1f}%")
    print(f"  fitted threshold {best_t:.2f}: train {100*best_g:5.1f}%   "
          f"held-out {100*gate_acc(held, best_w, best_t):5.1f}%")

    out = {"weights": best_w, "threshold": best_t,
           "train_top1": best_s, "held_top1": top1(hd_pos, best_w),
           "train_gate": best_g, "held_gate": gate_acc(held, best_w, best_t),
           "n_train": len(train), "n_held": len(held)}
    (CACHE.parent / "fit_trade_result.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {CACHE.parent / 'fit_trade_result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
