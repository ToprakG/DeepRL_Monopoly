"""Test D7.3's evidenced feature list, with the two effects separated.

D7.3 found that 90% of the fitted ranker's error comes from two terms it
already has, and that our pick sits at the extreme of the population on every
quantity that separates it from the teacher's. That supports two different
fixes, and they must not be confounded:

  A. **Refit the same six features on this corpus.** The shipped `TRADE_W`
     was fitted on the original harvest, which no longer exists. If simply
     re-fitting recovers most of the gap, the features were never the problem
     and no new ones are needed.
  B. **Add the features D7.3 surfaced.** Only meaningful as a delta over A.

So three numbers are reported on the identical held-out states: the shipped
weights as they stand, A, and B.

Linear, not an MLP. Candidate D put 314 dimensions on 3,589 states and
starved; a linear model on ~16 features cannot, so a gain here is attributable
to the features rather than to capacity. Listwise softmax is kept from
Candidate D because it trains exactly the argmax the policy performs.

Split by game seed, with the same rng seed Candidate D used, so every number in
this file and in `rank_anchor.py` is measured on the same 1,327 states.

Top-1 is DIAGNOSTIC. D2.10, D3.5 and the capture-ratio estimate are three
recorded occasions where an agreement gain did not become a win-rate gain.
Nothing here ships without a measured win rate.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import random
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from competition_agent.spec_policy import TRADE_W  # noqa: E402

PROBES = Path(__file__).resolve().parent / "probes"
OUT = PROBES / "trade_weights_v3.json"

# The six the shipped scorer uses. Order matters: `SHIPPED` indexes into it.
BASE = ("d_rent", "d_price", "completes", "d_ours", "off_mort", "d_houses")
# D7.3's list. `req_theirs` is the denial signal `denial.py` already prices
# for purchases and nothing feeds into trade.
EXTRA = ("off_price", "off_rent", "off_breaks_ours", "req_theirs",
         "off_group_size", "mutual_swap", "req_price", "req_mort",
         "off_completes_theirs", "req_base_rent")


def features(c) -> list:
    r, o = c["req"], c["off"]
    completes = 1.0 if r["ours_in_group"] == r["group_size"] - 1 else 0.0
    off_completes = 1.0 if o["theirs_in_group"] == o["group_size"] - 1 else 0.0
    return [
        # --- BASE, in the shipped scorer's own units so its weights transfer
        r["rent_if_ours"] - o["rent_if_ours"],
        (r["price"] - o["price"]) / 100.0,
        completes,
        float(r["ours_in_group"] - o["ours_in_group"]),
        1.0 if o["mortgaged"] else 0.0,
        float(r["houses"] - o["houses"]),
        # --- EXTRA
        o["price"] / 100.0,
        o["rent_if_ours"] / 10.0,
        1.0 if o["ours_in_group"] >= 2 else 0.0,
        float(r["theirs_in_group"]),
        float(o["group_size"]),
        completes * off_completes,
        r["price"] / 100.0,
        1.0 if r["mortgaged"] else 0.0,
        off_completes,
        r["base_rent"] / 10.0,
    ]


SHIPPED = np.asarray([TRADE_W[k] for k in BASE], np.float32)


def opener(p: Path):
    if p.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(p, "rb"))
    return p.open()


def load(paths):
    """-> list of (seed, feature matrix [n_cands x F], target index)."""
    states = []
    for p in paths:
        with opener(p) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if not r.get("proposed"):
                    continue
                tgt = next((i for i, c in enumerate(r["cands"])
                            if c["a"] == r["chosen"]), -1)
                if tgt < 0 or len(r["cands"]) < 2:
                    continue
                states.append((
                    r["seed"],
                    np.asarray([features(c) for c in r["cands"]], np.float32),
                    tgt))
    return states


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - m), min(1.0, c + m)


def top1_mask(w, states, cols):
    """Per-state correctness under weights `w` restricted to `cols`."""
    return [int(np.argmax(X[:, cols] @ w) == t) for _, X, t in states]


def pack(states, cols, mu=None, sd=None):
    """Flatten to (X, segment id, target row index) for a segment softmax.

    The first version of this file looped one state at a time under Adam and
    was still climbing when it stopped — B read as 'worse' when it was only
    under-trained. Full batch removes that failure mode: LBFGS runs to a
    convergence criterion instead of to an epoch count.
    """
    X = np.concatenate([s[1][:, cols] for s in states]).astype(np.float32)
    if mu is None:
        mu, sd = X.mean(0), X.std(0)
        sd = np.where(sd < 1e-6, 1.0, sd)
    X = (X - mu) / sd
    seg = np.empty(len(X), np.int64)
    tgt = np.empty(len(states), np.int64)
    off = 0
    for i, (_, M, t) in enumerate(states):
        seg[off:off + len(M)] = i
        tgt[i] = off + t
        off += len(M)
    return (torch.from_numpy(X), torch.from_numpy(seg), torch.from_numpy(tgt),
            mu, sd)


def listwise_loss(s, seg, tgt, n_states):
    """-log softmax of the chosen candidate, normalised within its state."""
    m = torch.full((n_states,), -1e30).index_reduce_(
        0, seg, s.detach(), "amax", include_self=False)      # shift constant
    e = torch.exp(s - m[seg])
    tot = torch.zeros(n_states).index_add_(0, seg, e)
    lse = m + torch.log(tot + 1e-30)
    return (lse - s[tgt]).mean()


def acc_of(w, X, seg, tgt, n_states):
    with torch.no_grad():
        s = X @ w
        best = torch.full((n_states,), -1e30).index_reduce_(
            0, seg, s, "amax", include_self=False)
        return float((s[tgt] >= best - 1e-9).float().mean())


def npack(states, cols, mu=None, sd=None):
    """Same flattening as `pack`, in numpy, plus the segment start offsets a
    vectorised per-state argmax needs."""
    X = np.concatenate([s[1][:, cols] for s in states]).astype(np.float32)
    if mu is None:
        mu, sd = X.mean(0), X.std(0)
        sd = np.where(sd < 1e-6, 1.0, sd)
    X = (X - mu) / sd
    starts = np.zeros(len(states), np.int64)
    tgt = np.zeros(len(states), np.int64)
    off = 0
    for i, (_, M, t) in enumerate(states):
        starts[i] = off
        tgt[i] = off + t
        off += len(M)
    return X, starts, tgt, mu, sd


def ntop1(w, X, starts, tgt):
    s = X @ w
    return float(np.mean(s[tgt] >= np.maximum.reduceat(s, starts) - 1e-9))


def search(train, val, held, cols, iters, restarts, tag, init=None,
           seed=20250811):
    """Maximise train top-1 directly, by hill-climbing.

    The shipped `TRADE_W` was produced this way (`fit_trade_v2.py`), and the
    LBFGS fit above optimises log-likelihood instead. Those are different
    objectives, and on this corpus the likelihood optimum scores *worse* top-1
    than the shipped weights — so a likelihood fit cannot be called a refit of
    them. This is the like-for-like comparison.

    Train picks the weights, validation picks between restarts, held-out is
    scored once at the end.
    """
    Xt, sT, tT, mu, sd = npack(train, cols)
    Xv, sV, tV, _, _ = npack(val, cols, mu, sd)
    Xh, sH, tH, _, _ = npack(held, cols, mu, sd)
    rng = np.random.default_rng(seed)
    F = len(cols)
    winners = []
    for r in range(restarts):
        w = (np.zeros(F, np.float32) if r == 0 and init is None
             else (init.astype(np.float32) if r == 0
                   else rng.normal(0, 1, F).astype(np.float32)))
        cur = ntop1(w, Xt, sT, tT)
        for i in range(iters):
            sigma = 0.6 * (1.0 - i / iters) + 0.05
            cand = w + rng.normal(0, sigma, F).astype(np.float32)
            a = ntop1(cand, Xt, sT, tT)
            if a > cur:
                w, cur = cand, a
        v = ntop1(w, Xv, sV, tV)
        print(f"    {tag}  restart {r}  train {100 * cur:5.2f}%  "
              f"val {100 * v:5.2f}%", flush=True)
        winners.append((v, cur, w))
    v, trn, w = max(winners, key=lambda t: t[0])
    return ntop1(w, Xh, sH, tH), w / sd, trn, v


def fit(train, val, held, cols, l2_grid, tag):
    """L2 chosen on `val`; `held` is scored once, with the winning L2.

    Held-out is not used for model selection anywhere, so the number reported
    for it is not optimistic in the way an early-stopped curve would be.
    """
    Xt, st, tt, mu, sd = pack(train, cols)
    Xv, sv, tv, _, _ = pack(val, cols, mu, sd)
    Xh, sh, th, _, _ = pack(held, cols, mu, sd)
    best = (-1.0, None, None)
    for l2 in l2_grid:
        w = torch.zeros(len(cols), requires_grad=True)
        opt = torch.optim.LBFGS([w], max_iter=400, history_size=20,
                                line_search_fn="strong_wolfe",
                                tolerance_grad=1e-9)

        def closure():
            opt.zero_grad()
            loss = (listwise_loss(Xt @ w, st, tt, len(train))
                    + l2 * (w * w).sum())
            loss.backward()
            return loss

        opt.step(closure)
        a = acc_of(w.detach(), Xv, sv, tv, len(val))
        print(f"    {tag}  l2 {l2:<8g} val top-1 {100 * a:5.2f}%", flush=True)
        if a > best[0]:
            best = (a, w.detach().clone(), l2)
    _, w, l2 = best
    return (acc_of(w, Xh, sh, th, len(held)), w.numpy() / sd, l2,
            (Xh, sh, th))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default=None)
    ap.add_argument("--l2", type=str, default="0,1e-3,1e-2,0.1",
                    help="ridge grid for the likelihood fit")
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--restarts", type=int, default=4)
    args = ap.parse_args()

    if args.src:
        paths = sorted(Path().glob(args.src)) or [Path(args.src)]
    else:
        shards = sorted((PROBES / "trade_shards").glob("*.jsonl.gz"))
        paths = shards if shards else [PROBES / "trade_harvest.jsonl"]
    paths = [p for p in paths if p.exists()]
    states = load(paths)

    # The 70/30 held-out split is Candidate D's, unchanged, so every number
    # here sits on the same 1,327 states as rank_anchor.py. The validation
    # slice used for L2 is carved out of the TRAIN side only.
    seeds = sorted({s[0] for s in states})
    random.Random(20250811).shuffle(seeds)
    tr_seeds = seeds[: int(0.7 * len(seeds))]
    va = set(tr_seeds[int(0.8 * len(tr_seeds)):])
    tr = set(tr_seeds) - va
    train = [s for s in states if s[0] in tr]
    val = [s for s in states if s[0] in va]
    held = [s for s in states if s[0] not in set(tr_seeds)]
    mean_c = sum(len(s[1]) for s in states) / len(states)
    print(f"{len(paths)} file(s)  {len(states)} proposals  {len(seeds)} seeds  "
          f"mean {mean_c:.1f} candidates -> random {100 / mean_c:.2f}%")
    print(f"split by game seed: {len(train)} train / {len(val)} val / "
          f"{len(held)} held-out\n")

    base_cols = list(range(len(BASE)))
    all_cols = list(range(len(BASE) + len(EXTRA)))
    grid = [float(x) for x in args.l2.split(",")]

    shipped = top1_mask(SHIPPED, held, base_cols)
    ks = sum(shipped)
    p, lo, hi = wilson(ks, len(held))
    print(f"  shipped TRADE_W          {ks}/{len(held)}  {100 * p:5.2f}%  "
          f"[{100 * lo:.2f}, {100 * hi:.2f}]")

    hits = {}

    def report(label, w, cols_):
        m = top1_mask(w, held, cols_)
        k = sum(m)
        hits[label.split()[0]] = k
        pp, llo, hhi = wilson(k, len(held))
        print(f"  {label:<26}{k}/{len(held)}  {100 * pp:5.2f}%  "
              f"[{100 * llo:.2f}, {100 * hhi:.2f}]")
        return m

    print("\n  L. likelihood fit, same six features (surrogate check)")
    _, wL, l2L, _ = fit(train, val, held, base_cols, grid, "L")
    maskL = report(f"L likelihood (l2 {l2L:g})", wL, base_cols)

    print("\n  A. direct top-1 refit, same six features")
    _, wA, trA, vA = search(train, val, held, base_cols, args.iters,
                            args.restarts, "A", init=SHIPPED)
    maskA = report("A refit (6 features)", wA, base_cols)

    print("\n  B. direct top-1 refit, D7.3's extended set")
    initB = np.concatenate([SHIPPED, np.zeros(len(EXTRA), np.float32)])
    _, wB, trB, vB = search(train, val, held, all_cols, args.iters,
                            args.restarts, "B", init=initB)
    maskB = report("B refit (16 features)", wB, all_cols)

    def mcnemar(x, y, name):
        b = sum(1 for i, j in zip(x, y) if i and not j)
        c = sum(1 for i, j in zip(x, y) if j and not i)
        z = (c - b) / math.sqrt(b + c) if b + c else 0.0
        pv = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        print(f"  {name:<34} {c - b:+4d} net   z {z:+.2f}   p {pv:.4f}   "
              f"{'SIGNIFICANT' if pv < 0.05 else 'ns'}")

    print("\npaired, same held-out states:")
    mcnemar(shipped, maskL, "likelihood fit vs shipped")
    mcnemar(shipped, maskA, "A vs shipped (refit alone)")
    mcnemar(maskA, maskB, "B vs A (the new features)")
    mcnemar(shipped, maskB, "B vs shipped (total)")

    names = list(BASE) + list(EXTRA)
    print("\nB's weights, raw units (shipped value in brackets):")
    for n, v in sorted(zip(names, wB), key=lambda t: -abs(t[1])):
        s = f"  [{TRADE_W[n]:+.3f}]" if n in TRADE_W else ""
        print(f"  {n:<22}{v:+9.4f}{s}")

    OUT.write_text(json.dumps({
        "features": names,
        "weights_B": {n: float(v) for n, v in zip(names, wB)},
        "weights_A": {n: float(v) for n, v in zip(BASE, wA)},
        "weights_L": {n: float(v) for n, v in zip(BASE, wL)},
        "held_top1": {"shipped": ks / len(held),
                      "L": hits["L"] / len(held),
                      "A": hits["A"] / len(held),
                      "B": hits["B"] / len(held)},
        "search": {"A_train": trA, "A_val": vA, "B_train": trB,
                   "B_val": vB},
        "n_held": len(held), "n_val": len(val),
        "n_train": len(train),
        "corpus_files": len(paths), "proposals": len(states),
    }, indent=2))
    print(f"\nwrote {OUT}")
    print("Top-1 is diagnostic. A win rate decides whether any of this ships.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
