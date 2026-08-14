"""Phase 6 Part B — train P(win | state).

Objective
---------
Binary cross-entropy on "did this seat go on to win", over the 300-dim
observation. This is the first model in the project trained against winning
rather than against the teacher's action choices.

Label variants (two, as specified — not three)
----------------------------------------------
1. **plain**    y = 1 if the seat won.

2. **discounted** y = 0.25 + (win - 0.25) * gamma**(remaining_steps / scale)

   Chosen over the rank-blend alternative. The reasoning: a binary end-of-game
   outcome assigns identical credit to the opening position and to the move
   that decided the game, and the opening position of a seat that later won
   is barely distinguishable from one that later lost — it is mostly noise.
   Discounting by distance-to-end shrinks early states toward the 0.25 base
   rate, which is exactly what they deserve when four seats are still level,
   while leaving late states near the true outcome. A rank blend would instead
   change *what* is predicted (finishing position rather than winning), and
   winning is the objective; second place is worth no more than fourth here.

Split is by **game id**, never by decision: states inside one game share a
board and would leak, as established in Phase 3.

Calibration is reported alongside log-loss and AUC and matters more than
either, because this model is used to *compare* candidate actions. A scorer
that ranks well but is mis-calibrated still picks the right action; one that
is well calibrated but flat in the region where candidates differ does not.
"""

from __future__ import annotations

import argparse
import glob
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
import torch.nn as nn  # noqa: E402

OUTDIR = Path(__file__).resolve().parent / "probes" / "outcomes"
CKPT = Path(__file__).resolve().parent / "value_head.pt"
OBS_DIM = 300
BASE_RATE = 0.25


class ValueHead(nn.Module):
    def __init__(self, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)          # logits


def load(limit=None, gamma=0.98, scale=50.0, variant="plain"):
    files = sorted(glob.glob(str(OUTDIR / "g*.npz")))
    if limit:
        files = files[:limit]
    X, Y, G = [], [], []
    for f in files:
        d = np.load(f)
        obs = d["obs"].astype(np.float32)
        if not len(obs):
            continue
        win = d["win"].astype(np.float32)
        step = d["step"].astype(np.float32)
        if variant == "discounted":
            remaining = step.max() - step
            w = gamma ** (remaining / scale)
            y = BASE_RATE + (win - BASE_RATE) * w
        else:
            y = win
        X.append(obs)
        Y.append(y)
        G.append(np.full(len(obs), int(Path(f).stem[1:]), dtype=np.int64))
    return (np.concatenate(X), np.concatenate(Y), np.concatenate(G))


def metrics(model, X, Y, bs=4096):
    model.eval()
    ps = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            ps.append(torch.sigmoid(model(torch.from_numpy(X[i:i + bs]))))
    p = torch.cat(ps).numpy()
    eps = 1e-7
    ll = -np.mean(Y * np.log(p + eps) + (1 - Y) * np.log(1 - p + eps))
    # AUC against the hard win label
    hard = (Y > 0.5).astype(np.int8) if Y.max() <= 1.0 else Y
    order = np.argsort(p)
    ranks = np.empty(len(p), float)
    ranks[order] = np.arange(1, len(p) + 1)
    npos, nneg = hard.sum(), len(hard) - hard.sum()
    auc = ((ranks[hard == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
           if npos and nneg else float("nan"))
    return ll, auc, p, hard


def calibration(p, hard, bins=10):
    out = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        m = (p >= lo) & (p < hi)
        if m.sum() >= 50:
            out.append((lo, hi, float(p[m].mean()), float(hard[m].mean()),
                        int(m.sum())))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["plain", "discounted", "both"],
                    default="both")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    torch.manual_seed(20250811)
    variants = (["plain", "discounted"] if args.variant == "both"
                else [args.variant])
    best_overall = None

    for variant in variants:
        X, Y, G = load(limit=args.limit, variant=variant)
        games = np.unique(G)
        rng = random.Random(20250811)
        gl = list(games)
        rng.shuffle(gl)
        tr = set(gl[: int(0.7 * len(gl))])
        m = np.array([g in tr for g in G])
        Xtr, Ytr, Xh, Yh = X[m], Y[m], X[~m], Y[~m]

        print(f"\n=== variant: {variant} ===")
        print(f"rows {len(X)}  games {len(games)}  "
              f"train {len(Xtr)}  held-out {len(Xh)}  (split by game)")

        model = ValueHead(args.hidden, args.dropout)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        lossf = nn.BCEWithLogitsLoss()
        idx = np.arange(len(Xtr))
        best = 1e9

        for ep in range(1, args.epochs + 1):
            model.train()
            np.random.shuffle(idx)
            for i in range(0, len(idx), args.bs):
                b = idx[i:i + args.bs]
                opt.zero_grad()
                loss = lossf(model(torch.from_numpy(Xtr[b])),
                             torch.from_numpy(Ytr[b]))
                loss.backward()
                opt.step()
            ll, auc, p, hard = metrics(model, Xh, Yh)
            if ll < best:
                best = ll
                torch.save({"state_dict": model.state_dict(),
                            "variant": variant, "hidden": args.hidden,
                            "dropout": args.dropout, "held_logloss": ll,
                            "held_auc": auc},
                           str(CKPT).replace(".pt", f"_{variant}.pt"))
            if ep % 3 == 0 or ep == 1:
                print(f"  ep {ep:>2}  held-out logloss {ll:.4f}  AUC {auc:.4f}")

        ll, auc, p, hard = metrics(model, Xh, Yh)
        print(f"  best held-out logloss {best:.4f}   AUC {auc:.4f}")
        print(f"  {'bin':<12}{'pred':>8}{'actual':>9}{'n':>8}")
        for lo, hi, pm, am, n in calibration(p, hard):
            print(f"  {f'[{lo:.1f},{hi:.1f})':<12}{pm:>8.3f}{am:>9.3f}{n:>8}")
        if best_overall is None or best < best_overall[0]:
            best_overall = (best, variant)

    print(f"\nbest variant by held-out log-loss: {best_overall[1]} "
          f"({best_overall[0]:.4f})")
    print("NOTE: log-loss and AUC do NOT project to win rate. "
          "Part C measures win rate directly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
