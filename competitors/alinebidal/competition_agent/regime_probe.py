"""Two questions the regime-switched scorer depends on, answered before it is built.

Q1 — is there an observable proxy for "how late is this game"?
    The slice table that motivates the switch is indexed by *remaining* steps,
    which no policy can see. This measures how well each observable board
    quantity tracks it, so the switch point can be derived rather than picked.

Q2 — does the network's late-game skill survive once the game is still live?
    D6.2 reports AUC 0.766 in the last 50 steps. By then opponents are
    frequently bankrupt already, and predicting a decided game is worth
    nothing to a policy: there is no longer an action that changes it. So the
    same slices are recomputed restricted to states where **all four seats are
    still solvent**. If the late-slice AUC collapses there, the network is
    reading bankruptcies off the board rather than valuing a live position,
    and switching to it late buys nothing.

Everything is decoded from the stored 300-dim observation, whose layout is
documented in `monopoly_game_engine/state.py`. No re-collection needed.
"""

from __future__ import annotations

import argparse
import glob
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

from competition_agent.train_value import ValueHead  # noqa: E402

OUTDIR = Path(__file__).resolve().parent / "probes" / "outcomes"

# --- observation offsets, derived from build_state_vector -------------------
N_PROP = 28
PROP0 = 16                      # 4 players x 4 features
OWNER = lambda i: PROP0 + 8 * i          # noqa: E731  5 dims one-hot
MORT = lambda i: PROP0 + 8 * i + 5       # noqa: E731
MONO = lambda i: PROP0 + 8 * i + 6       # noqa: E731
IMPR = lambda i: PROP0 + 8 * i + 7       # noqa: E731
HOUSES_AVAIL, HOTELS_AVAIL = 256, 257
BANKRUPT0 = 258                 # 4 dims, agent-relative order
ROUND_FRAC = 278


def proxies(obs: np.ndarray) -> dict:
    """Board quantities a policy can read at decision time."""
    mono = obs[:, [MONO(i) for i in range(N_PROP)]].sum(1)
    impr = obs[:, [IMPR(i) for i in range(N_PROP)]].sum(1) * 5.0
    mort = obs[:, [MORT(i) for i in range(N_PROP)]].sum(1)
    owned = sum(obs[:, OWNER(i):OWNER(i) + 5].sum(1) for i in range(N_PROP))
    cash = obs[:, [1, 5, 9, 13]]
    return {
        "round_frac": obs[:, ROUND_FRAC],
        "houses_on_board": impr,
        "houses_taken": 1.0 - obs[:, HOUSES_AVAIL],
        "hotels_taken": 1.0 - obs[:, HOTELS_AVAIL],
        "monopoly_deeds": mono,
        "deeds_owned": owned,
        "mortgaged": mort,
        "n_bankrupt": obs[:, BANKRUPT0:BANKRUPT0 + 4].sum(1),
        "cash_spread": cash.max(1) - cash.min(1),
        "cash_total": cash.sum(1),
    }


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    d = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return float((ra * rb).sum() / d) if d else 0.0


def auc(pred, label):
    pos, neg = label.sum(), len(label) - label.sum()
    if not pos or not neg:
        return float("nan")
    order = np.argsort(pred)
    ranks = np.empty(len(pred), np.float64)
    ranks[order] = np.arange(1, len(pred) + 1)
    return float((ranks[label == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(
        Path(__file__).resolve().parent / "value_head_discounted.pt"))
    ap.add_argument("--games", type=int, default=500)
    args = ap.parse_args()

    files = sorted(glob.glob(str(OUTDIR / "g*.npz")))[:args.games]
    if not files:
        print("no outcome corpus found")
        return 1

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = ValueHead(ck.get("hidden", 256), ck.get("dropout", 0.2))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    torch.set_num_threads(1)

    # train_value.py's split, reproduced exactly. Without it every AUC below
    # is measured on games the network was fitted to, and D6.2 recorded that
    # this network overfits hard (best log-loss at epoch 1, degrading after) —
    # so in-sample numbers run ~0.15 AUC high and mean nothing.
    ids = sorted(int(Path(f).stem[1:]) for f in files)
    gl = list(np.unique(ids))
    random.Random(20250811).shuffle(gl)
    train_ids = set(gl[: int(0.7 * len(gl))])
    print(f"split by game: {len(train_ids)} train / "
          f"{len(gl) - len(train_ids)} held-out")

    keys = None
    P, REM, WIN, PRED, ALIVE4, HELD = [], [], [], [], [], []
    with torch.no_grad():
        for f in files:
            d = np.load(f)
            obs = d["obs"].astype(np.float32)
            if not len(obs):
                continue
            pr = proxies(obs)
            keys = list(pr)
            P.append(np.stack([pr[k] for k in keys], 1))
            REM.append(d["step"].max() - d["step"].astype(np.float32))
            WIN.append(d["win"].astype(np.int8))
            ALIVE4.append(pr["n_bankrupt"] == 0)
            HELD.append(np.full(len(obs),
                                int(Path(f).stem[1:]) not in train_ids, bool))
            ps = []
            for i in range(0, len(obs), 8192):
                ps.append(torch.sigmoid(model(torch.from_numpy(obs[i:i + 8192]))))
            PRED.append(torch.cat(ps).numpy())

    P = np.concatenate(P)
    rem = np.concatenate(REM)
    win = np.concatenate(WIN)
    pred = np.concatenate(PRED)
    alive4 = np.concatenate(ALIVE4)
    held = np.concatenate(HELD)
    n = len(rem)
    print(f"{len(files)} games, {n} states, "
          f"{100 * alive4.mean():.1f}% with all four seats solvent\n")

    print("=== Q1: which observable proxy tracks remaining steps? ===")
    print(f"  {'proxy':<20}{'Spearman vs remaining':>24}")
    rows = [(abs(spearman(P[:, i], rem)), keys[i], spearman(P[:, i], rem))
            for i in range(P.shape[1])]
    for _, k, r in sorted(rows, reverse=True):
        print(f"  {k:<20}{r:>+24.4f}")

    # Everything below is held-out only.
    rem, win, pred, alive4 = rem[held], win[held], pred[held], alive4[held]
    print(f"\nheld-out states: {len(rem)}")
    print("\n=== Q2: does the late-game AUC survive a live board? ===")
    print(f"  {'slice':<12}{'n':>9}{'AUC all':>10}{'n solvent':>11}"
          f"{'AUC solvent':>13}{'win rate':>10}")
    bounds = [(0, 50), (50, 150), (150, 300), (300, 10 ** 9)]
    for lo, hi in bounds:
        m = (rem >= lo) & (rem < hi)
        ms = m & alive4
        lab = f"{lo}-{hi if hi < 10**9 else '+'}"
        a_all = auc(pred[m], win[m]) if m.sum() else float("nan")
        a_sol = auc(pred[ms], win[ms]) if ms.sum() > 50 else float("nan")
        print(f"  {lab:<12}{m.sum():>9}{a_all:>10.3f}{ms.sum():>11}"
              f"{a_sol:>13.3f}{win[m].mean():>10.3f}")

    print("\n=== Q2b: is the network saturated late? ===")
    print("  (a scorer pinned at 0 or 1 cannot rank two candidate actions)")
    print(f"  {'slice':<12}{'mean p':>9}{'sd p':>9}"
          f"{'|p-0.5|>0.4':>13}{'sd p (solvent)':>16}")
    for lo, hi in bounds:
        m = (rem >= lo) & (rem < hi)
        ms = m & alive4
        p = pred[m]
        lab = f"{lo}-{hi if hi < 10**9 else '+'}"
        print(f"  {lab:<12}{p.mean():>9.3f}{p.std():>9.3f}"
              f"{100 * (np.abs(p - 0.5) > 0.4).mean():>12.1f}%"
              f"{pred[ms].std() if ms.sum() else float('nan'):>16.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
