"""Live view of the field runs: games played, wins, and the paired delta.

The runs stream every finished game to a `.partial.jsonl` sidecar (D0.8), so
progress is readable without touching the running process. This tails those
files and redraws.

    python3 competition_agent/watch.py              # live, refreshes
    python3 competition_agent/watch.py --once       # one snapshot
    python3 competition_agent/watch.py --every 10   # slower refresh

Columns
-------
    games      finished so far / target, where the target is read from the
               run's .log ("N to play, M reused")
    win        wins with a 95% Wilson interval, against the field's parity
    bank       games where every seat we hold went bankrupt
    prop/acc   trade proposals made, and the share accepted — the quantity
               that decides whether a ranking change can matter at all
    rate       games per minute, measured between refreshes

When both arms of a field are present the paired delta is shown underneath,
computed only on seeds both arms have finished, which is the comparison that
counts. Board luck cancels in the pairing; the harness's own noise floor is
0 flipped games in 2,000.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

PROBES = Path(__file__).resolve().parent / "probes"
PARITY = {"asu": 50.0, "strong": 25.0, "weak": 25.0}
CLEAR = "\033[H\033[J"
BOLD, DIM, GRN, RED, YEL, OFF = (
    "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[0m")


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - m), min(1.0, c + m)


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def read_partial(path: Path):
    """Tolerates a torn final line: the writer may be mid-append."""
    rows = []
    try:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        pass
    return rows


def target_of(field: str, tag: str):
    log = PROBES / f"field_{field}_{tag}.log"
    try:
        m = re.search(r"(\d+) to play, (\d+) reused", log.read_text())
        if m:
            return int(m.group(1)) + int(m.group(2))
    except OSError:
        pass
    return None


def discover():
    out = []
    for p in sorted(PROBES.glob("field_*.partial.jsonl")):
        m = re.match(r"field_([a-z]+)_(.+)\.partial\.jsonl", p.name)
        if m:
            out.append((m.group(1), m.group(2), p))
    return out


def render(state, once):
    runs = discover()
    if not runs:
        return "no field runs found in probes/\n"

    now = time.time()
    lines = [f"{BOLD}field runs{OFF}   {time.strftime('%H:%M:%S')}",
             f"{DIM}{'field/arm':<16}{'games':>13}{'win rate':>24}"
             f"{'bank':>7}{'prop/acc':>16}{'rate':>10}{OFF}",
             "-" * 86]
    by_field = {}
    for field, tag, path in runs:
        rows = read_partial(path)
        n = len(rows)
        by_field.setdefault(field, {})[tag] = rows
        k = sum(r.get("leader_win", False) for r in rows)
        p, lo, hi = wilson(k, n)
        bank = sum(r.get("bankrupt", False) for r in rows)
        prop = sum(r.get("proposed", 0) for r in rows)
        acc = sum(r.get("accepted", 0) for r in rows)
        tgt = target_of(field, tag)

        prev = state.get((field, tag))
        rate = ""
        if prev and now > prev[1] and n > prev[0]:
            rate = f"{(n - prev[0]) / (now - prev[1]) * 60:.0f}/min"
        if not prev or n != prev[0]:
            state[(field, tag)] = (n, now)

        par = PARITY.get(field, 25.0)
        col = GRN if 100 * p > par else RED
        live = f"{YEL}*{OFF}" if rate else " "
        lines.append(
            f"{live}{field + '/' + tag:<15}"
            f"{f'{n}' + (f'/{tgt}' if tgt else ''):>13}"
            f"{col}{f'{k}  {100*p:5.2f}% [{100*lo:.1f},{100*hi:.1f}]':>24}{OFF}"
            f"{f'{100*bank/max(n,1):.0f}%':>7}"
            f"{f'{prop} / {100*acc/max(prop,1):.1f}%':>16}"
            f"{rate:>10}")

    for field, arms in by_field.items():
        if len(arms) < 2:
            continue
        (ta, ra), (tb, rb) = sorted(arms.items())[:2]
        A = {r["seed"]: r for r in ra}
        B = {r["seed"]: r for r in rb}
        shared = sorted(set(A) & set(B))
        if not shared:
            continue
        ka = sum(A[s]["leader_win"] for s in shared)
        kb = sum(B[s]["leader_win"] for s in shared)
        b = sum(1 for s in shared if A[s]["leader_win"] and not B[s]["leader_win"])
        c = sum(1 for s in shared if B[s]["leader_win"] and not A[s]["leader_win"])
        z = (c - b) / math.sqrt(b + c) if b + c else 0.0
        pv = 2 * (1 - phi(abs(z)))
        d = 100 * (kb - ka) / len(shared)
        col = GRN if d > 0 else (RED if d < 0 else "")
        lines.append(
            f"{DIM}  paired {field}: {tb} - {ta} = {OFF}{col}{d:+.2f}pp{OFF}"
            f"{DIM} on {len(shared)} shared seeds, "
            f"discordant {b + c}, p {pv:.4f}{OFF}")

    lines.append("")
    if not once:
        lines.append(f"{DIM}ctrl-c to stop{OFF}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=float, default=3.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    state = {}
    if args.once:
        print(render(state, True), end="")
        return 0
    try:
        while True:
            out = render(state, False)
            print(CLEAR + out, end="", flush=True)
            time.sleep(args.every)
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
