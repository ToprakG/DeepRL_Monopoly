"""Live scoreboard for the plus vs 0b-trio gate."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "artifacts_scratch" / "plus_gate_0b.log"
CKPT = ROOT / "artifacts_scratch" / "plus_gate_0b_ckpt"
PID = ROOT / "artifacts_scratch" / "plus_gate_0b.pid"
ORDER = ("oracle-plus-v1", "alinebidal-final", "slayer-v1", "inncenta-heuristic")
PASS_WR = 0.30
GAMES = 48


def _alive() -> bool:
    if not PID.exists():
        return False
    try:
        os.kill(int(PID.read_text().strip()), 0)
        return True
    except (OSError, ValueError):
        return False


def _tally() -> tuple[int, int, Counter, list[str]]:
    wins: Counter[str] = Counter()
    timeouts = 0
    lineup: list[str] = []
    for path in sorted(CKPT.glob("game_*.json")):
        rec = json.loads(path.read_text())
        result = rec["result"]
        if result.get("timed_out"):
            timeouts += 1
        policy = result.get("winner_policy")
        if policy:
            wins[policy] += 1
        if not lineup:
            lineup = list(result.get("policies") or [])
    return len(list(CKPT.glob("game_*.json"))), timeouts, wins, lineup


def _render() -> str:
    n, timeouts, wins, lineup = _tally()
    ours = wins.get("oracle-plus-v1", 0)
    wr = (ours / n) if n else 0.0
    lines = [
        time.strftime("%H:%M:%S"),
        f"job {'RUNNING' if _alive() else 'STOPPED'}   plus vs 0b trio   {n}/{GAMES}",
        f"our WR {wr:.1%}  ({ours}/{n})   pass30={'YES' if n and wr >= PASS_WR else 'no'}   timeouts={timeouts}",
        "",
    ]
    names = lineup or [name for name in ORDER]
    for extra in wins:
        if extra not in names:
            names.append(extra)
    for name in names:
        w = wins[name]
        pct = f"{100.0 * w / n:5.1f}%" if n else "  n/a"
        marker = " <-- us" if name == "oracle-plus-v1" else ""
        lines.append(f"  {name:<22} {w:2d}  {pct}  {'#' * w}{marker}")
    if LOG.exists():
        tail = [ln for ln in LOG.read_text().splitlines() if ln.strip()][-8:]
        lines.append("")
        lines.append("log")
        lines.extend(f"  {ln[:110]}" for ln in tail)
    return "\n".join(lines)


def main() -> int:
    while True:
        print("\033[2J\033[H" + _render(), flush=True)
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
