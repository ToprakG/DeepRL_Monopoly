"""Live scoreboard: plus vs Underdog / Slayer / Alinebidal."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "artifacts_scratch" / "plus_first_race.log"
CKPT = ROOT / "artifacts_scratch" / "plus_first_race_ckpt"
PID = ROOT / "artifacts_scratch" / "plus_first_race.pid"
ORDER = ("oracle-plus-v1", "slayer-v1", "underdog-v1", "inncenta-heuristic")
GAMES = 24


def _alive() -> bool:
    if not PID.exists():
        return False
    try:
        os.kill(int(PID.read_text().strip()), 0)
        return True
    except (OSError, ValueError):
        return False


def _render() -> str:
    wins: Counter[str] = Counter()
    timeouts = 0
    games = sorted(CKPT.glob("game_*.json"))
    lineup: list[str] = []
    for path in games:
        rec = json.loads(path.read_text())
        result = rec["result"]
        if result.get("timed_out"):
            timeouts += 1
        policy = result.get("winner_policy")
        if policy:
            wins[policy] += 1
        if not lineup:
            lineup = list(result.get("policies") or [])
    n = len(games)
    ours = wins.get("oracle-plus-v1", 0)
    wr = (ours / n) if n else 0.0
    lines = [
        time.strftime("%H:%M:%S"),
        f"job {'RUNNING' if _alive() else 'STOPPED'}   plus vs Slayer/Underdog/Inncenta   {n}/{GAMES}",
        f"our WR {wr:.1%}  ({ours}/{n})   timeouts={timeouts}",
        "",
    ]
    names = lineup or list(ORDER)
    for extra in wins:
        if extra not in names:
            names.append(extra)
    for name in names:
        w = wins[name]
        pct = f"{100.0 * w / n:5.1f}%" if n else "  n/a"
        mark = " <-- us" if name == "oracle-plus-v1" else ""
        lines.append(f"  {name:<22} {w:2d}  {pct}  {'#' * w}{mark}")
    if LOG.exists():
        tail = [ln for ln in LOG.read_text().splitlines() if ln.strip()][-8:]
        lines += ["", "log"]
        lines.extend(f"  {ln[:110]}" for ln in tail)
    return "\n".join(lines)


def main() -> int:
    while True:
        print("\033[2J\033[H" + _render(), flush=True)
        time.sleep(0.25)


if __name__ == "__main__":
    raise SystemExit(main())
