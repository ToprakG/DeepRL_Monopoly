"""Live scoreboard: plus vs Slayer / Underdog / Inncenta. Refresh every 5s."""

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
REFRESH_S = 5.0
_LOG_KEEP = ("h2h progress", "lineup=", "oracle WR=", "  vs ", "started pid", "Error", "Traceback", "TypeError", "ModuleNotFound")


def _alive() -> bool:
    if not PID.exists():
        return False
    try:
        os.kill(int(PID.read_text().strip()), 0)
        return True
    except (OSError, ValueError):
        return False


def _log_tail() -> list[str]:
    if not LOG.exists():
        return []
    kept = []
    for ln in LOG.read_text().splitlines():
        text = ln.strip()
        if not text or text.startswith("{") or text.startswith("}") or text.startswith('"'):
            continue
        if any(text.startswith(prefix) or prefix in text for prefix in _LOG_KEEP):
            kept.append(text[:110])
    return kept[-6:]


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
    tail = _log_tail()
    if tail:
        lines += ["", "log"]
        lines.extend(f"  {ln}" for ln in tail)
    return "\n".join(lines)


def main() -> int:
    idle = 0
    while True:
        print("\033[2J\033[H" + _render(), flush=True)
        done = not _alive()
        n = len(list(CKPT.glob("game_*.json")))
        if done:
            idle += 1
            if n >= GAMES:
                return 0
            if idle >= 12:
                return 1
        else:
            idle = 0
        time.sleep(REFRESH_S)


if __name__ == "__main__":
    raise SystemExit(main())
