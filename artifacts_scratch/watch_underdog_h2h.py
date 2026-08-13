"""Live H2H scoreboard for the Underdog strong-field run."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "artifacts_scratch" / "h2h_underdog_strong.log"
CKPT = ROOT / "artifacts_scratch" / "h2h_underdog_strong_ckpt"
PID = ROOT / "artifacts_scratch" / "h2h_underdog_strong.pid"
ORDER = (
    "oracle-fast-v1",
    "underdog-v1",
    "slayer-v1",
    "alinebidal-final",
    "inncenta-heuristic",
)


def _alive() -> bool:
    if not PID.exists():
        return False
    try:
        os.kill(int(PID.read_text().strip()), 0)
        return True
    except (OSError, ValueError):
        return False


def _tally(ckpt: Path) -> tuple[int, int, int, Counter, list[str]]:
    games = sorted(ckpt.glob("game_*.json"))
    wins: Counter[str] = Counter()
    timeouts = trunc = 0
    lineup: list[str] = []
    for path in games:
        rec = json.loads(path.read_text())
        result = rec["result"]
        if result.get("timed_out"):
            timeouts += 1
        if result.get("truncated"):
            trunc += 1
        policy = result.get("winner_policy")
        if policy:
            wins[policy] += 1
        if not lineup:
            lineup = list(result.get("policies") or [])
    return len(games), timeouts, trunc, wins, lineup


def _render() -> str:
    lines = [
        time.strftime("%H:%M:%S"),
        f"job {'RUNNING' if _alive() else 'STOPPED'}",
        "",
    ]
    fields = sorted(p for p in CKPT.glob("field_*") if p.is_dir())
    if not fields and CKPT.is_dir():
        fields = [CKPT]
    if not fields:
        lines.append("no checkpoints yet")
    for ckpt in fields:
        n, timeouts, trunc, wins, lineup = _tally(ckpt)
        title = ckpt.name.replace("field_00_", "field 1  ").replace(
            "field_01_", "field 2  "
        ).replace("field_02_", "field 3  ")
        lines.append(f"{title}   {n}/24   timeouts={timeouts} trunc={trunc}")
        names = lineup or [name for name in ORDER if name in wins]
        for extra in wins:
            if extra not in names:
                names.append(extra)
        for name in names:
            w = wins[name]
            wr = f"{100.0 * w / n:5.1f}%" if n else "  n/a"
            bar = "#" * w
            lines.append(f"  {name:<22} {w:2d}  {wr}  {bar}")
        lines.append("")
    if LOG.exists():
        tail = [ln for ln in LOG.read_text().splitlines() if ln.strip()][-8:]
        lines.append("log")
        lines.extend(f"  {ln[:100]}" for ln in tail)
    return "\n".join(lines)


def main() -> int:
    while True:
        text = _render()
        print("\033[2J\033[H" + text, flush=True)
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
