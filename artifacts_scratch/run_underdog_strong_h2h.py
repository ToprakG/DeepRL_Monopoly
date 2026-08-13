"""oracle-fast-v1 vs UNDERDOG + other strong heuristics. 72 games, 3 fields."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
LOG_DIR = ROOT / "artifacts_scratch"

LINEUPS = [
    "oracle-fast-v1,underdog-v1,slayer-v1,alinebidal-final",
    "oracle-fast-v1,underdog-v1,inncenta-heuristic,slayer-v1",
    "oracle-fast-v1,underdog-v1,alinebidal-final,inncenta-heuristic",
]


def main() -> int:
    argv = [
        str(PYTHON),
        "-u",
        "-m",
        "oracle.eval_h2h",
        "--games",
        "24",
        "--seed",
        "9200000",
        "--sims",
        "32",
        "--horizon",
        "16",
        "--rollouts",
        "1",
        "--workers",
        "6",
        "--deadline-s",
        "0",
        "--early-stop-lead",
        "0",
        "--game-timeout-s",
        "0",
        "--resume",
        "--pretty",
        "--checkpoint-dir",
        str(LOG_DIR / "h2h_underdog_strong_ckpt"),
        "--output",
        str(LOG_DIR / "h2h_underdog_strong.json"),
    ]
    for lineup in LINEUPS:
        argv.extend(["--lineup", lineup])
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    print(" ".join(argv), flush=True)
    return int(subprocess.run(argv, cwd=ROOT, env=env, check=False).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
