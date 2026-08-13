"""Gate: closed-form oracle-plus-v1 vs the 0b trio. Pass = our WR >= 30%."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
LOG_DIR = ROOT / "artifacts_scratch"
LINEUP = "oracle-plus-v1,alinebidal-final,slayer-v1,inncenta-heuristic"
GAMES = 48
SEED = 9_400_000
PASS_WR = 0.30


def main() -> int:
    argv = [
        str(PYTHON),
        "-u",
        "-m",
        "oracle.eval_h2h",
        "--games",
        str(GAMES),
        "--seed",
        str(SEED),
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
        "--lineup",
        LINEUP,
        "--leaf",
        "asu",
        "--no-one-ply",
        "--denial",
        "--completing-trade",
        "--auction",
        "--inncenta-trade",
        "--checkpoint-dir",
        str(LOG_DIR / "plus_gate_0b_ckpt"),
        "--output",
        str(LOG_DIR / "plus_gate_0b.json"),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    print(" ".join(argv), flush=True)
    code = int(subprocess.run(argv, cwd=ROOT, env=env, check=False).returncode)
    out = LOG_DIR / "plus_gate_0b.json"
    if not out.is_file():
        print(f"plus gate failed code={code} no output", flush=True)
        return 1
    import json

    payload = json.loads(out.read_text())
    ours = (payload.get("win_rates") or {}).get("oracle-plus-v1") or {}
    wr = float(ours.get("win_rate") or 0.0)
    print(
        f"plus gate ours={wr:.3f} ({ours.get('wins')}/{ours.get('games')}) "
        f"pass30={wr >= PASS_WR}",
        flush=True,
    )
    return 0 if wr >= PASS_WR else 2


if __name__ == "__main__":
    raise SystemExit(main())
