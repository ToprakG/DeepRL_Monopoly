"""Colab/local chain: probe pool → 32-sim field gap → best-response train."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from oracle.field_br import main as field_br_main

CLONE = Path(os.environ.get("FIELD_BR_CLONE", "/content/hybrid_clone_0000.pt"))
RUN_DIR = Path(os.environ.get("FIELD_BR_RUN", "/content/oracle-field-br-run"))
WORKERS = os.environ.get("FIELD_BR_WORKERS", "8")
GAMES = os.environ.get("FIELD_BR_GAP_GAMES", "40")
EXPO = os.environ.get("FIELD_BR_EXPO_GAMES", "8")
SIMS = os.environ.get("FIELD_BR_SIMS", "32")
GENERATIONS = os.environ.get("FIELD_BR_GENERATIONS", "3")


def _run(argv: list[str]) -> int:
    print(" ".join(["field_br"] + argv), flush=True)
    return int(field_br_main(argv))


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    gap_dir = RUN_DIR / "gap"
    gap_json = gap_dir / "gap.json"
    probe = RUN_DIR / "reports" / "probe.json"
    probe.parent.mkdir(parents=True, exist_ok=True)

    code = _run(["probe", "--output", str(probe), "--max-rounds", "8"])
    if code != 0:
        return code

    code = _run(
        [
            "gap",
            "--clone",
            str(CLONE),
            "--games",
            GAMES,
            "--expo-games",
            EXPO,
            "--sims",
            SIMS,
            "--workers",
            WORKERS,
            "--checkpoint-dir",
            str(gap_dir),
            "--resume",
            "--output",
            str(gap_json),
        ]
    )
    if code != 0:
        return code

    train = [
        "train",
        "--clone",
        str(CLONE),
        "--run-dir",
        str(RUN_DIR),
        "--generations",
        GENERATIONS,
        "--games-per-generation",
        "32",
        "--promotion-games",
        GAMES,
        "--sims",
        SIMS,
        "--workers",
        WORKERS,
        "--device",
        "auto",
    ]
    if gap_json.is_file():
        train.extend(["--gap-json", str(gap_json)])
    return _run(train)


if __name__ == "__main__":
    raise SystemExit(main())
