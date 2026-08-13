"""Sequential Wave 0 + Wave 1 H2H jobs. One field at a time so 8 cores survive."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
LOG_DIR = ROOT / "artifacts_scratch"
WORKERS = "4"
GAMES = "48"

# Isolated Inncenta matchup, paired across leaves.
SEED_0C = "7002000"
LINEUP_0C = "oracle-fast-v1,inncenta-heuristic,fixed-b,fixed-c"
CLONE_CKPT = str(
    ROOT / "monopoly_bench/runs/oracle_hybrid_bc_new25k/snapshots/hybrid_clone_0000.pt"
)


def _oracle_base(seed: str, tag: str, lineup: str, live: bool) -> list[str]:
    argv = [
        PYTHON,
        "-u",
        "-m",
        "oracle.eval_h2h",
        "--games",
        GAMES,
        "--seed",
        seed,
        "--workers",
        WORKERS,
        "--sims",
        "128",
        "--horizon",
        "16",
        "--rollouts",
        "1",
        "--deadline-s",
        "0",
        "--early-stop-lead",
        "0",
        "--game-timeout-s",
        "0",
        "--resume",
        "--pretty",
        "--lineup",
        lineup,
        "--checkpoint-dir",
        str(LOG_DIR / f"{tag}_ckpt"),
        "--output",
        str(LOG_DIR / f"{tag}.json"),
    ]
    if live:
        argv.extend(["--live", "--turn-deadline-s", "4"])
    return argv


JOBS: list[tuple[str, list[str]]] = [
    (
        "h2h_0b_indep",
        [
            PYTHON,
            "-u",
            "-m",
            "oracle.eval_h2h",
            "--games",
            GAMES,
            "--seed",
            "7001000",
            "--workers",
            WORKERS,
            "--game-timeout-s",
            "0",
            "--resume",
            "--pretty",
            "--lineup",
            "inncenta-heuristic,alinebidal-final,expo-heuristic-v1,slayer-v1",
            "--checkpoint-dir",
            str(LOG_DIR / "h2h_0b_indep_ckpt"),
            "--output",
            str(LOG_DIR / "h2h_0b_indep.json"),
        ],
    ),
    (
        "h2h_0c_rollout",
        _oracle_base(SEED_0C, "h2h_0c_rollout", LINEUP_0C, live=True)
        + ["--leaf", "rollout"],
    ),
    (
        "h2h_1c_networth",
        _oracle_base(SEED_0C, "h2h_1c_networth", LINEUP_0C, live=True)
        + ["--leaf", "networth"],
    ),
    (
        "h2h_1c_clone",
        _oracle_base(SEED_0C, "h2h_1c_clone", LINEUP_0C, live=True)
        + ["--leaf", "clone", "--leaf-checkpoint", CLONE_CKPT],
    ),
    (
        "h2h_1c_asu",
        _oracle_base(SEED_0C, "h2h_1c_asu", LINEUP_0C, live=True)
        + ["--leaf", "asu"],
    ),
    (
        "h2h_1c_asu_plus",
        _oracle_base(SEED_0C, "h2h_1c_asu_plus", LINEUP_0C, live=True)
        + ["--leaf", "asu_plus"],
    ),
    (
        "h2h_0a_live_field",
        _oracle_base(
            "7000000",
            "h2h_0a_live_field",
            "oracle-fast-v1,inncenta-heuristic,alinebidal-final,slayer-v1",
            live=True,
        )
        + ["--leaf", "rollout"],
    ),
]


def _run_job(name: str, argv: list[str]) -> int:
    log_path = LOG_DIR / f"{name}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== {name} start {started} pid={os.getpid()} ===\n")
        log.write(" ".join(argv) + "\n")
        log.flush()
        proc = subprocess.run(
            argv,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(f"\n=== {name} exit={proc.returncode} ===\n")
        log.flush()
    print(f"{name} exit={proc.returncode}", flush=True)
    return int(proc.returncode)


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    codes = []
    for name, argv in JOBS:
        print(f"starting {name}", flush=True)
        codes.append((name, _run_job(name, argv)))
    failed = [name for name, code in codes if code != 0]
    for name, code in codes:
        print(f"  {name}: {code}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
