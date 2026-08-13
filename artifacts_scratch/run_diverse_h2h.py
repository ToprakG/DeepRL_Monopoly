"""68 diverse H2H games vs fixed + asu-value at low sim budgets.

Runs beside the in-flight clone-field tests at 2 workers so we do not
steal the 6-worker pool. --resume on every cell; SIGKILL/segfault retry.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
LOG_DIR = ROOT / "artifacts_scratch"
MASTER = LOG_DIR / "diverse_h2h.log"
WORKERS = "2"

HEAVY = [
    "--leaf",
    "asu",
    "--one-ply",
    "--denial",
    "--denial-weight",
    "2.0",
    "--no-solvency",
    "--completing-trade",
    "--auction",
    "--inncenta-trade",
]

JOBS: list[dict] = [
    {
        "tag": "div_fixed_s8",
        "games": 16,
        "seed": 9_100_000,
        "sims": "8",
        "lineup": "oracle-fast-v1,fixed-a,fixed-b,fixed-c",
        "extra": ["--horizon", "16", "--rollouts", "1"],
    },
    {
        "tag": "div_asu_s8",
        "games": 18,
        "seed": 9_110_000,
        "sims": "8",
        "lineup": "oracle-fast-v1,asu-value-v1,fixed-a,fixed-b",
        "extra": ["--horizon", "16", "--rollouts", "1"],
    },
    {
        "tag": "div_asu_s16",
        "games": 16,
        "seed": 9_120_000,
        "sims": "16",
        "lineup": "oracle-fast-v1,asu-value-v1,fixed-a,fixed-b",
        "extra": ["--horizon", "16", "--rollouts", "1"],
    },
    {
        "tag": "div_plus_asu",
        "games": 18,
        "seed": 9_130_000,
        "sims": "8",
        "lineup": "oracle-plus-v1,asu-value-v1,fixed-a,fixed-c",
        "extra": HEAVY,
    },
]


def _cmd(job: dict) -> list[str]:
    tag = job["tag"]
    return [
        str(PYTHON),
        "-u",
        "-m",
        "oracle.eval_h2h",
        "--games",
        str(job["games"]),
        "--seed",
        str(job["seed"]),
        "--sims",
        str(job["sims"]),
        "--workers",
        WORKERS,
        "--deadline-s",
        "0",
        "--early-stop-lead",
        "0",
        "--game-timeout-s",
        "0",
        "--resume",
        "--pretty",
        "--lineup",
        job["lineup"],
        "--checkpoint-dir",
        str(LOG_DIR / f"{tag}_ckpt"),
        "--output",
        str(LOG_DIR / f"{tag}.json"),
        *job["extra"],
    ]


def _run(job: dict) -> int:
    argv = _cmd(job)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    with MASTER.open("a", encoding="utf-8") as log:
        log.write(f"\n=== {job['tag']} start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log.write(" ".join(argv) + "\n")
        log.flush()
        proc = subprocess.run(
            argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
        )
        log.write(f"\n=== {job['tag']} exit={proc.returncode} ===\n")
        log.flush()
    return int(proc.returncode)


def _report(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rates = payload.get("win_rates") or {}
    ranked = sorted(
        rates.items(),
        key=lambda item: (-(item[1].get("win_rate") or 0.0), item[0]),
    )
    ours = rates.get("oracle-fast-v1") or rates.get("oracle-plus-v1") or {}
    return {
        "winner": ranked[0][0] if ranked else "",
        "ours_wr": float(ours.get("win_rate") or 0.0),
        "ours_wins": ours.get("wins"),
        "ranked": [(name, row.get("wins"), row.get("win_rate")) for name, row in ranked],
        "timeouts": payload.get("timeouts"),
        "games": payload.get("completed_games"),
        "lineup": payload.get("lineup"),
        "sims": payload.get("config", {}).get("simulations") if isinstance(payload.get("config"), dict) else None,
    }


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MASTER.open("a", encoding="utf-8").write(
        f"\n=== diverse h2h {time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"games={sum(j['games'] for j in JOBS)} workers={WORKERS} ===\n"
    )
    print(
        f"diverse h2h starting {sum(j['games'] for j in JOBS)} games "
        f"across {len(JOBS)} fields, workers={WORKERS}",
        flush=True,
    )
    summaries: list[dict] = []
    for job in JOBS:
        tag = job["tag"]
        out = LOG_DIR / f"{tag}.json"
        attempts = 0
        failed = False
        while True:
            attempts += 1
            print(
                f"starting {tag} games={job['games']} sims={job['sims']} lineup={job['lineup']}",
                flush=True,
            )
            code = _run(job)
            if code == 0 and out.is_file():
                break
            retryable = code in {-9, -11, 137, 139}
            print(
                f"{tag} failed code={code} attempt={attempts} retryable={retryable}",
                flush=True,
            )
            if not retryable or attempts >= 8:
                summaries.append({"tag": tag, "failed": True, "code": code})
                failed = True
                break
            time.sleep(8)
        if failed or not out.is_file():
            continue
        summary = _report(out)
        print(
            f"{tag} ours={summary['ours_wr']:.3f} "
            f"({summary['ours_wins']}/{summary['games']}) winner={summary['winner']}",
            flush=True,
        )
        summaries.append({"tag": tag, **summary})
        with MASTER.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"tag": tag, **summary}) + "\n")

    (LOG_DIR / "diverse_h2h_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    failed = sum(1 for row in summaries if row.get("failed"))
    print(f"done fields={len(summaries)} failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
