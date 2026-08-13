#!/usr/bin/env python3
"""48-game clone gate: vs fixed and vs ASU, at 32 and 128 sims. Sequential."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "monopoly_bench/runs/oracle_hybrid_bc_new25k/snapshots/hybrid_clone_0000.pt"
OUT_DIR = ROOT / "artifacts_scratch"
H2H = ROOT / "artifacts_scratch/hybrid_clone_vs_fixed.py"
SUMMARY = OUT_DIR / "clone_gate_48_summary.json"

CELLS = (
    ("fixed", 32),
    ("asu", 32),
    ("fixed", 128),
    ("asu", 128),
)


def cell_path(vs: str, sims: int) -> Path:
    return OUT_DIR / f"hybrid_clone_new25k_vs_{vs}_48_sims{sims}.json"


def write_summary(rows: list[dict], started: float) -> None:
    payload = {
        "checkpoint": str(CKPT),
        "games": 48,
        "seed": 5_300_000,
        "workers": 8,
        "elapsed_s": time.perf_counter() - started,
        "cells": rows,
    }
    SUMMARY.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    if not CKPT.is_file():
        raise SystemExit(f"missing checkpoint: {CKPT}")
    started = time.perf_counter()
    rows: list[dict] = []
    write_summary(rows, started)
    for vs, sims in CELLS:
        out = cell_path(vs, sims)
        print(f"\n=== START vs={vs} sims={sims} -> {out.name} ===", flush=True)
        cmd = [
            sys.executable,
            str(H2H),
            "--checkpoint",
            str(CKPT),
            "--vs",
            vs,
            "--games",
            "48",
            "--workers",
            "8",
            "--sims",
            str(sims),
            "--seed",
            "5300000",
            "--quiet",
            "--output",
            str(out),
        ]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
        wall = time.perf_counter() - t0
        if proc.returncode != 0:
            rows.append(
                {
                    "vs": vs,
                    "sims": sims,
                    "ok": False,
                    "returncode": proc.returncode,
                    "wall_s": wall,
                    "output": str(out),
                }
            )
            write_summary(rows, started)
            raise SystemExit(f"cell vs={vs} sims={sims} failed rc={proc.returncode}")
        report = json.loads(out.read_text())
        row = {
            "vs": vs,
            "sims": sims,
            "ok": True,
            "wins": report["clone_wins"],
            "games": report["games"],
            "win_rate": report["summary"]["win_rate"],
            "wilson_lower": report["summary"]["wilson_lower"],
            "completed": report["summary"]["completed"],
            "crashes": report["summary"]["crashes"],
            "illegal_actions": report["summary"]["illegal_actions"],
            "n_searches": report.get("n_searches"),
            "latency_mean_s": report.get("latency_mean_s"),
            "latency_p50_s": report.get("latency_p50_s"),
            "latency_p95_s": report.get("latency_p95_s"),
            "wall_s": report["wall_seconds"],
            "output": str(out),
        }
        rows.append(row)
        write_summary(rows, started)
        print(
            (
                f"=== DONE vs={vs} sims={sims} "
                f"{row['wins']}/{row['games']} WR={row['win_rate']:.3f} "
                f"wilson_lo={row['wilson_lower']:.3f} p95={row['latency_p95_s']:.3f}s ==="
            ),
            flush=True,
        )
    print(f"\nall cells wrote {SUMMARY}", flush=True)
    print(json.dumps({"cells": rows}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
