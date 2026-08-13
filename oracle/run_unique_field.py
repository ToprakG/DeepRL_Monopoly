"""Unique plus recipes on the real table. Pass = OUR win rate >= 50%.

Does not wrap competitor agents. Recipes are from our H2H logs:
we bankrupt 60/96 games; fast wins are wipeouts after we built.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
LOG_DIR = ROOT / "artifacts_scratch" / "unique_field"
MASTER = LOG_DIR / "unique_field.log"
PASS_WR = 0.50
GAMES = "48"
MAIN = "oracle-plus-v1,alinebidal-final,slayer-v1,inncenta-heuristic"

BASE = [
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

RECIPES: list[tuple[str, list[str]]] = [
    ("asu_delta", BASE + ["--auction-kind", "asu_delta"]),
    ("cash_gate", BASE + ["--cash-gate"]),
    ("build_first", BASE + ["--build-first"]),
    ("race_buy", BASE + ["--race-buy"]),
    ("lethal_jail", BASE + ["--lethal-jail"]),
    (
        "survive",
        BASE
        + [
            "--auction-kind",
            "asu_delta",
            "--cash-gate",
            "--lethal-jail",
        ],
    ),
    (
        "convert",
        BASE
        + [
            "--auction-kind",
            "asu_delta",
            "--build-first",
            "--race-buy",
        ],
    ),
    (
        "all_unique",
        BASE
        + [
            "--auction-kind",
            "asu_delta",
            "--cash-gate",
            "--build-first",
            "--race-buy",
            "--lethal-jail",
        ],
    ),
]


def _workers() -> str:
    return str(max(2, min(10, (os.cpu_count() or 4) - 2)))


def _h2h_cmd(tag: str, extra: list[str], seed: int) -> list[str]:
    return [
        PYTHON,
        "-u",
        "-m",
        "oracle.eval_h2h",
        "--games",
        GAMES,
        "--seed",
        str(seed),
        "--workers",
        _workers(),
        "--deadline-s",
        "0",
        "--early-stop-lead",
        "0",
        "--game-timeout-s",
        "0",
        "--resume",
        "--pretty",
        "--lineup",
        MAIN,
        "--checkpoint-dir",
        str(LOG_DIR / f"{tag}_ckpt"),
        "--output",
        str(LOG_DIR / f"{tag}.json"),
        *extra,
    ]


def _run(tag: str, argv: list[str]) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    MASTER.parent.mkdir(parents=True, exist_ok=True)
    with MASTER.open("a", encoding="utf-8") as log:
        log.write(f"\n=== {tag} start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log.write(" ".join(argv) + "\n")
        log.flush()
        proc = subprocess.run(
            argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
        )
        log.write(f"\n=== {tag} exit={proc.returncode} ===\n")
        log.flush()
    return int(proc.returncode)


def _report(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rates = payload.get("win_rates") or {}
    ranked = sorted(
        rates.items(),
        key=lambda item: (-(item[1].get("win_rate") or 0.0), item[0]),
    )
    ours = rates.get("oracle-plus-v1") or {}
    ours_wr = float(ours.get("win_rate") or 0.0)
    return {
        "winner": ranked[0][0] if ranked else "",
        "ours_wr": ours_wr,
        "ours_wins": ours.get("wins"),
        "passed_50": ours_wr >= PASS_WR,
        "ranked": [(name, row.get("wins"), row.get("win_rate")) for name, row in ranked],
        "timeouts": payload.get("timeouts"),
        "games": payload.get("completed_games"),
    }


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MASTER.open("a").write(f"\n=== unique field {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    seed = 8_500_000
    results: list[dict] = []
    for name, extra in RECIPES:
        tag = f"unique_{name}"
        print(f"starting {tag} target=OUR {PASS_WR:.0%} field={MAIN}", flush=True)
        code = _run(tag, _h2h_cmd(tag, extra, seed))
        seed += 100
        out = LOG_DIR / f"{tag}.json"
        if code != 0 or not out.is_file():
            print(f"{tag} failed code={code}", flush=True)
            results.append({"tag": tag, "failed": True, "code": code})
            continue
        summary = _report(out)
        print(
            f"{tag} ours={summary['ours_wr']:.3f} "
            f"({summary['ours_wins']}/{summary['games']}) "
            f"winner={summary['winner']} pass50={summary['passed_50']}",
            flush=True,
        )
        row = {"tag": tag, **summary}
        results.append(row)
        with MASTER.open("a", encoding="utf-8") as log:
            log.write(json.dumps(row) + "\n")
        if summary["passed_50"]:
            (LOG_DIR / "unique_50_passed.json").write_text(
                json.dumps(row, indent=2) + "\n"
            )
            print(f"PASSED 50% OUR wr {tag} {summary['ours_wr']:.3f}", flush=True)
    (LOG_DIR / "unique_field_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    hits = [row for row in results if row.get("passed_50")]
    print(f"done recipes={len(results)} hits={len(hits)}", flush=True)
    return 0 if hits else 1


if __name__ == "__main__":
    raise SystemExit(main())
