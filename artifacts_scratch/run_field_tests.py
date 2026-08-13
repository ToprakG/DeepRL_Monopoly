"""Actual competitor-field tests for oracle-plus-v1.

Pass = OUR win rate >= 50% in a 4-player field of real clones
(no fixed-b/fixed-c). Isolated Inncenta+scripted does not count.
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
LOG_DIR = ROOT / "artifacts_scratch"
WORKERS = "6"
PASS_WR = 0.50
MASTER = LOG_DIR / "field_tests.log"

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

RECIPES: list[tuple[str, list[str]]] = [
    ("heavy", HEAVY),
    (
        "denial",
        [
            "--leaf",
            "asu",
            "--one-ply",
            "--denial",
            "--no-solvency",
            "--completing-trade",
            "--auction",
            "--inncenta-trade",
        ],
    ),
    (
        "plain",
        [
            "--leaf",
            "asu",
            "--one-ply",
            "--no-denial",
            "--no-solvency",
            "--completing-trade",
            "--auction",
            "--inncenta-trade",
        ],
    ),
    (
        "blend",
        [
            "--leaf",
            "asu",
            "--one-ply",
            "--denial",
            "--no-solvency",
            "--completing-trade",
            "--auction",
            "--inncenta-trade",
            "--networth-mix",
            "0.5",
        ],
    ),
    (
        "cheap_trade",
        [
            "--leaf",
            "asu",
            "--one-ply",
            "--denial",
            "--no-solvency",
            "--completing-trade",
            "--auction",
            "--no-inncenta-trade",
        ],
    ),
]

MAIN = "oracle-plus-v1,alinebidal-final,slayer-v1,inncenta-heuristic"
COVERAGE = [
    ("expo", "oracle-plus-v1,alinebidal-final,slayer-v1,expo-heuristic-v1"),
    ("boom", "oracle-plus-v1,alinebidal-final,slayer-v1,boom-hybrid"),
    ("code", "oracle-plus-v1,alinebidal-final,slayer-v1,code-exposure"),
    ("inn_boom", "oracle-plus-v1,alinebidal-final,inncenta-heuristic,boom-hybrid"),
    ("slayer_expo", "oracle-plus-v1,slayer-v1,inncenta-heuristic,expo-heuristic-v1"),
    ("weak3", "oracle-plus-v1,boom-hybrid,expo-heuristic-v1,code-exposure"),
]


def _busy() -> bool:
    try:
        out = subprocess.check_output(["pgrep", "-fl", "oracle.eval_h2h"], text=True)
    except subprocess.CalledProcessError:
        return False
    return any("oracle.eval_h2h" in line for line in out.splitlines())


def _wait_for_eval() -> None:
    if not _busy():
        return
    print("waiting for in-flight eval_h2h to finish", flush=True)
    while _busy():
        time.sleep(15)
    print("eval_h2h free", flush=True)


def _h2h_cmd(
    tag: str, field: str, extra: list[str], seed: int, games: int, ckpt: Path | None = None
) -> list[str]:
    return [
        PYTHON,
        "-u",
        "-m",
        "oracle.eval_h2h",
        "--games",
        str(games),
        "--seed",
        str(seed),
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
        field,
        "--checkpoint-dir",
        str(ckpt or (LOG_DIR / f"{tag}_ckpt")),
        "--output",
        str(LOG_DIR / f"{tag}.json"),
        *extra,
    ]


def _run(tag: str, argv: list[str]) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
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
    ours = rates.get("oracle-plus-v1") or rates.get("oracle-fast-v1") or {}
    ours_wr = float(ours.get("win_rate") or 0.0)
    return {
        "winner": ranked[0][0] if ranked else "",
        "ours_wr": ours_wr,
        "ours_wins": ours.get("wins"),
        "passed_50": ours_wr >= PASS_WR,
        "ranked": [(name, row.get("wins"), row.get("win_rate")) for name, row in ranked],
        "timeouts": payload.get("timeouts"),
        "games": payload.get("completed_games"),
        "lineup": payload.get("lineup"),
    }


def _eval(tag: str, field: str, extra: list[str], seed: int, games: int, ckpt: Path | None = None) -> dict:
    argv = _h2h_cmd(tag, field, extra, seed, games, ckpt)
    print(f"starting {tag} games={games} field={field} target=OUR {PASS_WR:.0%}", flush=True)
    code = _run(tag, argv)
    out = LOG_DIR / f"{tag}.json"
    if code != 0 or not out.is_file():
        print(f"{tag} failed code={code}", flush=True)
        return {"tag": tag, "failed": True, "code": code}
    summary = _report(out)
    print(
        f"{tag} ours={summary['ours_wr']:.3f} "
        f"({summary['ours_wins']}/{summary['games']}) "
        f"winner={summary['winner']} pass50={summary['passed_50']}",
        flush=True,
    )
    with MASTER.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"tag": tag, **summary}) + "\n")
    if summary["passed_50"]:
        (LOG_DIR / "field_50_passed.json").write_text(
            json.dumps({"tag": tag, "extra": extra, **summary}, indent=2) + "\n"
        )
        print(f"PASSED 50% OUR wr recipe/field={tag} {summary['ours_wr']:.3f}", flush=True)
    return {"tag": tag, **summary}


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MASTER.open("a").write(f"\n=== field tests {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    _wait_for_eval()

    results: list[dict] = []

    # 96 games on the real table; resume the 48 already in flight.
    results.append(
        _eval(
            "field_main_heavy96",
            MAIN,
            HEAVY,
            8_200_300,
            96,
            LOG_DIR / "beat50_plus_auction_heavy_strong_ckpt",
        )
    )

    seed = 8_300_000
    for name, extra in RECIPES:
        if name == "heavy":
            continue
        results.append(_eval(f"field_main_{name}", MAIN, extra, seed, 48))
        seed += 100

    results.append(
        _eval(
            "field_main_oracle_v2",
            "oracle-fast-v1,alinebidal-final,slayer-v1,inncenta-heuristic",
            [],
            seed,
            48,
        )
    )
    seed += 100

    for name, field in COVERAGE:
        results.append(_eval(f"field_{name}_heavy", field, HEAVY, seed, 48))
        seed += 100

    (LOG_DIR / "field_tests_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    hits = [row for row in results if row.get("passed_50")]
    print(f"done fields={len(results)} hits={len(hits)}", flush=True)
    return 0 if hits else 1


if __name__ == "__main__":
    raise SystemExit(main())
