"""Iterate plus-agent recipes until oracle-plus-v1 wins at least 50%.

Does not wait on Wave 0+1. Pass = win rate >= 0.50 in the isolated field:

    plus vs Inncenta / fixed-b / fixed-c

That is the field where Inncenta already scores ~48% and live Max-N does not.
After 50% there, keep going on the strong field (Alinebidal/Slayer/Inncenta)
and record whether 50% holds.
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
WORKERS = "4"
GAMES = "48"
SEED_BASE = 8_200_000
PASS_WR = 0.50

FIELD_50 = "oracle-plus-v1,inncenta-heuristic,fixed-b,fixed-c"
FIELD_STRONG = "oracle-plus-v1,alinebidal-final,slayer-v1,inncenta-heuristic"

RECIPES: list[tuple[str, list[str]]] = [
    (
        "plus_auction_denial",
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
        "plus_auction_plain",
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
        "plus_auction_heavy",
        [
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
        ],
    ),
    (
        "plus_auction_blend",
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
        "plus_auction_cheap_trade",
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


def _h2h_cmd(tag: str, field: str, extra: list[str], seed: int) -> list[str]:
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
        str(LOG_DIR / f"{tag}_ckpt"),
        "--output",
        str(LOG_DIR / f"{tag}.json"),
        *extra,
    ]


def _run(tag: str, argv: list[str], log_path: Path) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as log:
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
    winner = ranked[0][0] if ranked else ""
    return {
        "winner": winner,
        "ours_wr": ours_wr,
        "ours_wins": ours.get("wins"),
        "passed_50": ours_wr >= PASS_WR,
        "ranked": [(name, row.get("wins"), row.get("win_rate")) for name, row in ranked],
        "timeouts": payload.get("timeouts"),
        "games": payload.get("completed_games"),
    }


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    master = LOG_DIR / "beat_all.log"
    seed = SEED_BASE
    for name, extra in RECIPES:
        tag = f"beat50_{name}"
        argv = _h2h_cmd(tag, FIELD_50, extra, seed)
        seed += 100
        print(f"starting {tag} target={PASS_WR:.0%}", flush=True)
        with master.open("a", encoding="utf-8") as log:
            log.write(f"starting {tag}\n")
        code = _run(tag, argv, LOG_DIR / f"{tag}.log")
        out = LOG_DIR / f"{tag}.json"
        if code != 0 or not out.is_file():
            print(f"{tag} failed code={code}", flush=True)
            continue
        summary = _report(out)
        print(
            f"{tag} winner={summary['winner']} ours={summary['ours_wr']:.3f} "
            f"pass50={summary['passed_50']}",
            flush=True,
        )
        with master.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"tag": tag, **summary}) + "\n")
        if not summary["passed_50"]:
            continue
        strong_tag = f"beat50_{name}_strong"
        strong_argv = _h2h_cmd(strong_tag, FIELD_STRONG, extra, seed)
        seed += 100
        print(f"starting {strong_tag} (50% isolated already hit)", flush=True)
        _run(strong_tag, strong_argv, LOG_DIR / f"{strong_tag}.log")
        strong_out = LOG_DIR / f"{strong_tag}.json"
        strong = _report(strong_out) if strong_out.is_file() else {}
        (LOG_DIR / "beat_all_passed.json").write_text(
            json.dumps(
                {"recipe": name, "extra": extra, "isolated": summary, "strong": strong},
                indent=2,
            )
            + "\n"
        )
        print(f"PASSED 50% recipe={name} isolated={summary['ours_wr']:.3f}", flush=True)
        return 0
    print("no recipe reached 50% vs Inncenta+fixed", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
