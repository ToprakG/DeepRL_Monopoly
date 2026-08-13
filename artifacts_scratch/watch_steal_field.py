"""Poll Colab steal-field unique-field job and print a live status board."""

from __future__ import annotations

import subprocess
import sys
import time

POLL_S = 20
SESSION = "steal-field"

REMOTE = r"""
import glob, json, os, subprocess
print("=== steal-field unique-field ===")
log = "/content/job.log"
if os.path.isfile(log):
    for ln in open(log, errors="replace").read().splitlines():
        print(ln)
print("--- live ---")
r = subprocess.run(["ps", "aux"], capture_output=True, text=True)
for ln in r.stdout.splitlines():
    if "eval_h2h" in ln and "multiprocessing" not in ln:
        idx = ln.find("oracle.eval_h2h")
        print("H2H", ln[idx:idx + 200] if idx >= 0 else ln[:200])
root = "/content/DeepRL_Monopoly/artifacts_scratch/unique_field"
for path in sorted(glob.glob(root + "/*/summary.json")):
    payload = json.load(open(path))
    wr = (payload.get("win_rates") or {}).get("oracle-plus-v1") or {}
    name = os.path.basename(os.path.dirname(path)).replace("_ckpt", "")
    wins = wr.get("wins")
    games = payload.get("completed_games")
    rate = wr.get("win_rate")
    rate_s = "n/a" if rate is None else f"{100 * rate:.1f}%"
    print(f"  {name}: {wins}/{games}  WR={rate_s}")
print("=== end ===")
"""


def poll() -> str:
    proc = subprocess.run(
        ["colab", "exec", "-s", SESSION, "--timeout", "25"],
        input=REMOTE,
        text=True,
        capture_output=True,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return out.strip() or f"(empty, exit={proc.returncode})"


def main() -> int:
    print(f"watching {SESSION} every {POLL_S}s  (Ctrl-C to stop)\n", flush=True)
    while True:
        print(time.strftime("%H:%M:%S"), flush=True)
        print(poll(), flush=True)
        print(flush=True)
        time.sleep(POLL_S)


if __name__ == "__main__":
    raise SystemExit(main())
