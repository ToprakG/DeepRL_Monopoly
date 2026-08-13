"""Poll Colab field-br oracle-net gap/self-play and print a live status board."""

from __future__ import annotations

import subprocess
import time

POLL_S = 20
SESSION = "field-br"

REMOTE = r"""
import glob, json, os, subprocess
print("=== field-br oracle net (32 sims) ===")
log = "/content/field_br_job.log"
if os.path.isfile(log):
    lines = open(log, errors="replace").read().splitlines()
    for ln in lines[-18:]:
        print(ln[:220])
print("--- live ---")
r = subprocess.run(["ps", "aux"], capture_output=True, text=True)
for ln in r.stdout.splitlines():
    if "run_field_br" in ln or ("field_br" in ln and "python" in ln):
        print("JOB", ln[80:240] if len(ln) > 80 else ln)
print("--- ckpts ---")
for path in sorted(glob.glob("/content/oracle-field-br-run/gap/*/summary.json")):
    payload = json.load(open(path))
    wr = payload.get("learner_win_rate")
    name = os.path.basename(os.path.dirname(path))
    print(f"  gap/{name}: WR={wr} completed={payload.get('completed')} / {payload.get('games_scheduled')}")
hist = "/content/oracle-field-br-run/reports/field_history.json"
if os.path.isfile(hist):
    print("history", open(hist).read()[:500])
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
