#!/usr/bin/env python3
"""Babysit Colab goat-900 H2H until goat900.json exists."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

SESSION = "goat-900"
LOCAL = Path("/Users/toprakgundogdu/DeepRL_Monopoly/artifacts_scratch/goat900")
LOCAL.mkdir(parents=True, exist_ok=True)
LOG = LOCAL / "watch.log"
DEADLINE = time.time() + 150 * 60
POLL_S = 90
PULL_EVERY_S = 300
last_pull = 0.0

CHECK_PY = r'''
from pathlib import Path
pid = Path("/content/goat900.pid").read_text().strip() if Path("/content/goat900.pid").exists() else "?"
alive = Path(f"/proc/{pid}").exists() if pid.isdigit() else False
log = Path("/content/goat900.log").read_text() if Path("/content/goat900.log").exists() else ""
ckpt = Path("/content/goat900_ckpt")
n = len(list(ckpt.rglob("game_*.json"))) if ckpt.exists() else 0
json_ready = Path("/content/goat900.json").exists()
lines = [ln for ln in log.splitlines() if ln.strip() and not ln.startswith("nohup")]
print(f"ALIVE={alive} PID={pid} CKPT={n} JSON={json_ready}")
for ln in lines[-6:]:
    print(ln[:180])
'''

TAR_PY = r'''
import tarfile
from pathlib import Path
p = Path("/content/goat900_ckpt")
games = list(p.rglob("game_*.json")) if p.exists() else []
if games:
    with tarfile.open("/content/goat900_ckpt.tgz", "w:gz") as a:
        a.add(str(p), arcname="goat900_ckpt")
    print(f"tarred {len(games)}")
else:
    print("empty")
'''


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def colab_exec(code: str, timeout: int = 45) -> tuple[int, str]:
    proc = subprocess.run(
        ["colab", "exec", "-s", SESSION, "--timeout", str(timeout)],
        input=code,
        text=True,
        capture_output=True,
        timeout=timeout + 30,
    )
    out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    return proc.returncode, out.strip()


while time.time() < DEADLINE:
    try:
        code, out = colab_exec(CHECK_PY, timeout=45)
    except Exception as exc:
        log(f"POLL_ERR {exc!r}")
        time.sleep(POLL_S)
        continue
    log(out.replace("\n", " | ") if out else f"empty rc={code}")
    if "JSON=True" in out:
        log("DONE")
        raise SystemExit(0)
    if "appears to be lost" in out or "not found" in out:
        log("SESSION_DEAD")
        raise SystemExit(3)
    if "ALIVE=False" in out:
        log("PROCESS_DEAD")
        raise SystemExit(4)
    now = time.time()
    if now - last_pull >= PULL_EVERY_S:
        try:
            colab_exec(TAR_PY, timeout=90)
            subprocess.run(
                [
                    "colab",
                    "download",
                    "-s",
                    SESSION,
                    "/content/goat900_ckpt.tgz",
                    str(LOCAL / "goat900_ckpt.tgz"),
                ],
                capture_output=True,
                timeout=120,
            )
            last_pull = now
            log("ckpt pull attempted")
        except Exception as exc:
            log(f"PULL_ERR {exc!r}")
    time.sleep(POLL_S)

log("TIMEOUT")
raise SystemExit(2)
