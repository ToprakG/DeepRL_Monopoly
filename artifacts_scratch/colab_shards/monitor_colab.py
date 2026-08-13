#!/usr/bin/env python3
"""Poll Colab ASU+ H2H shards until JSON results are downloaded."""

from __future__ import annotations

import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHARD_DIR = ROOT / "artifacts_scratch" / "colab_shards"
MONITOR_LOG = SHARD_DIR / "monitor.log"
SHARDS = (
    ("asu0", 11, 0),
    ("asu1", 11, 11),
    ("asu2", 10, 22),
)
COLAB = str(Path.home() / ".local" / "bin" / "colab")
POLL_SECONDS = 120


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with MONITOR_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def probe(name: str, games: int, seed: int) -> str:
    out = f"asu_plus_h2h_{games}_seed{seed}.json"
    code = f"""
import os, subprocess
pid_path="/content/{name}.pid"
log_path="/content/{name}.run.log"
out_path="/content/{out}"
pid=open(pid_path).read().strip() if os.path.exists(pid_path) else "?"
alive=False
if pid.isdigit():
    alive=subprocess.call(["kill","-0",pid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)==0
size=os.path.getsize(out_path) if os.path.exists(out_path) else 0
tail=""
if os.path.exists(log_path):
    lines=open(log_path).read().splitlines()
    for ln in reversed(lines[-80:]):
        if "h2h progress" in ln or "DONE" in ln or "ASU+" in ln or "Error" in ln or "Traceback" in ln:
            tail=ln
            break
    if not tail and lines:
        tail=lines[-1][:160]
print(f"pid={{pid}} alive={{alive}} out_bytes={{size}} last={{tail!r}}")
"""
    try:
        proc = subprocess.run(
            [COLAB, "exec", "-s", name, "--timeout", "90"],
            input=code,
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001
        return f"PROBE_FAIL {exc}"
    text = (proc.stdout or "").strip() or (proc.stderr or "").strip()
    if proc.returncode != 0 and "pid=" not in text:
        return f"PROBE_FAIL rc={proc.returncode} {text[-300:]}"
    # keep the status line
    for line in reversed(text.splitlines()):
        if line.startswith("pid="):
            return line
    return text.splitlines()[-1] if text else "PROBE_FAIL empty"


def maybe_download(name: str, games: int, seed: int, status: str) -> None:
    out = f"asu_plus_h2h_{games}_seed{seed}.json"
    local = SHARD_DIR / out
    match = re.search(r"out_bytes=(\d+)", status)
    if not match or int(match.group(1)) <= 0 or local.exists():
        return
    subprocess.run(
        [COLAB, "download", "-s", name, f"/content/{out}", str(local)],
        check=False,
    )
    if local.exists():
        log(f"{name}: downloaded {out} ({local.stat().st_size} bytes)")


def main() -> int:
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    MONITOR_LOG.write_text("", encoding="utf-8")
    log("monitor start")
    while True:
        done = 0
        for name, games, seed in SHARDS:
            status = probe(name, games, seed)
            log(f"{name}: {status}")
            maybe_download(name, games, seed, status)
            out = SHARD_DIR / f"asu_plus_h2h_{games}_seed{seed}.json"
            if out.exists():
                done += 1
            elif "alive=False" in status and "out_bytes=0" in status:
                log(f"{name}: DEAD_NO_OUTPUT")
        log(f"local_json={done}/3")
        if done == 3:
            log("ALL_SHARDS_DOWNLOADED")
            return 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
