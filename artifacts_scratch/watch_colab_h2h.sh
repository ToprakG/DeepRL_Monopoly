#!/bin/bash
set -u
SESSION=oracle-v2-comp-h2h
ROOT=/Users/toprakgundogdu/DeepRL_Monopoly
MIRROR="$ROOT/artifacts_scratch/oracle_v2_comp_h2h_ckpt"
LOG="$ROOT/artifacts_scratch/watch_colab_h2h.log"
mkdir -p "$MIRROR"
while true; do
  date -u +"%Y-%m-%dT%H:%M:%SZ" | tee -a "$LOG"
  out=$(colab exec -s "$SESSION" --timeout 25 <<'PY' 2>&1
from pathlib import Path
import tarfile, os, subprocess
pid = Path("/content/job.pid").read_text().strip() if Path("/content/job.pid").exists() else "?"
alive = Path(f"/proc/{pid}").exists() if pid != "?" else False
log = Path("/content/job.log").read_text() if Path("/content/job.log").exists() else ""
last = log.strip().splitlines()[-3:] if log.strip() else []
manifest = Path("/content/h2h_ckpt/manifest.json")
n = "?"
if manifest.exists():
    import json
    n = json.loads(manifest.read_text())["n"]
src = Path("/content/h2h_ckpt")
outp = Path("/content/h2h_ckpt.tar.gz")
if src.exists() and any(src.iterdir()):
    with tarfile.open(outp, "w:gz") as a:
        a.add(src, arcname="h2h_ckpt")
    print("TAR_OK", outp.stat().st_size)
else:
    print("TAR_EMPTY")
print("pid", pid, "alive", alive, "completed", n)
print("LAST", " | ".join(last))
if Path("/content/h2h_ckpt/final.json").exists():
    print("FINAL_READY")
PY
)
  echo "$out" | tee -a "$LOG"
  if echo "$out" | grep -q "appears to be lost\|not found\|404\|401"; then
    echo "DEAD" | tee -a "$LOG"
    exit 2
  fi
  if echo "$out" | grep -q TAR_OK; then
    colab download -s "$SESSION" /content/h2h_ckpt.tar.gz "$ROOT/artifacts_scratch/h2h_ckpt.tar.gz" >/dev/null
    rm -rf "$MIRROR"
    mkdir -p "$MIRROR"
    tar -xzf "$ROOT/artifacts_scratch/h2h_ckpt.tar.gz" -C "$MIRROR" --strip-components=1
  fi
  if echo "$out" | grep -q FINAL_READY; then
    echo "DONE" | tee -a "$LOG"
    exit 0
  fi
  sleep 180
done
