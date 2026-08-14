#!/bin/zsh
set -euo pipefail
SESSION=goat-900
LOCAL=/Users/toprakgundogdu/DeepRL_Monopoly/artifacts_scratch/goat900
mkdir -p "$LOCAL"
: > "$LOCAL/watch.log"
deadline=$(( $(date +%s) + 150 * 60 ))
last_ckpt_pull=0

while true; now=$(date +%s); do
  if (( now > deadline )); then
    echo "TIMEOUT" | tee -a "$LOCAL/watch.log"
    exit 2
  fi
  out=$(colab exec -s "$SESSION" --timeout 40 <<'PY' 2>&1 || true
from pathlib import Path
pid = Path("/content/goat900.pid").read_text().strip() if Path("/content/goat900.pid").exists() else "?"
alive = Path(f"/proc/{pid}").exists() if pid.isdigit() else False
log = Path("/content/goat900.log").read_text() if Path("/content/goat900.log").exists() else ""
ckpt = Path("/content/goat900_ckpt")
n = len(list(ckpt.rglob("game_*.json"))) if ckpt.exists() else 0
json_ready = Path("/content/goat900.json").exists()
lines = [ln for ln in log.splitlines() if ln.strip() and not ln.startswith("nohup")]
tail = lines[-6:] if lines else []
print(f"ALIVE={alive} PID={pid} CKPT={n} JSON={json_ready}")
for ln in tail:
    print(ln[:180])
PY
)
  ts=$(date +%H:%M:%S)
  echo "[$ts] $out" | tee -a "$LOCAL/watch.log"
  if echo "$out" | grep -q "JSON=True"; then
    echo "DONE" | tee -a "$LOCAL/watch.log"
    exit 0
  fi
  if echo "$out" | grep -Eq "appears to be lost|Session .* not found|404"; then
    echo "SESSION_DEAD" | tee -a "$LOCAL/watch.log"
    exit 3
  fi
  if echo "$out" | grep -q "ALIVE=False"; then
    echo "PROCESS_DEAD" | tee -a "$LOCAL/watch.log"
    exit 4
  fi
  if (( now - last_ckpt_pull > 240 )); then
    colab exec -s "$SESSION" --timeout 60 <<'PY' >/dev/null 2>&1 || true
import tarfile
from pathlib import Path
p = Path("/content/goat900_ckpt")
if p.exists() and any(p.rglob("game_*.json")):
    with tarfile.open("/content/goat900_ckpt.tgz", "w:gz") as a:
        a.add(str(p), arcname="goat900_ckpt")
    print("tarred")
PY
    colab download -s "$SESSION" /content/goat900_ckpt.tgz "$LOCAL/goat900_ckpt.tgz" >/dev/null 2>&1 || true
    last_ckpt_pull=$now
  fi
  sleep 90
done
