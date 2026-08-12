#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"

declare -A LOGS=(
  [oracle-labels-1]=labels_seed20000_20466
  [oracle-labels-2]=labels_seed20467_20933
  [oracle-labels-3]=labels_seed20934_21399
)
declare -A CKPT=(
  [oracle-labels-1]=/content/ckpt_seed20000_20466
  [oracle-labels-2]=/content/ckpt_seed20467_20933
  [oracle-labels-3]=/content/ckpt_seed20934_21399
)

for s in oracle-labels-1 oracle-labels-2 oracle-labels-3; do
  base="${LOGS[$s]}"
  ckpt="${CKPT[$s]}"
  echo "=== $s ==="
  echo "
import subprocess, os
pid_path = '/content/${base}.pid'
if os.path.exists(pid_path):
    pid = open(pid_path).read().strip()
    r = subprocess.run(['ps', '-o', 'pid,etimes,cmd', '-p', pid], capture_output=True, text=True)
    print('process:', r.stdout.strip() or '(not running)')
else:
    print('process: (pid file missing)')

log_path = '/content/${base}.log'
if os.path.exists(log_path):
    data = open(log_path).read()
    lines = [l for l in data.splitlines() if 'label progress' in l or 'wrote' in l or 'resume' in l]
    print('last 3:', lines[-3:])
    print('games so far:', len([l for l in lines if 'label progress' in l]))
else:
    print('(log missing)')

ckpt_dir = '${ckpt}'
if os.path.isdir(ckpt_dir):
    parts = sorted(p for p in os.listdir(ckpt_dir) if p.startswith('part_') and p.endswith('.npz'))
    print(f'checkpoint parts: {len(parts)}', parts[-3:])
else:
    print('checkpoint dir: (none yet)')
" | timeout 25 colab exec -s "$s" --timeout 15 2>&1
  echo
done
