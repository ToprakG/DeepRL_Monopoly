#!/bin/zsh
set -euo pipefail
cd /Users/toprakgundogdu/DeepRL_Monopoly
mkdir -p artifacts_scratch/jail_first_race_ckpt
: > artifacts_scratch/jail_first_race.log
find artifacts_scratch/jail_first_race_ckpt -name 'game_*.json' -delete 2>/dev/null || true
rm -f artifacts_scratch/jail_first_race_ckpt/summary.json artifacts_scratch/jail_first_race_ckpt/manifest.json
nohup env PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 PYTHONPATH=. \
  .venv/bin/python -u -m oracle.eval_h2h \
    --games 24 --seed 9710000 --workers 6 \
    --lineup oracle-jail-v1,slayer-v1,underdog-v1,inncenta-heuristic \
    --checkpoint-dir artifacts_scratch/jail_first_race_ckpt \
    >> artifacts_scratch/jail_first_race.log 2>&1 &
echo $! > artifacts_scratch/jail_first_race.pid
disown
echo "started pid $(cat artifacts_scratch/jail_first_race.pid)"
