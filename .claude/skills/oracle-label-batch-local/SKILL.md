---
name: oracle-label-batch-local
description: Use when generating oracle MCTS teacher labels on the LOCAL machine (not Colab) in this repo (DeepRL_Monopoly) via oracle.label_gen, babysitting a long-running local background label-gen batch to completion, resuming one that got killed, or producing a chart+decoded-action report for a finished local label batch. For generating labels on Colab instead, see the oracle-label-batch-colab skill.
---

# Oracle Label Batch (Local)

## Overview

Repeatable workflow for running `oracle.label_gen` locally in fixed-size
iterations, surviving background-process kills without losing or duplicating
work, and producing a chart + decoded-action report per iteration.

## When to use

- "run N games with oracle to generate labels locally"
- "run the next iteration" / "keep going" / "regenerate the data"
- A previous label-gen background run got killed and needs resuming

## Workflow

1. **Pick a unique seed range.** List every existing
   `oracle_labels/labels_seed*.json` (and Colab job configs under
   `monopoly_bench/colab/jobs/*.json`, which use a separate range) to find
   every seed range already used. Pick the next contiguous, non-overlapping
   block of exactly **200 games**, starting right after the highest seed
   used so far. Never reuse or overlap a seed range — that wastes games on
   duplicate data.

2. **Launch in background** (Bash, `run_in_background: true`):
   ```
   python -m oracle.label_gen --calibrate --games 200 --seed <SEED> \
     --output oracle_labels/labels_seed<LO>_<HI>.json 2>&1 \
     | tee oracle_labels/run_seed<LO>_<HI>.log
   ```
   `--calibrate` = the standard config for this repo's batches (128 sims,
   checkpoint-only labels at buy/build/trade/auction, self+pool lineup mix)
   — don't change it unless asked. Checkpointing (`--checkpoint-every 25`,
   default) is on automatically, no flag needed.

3. **Poll and report frequently.** Background runs on this machine get
   killed externally from time to time — cause unconfirmed, seen across
   long unattended runs, not caused by the script. Poll via `tail` on the
   log file plus `ScheduleWakeup` every 5–10 min; report `X/200 games` to
   the user each time if they've asked to be kept updated.

4. **Auto-resume on kill — don't ask permission.** If a task notification
   reports `status: killed`, relaunch immediately:
   ```
   python -m oracle.label_gen --calibrate --games 200 --seed <SEED> \
     --output oracle_labels/labels_seed<LO>_<HI>.json --resume 2>&1 \
     | tee -a oracle_labels/run_seed<LO>_<HI>.log
   ```
   `--resume` skips seeds already recorded in
   `oracle_labels/labels_seed<LO>_<HI>.ckpt/manifest.json`, so nothing gets
   rerun. Keep relaunching until it exits 0 — every 25 games is checkpointed
   regardless of how the process dies, so progress is never lost.

5. **On true completion**, verify `oracle_labels/labels_seed<LO>_<HI>.{json,npz,jsonl}`
   exist and cover 200 unique games. The final save auto-merges every
   checkpoint part (`merge_checkpoint_dir()` in `oracle/label_gen.py`), so
   this is correct even after several kill+resume cycles. **Check
   completeness by counting unique seeds in `game_summaries`
   (`len({g["seed"] for g in meta["game_summaries"]})`), not the top-level
   `games_completed` field** — a file merged by hand before this skill
   existed (`labels_seed30400_30799.json`) had a stale `games_completed: 25`
   left over from an early manual merge, even though `game_summaries` had
   all 400 games correctly. `merge_checkpoint_dir()` itself sets
   `games_completed` correctly now, but treat the field as a hint, not proof.

6. **Generate the report:**
   ```
   python -m oracle.report_labels --labels oracle_labels/labels_seed<LO>_<HI>.json
   ```
   Writes `<stem>.report.html` (seat win-rate/value chart, labels-per-game
   histogram, decoded top-actions chart+table) and `<stem>.actions.md`
   (plain decoded action table) next to the input.

7. **Send both files to the user** via SendUserFile (`status: proactive`),
   and report final stats: games, labels, wall time, throughput.

## Quick reference

| Step | Command |
|---|---|
| Launch | `python -m oracle.label_gen --calibrate --games 200 --seed S --output oracle_labels/labels_seedS_E.json` (background) |
| Resume after kill | same + `--resume`, `tee -a` |
| Report | `python -m oracle.report_labels --labels oracle_labels/labels_seedS_E.json` |

## Common mistakes

- **Reusing/overlapping a seed range** — check every existing
  `oracle_labels/labels_seed*.json` (and Colab job configs) before picking
  the next block.
- **Waiting for permission to resume after a kill** — don't. Relaunch with
  `--resume` and just report it.
- **Trusting a resumed run's final output without the merge fix** — before
  the checkpoint-merge fix, `--resume`'s final save only held the last
  invocation's games, not the full set. Fixed in `oracle/label_gen.py`; on
  an older checkout, confirm with `grep merge_checkpoint_dir oracle/label_gen.py`
  before trusting a resumed run's output.
- **Changing `--calibrate` config** without being asked — it's the
  established config for this repo's batches.
