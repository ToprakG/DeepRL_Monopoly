"""
Turn raw EXPO self-play into a distillation corpus.

``collect.py`` produces one row per non-forced EXPO decision. That is not
yet a dataset. REPO_STUDY_NOTES.md section 10 lists what a usable teacher
corpus needs, and this module supplies it:

* **Legal masks.** A student trained without the mask learns to spend
  probability mass on illegal actions. Masks are stored bit-packed.
* **Deduplication.** Monopoly revisits states constantly (every forced roll
  in an unchanged position looks identical). Duplicates inflate the corpus
  and bias it toward common, easy positions.
* **Game-level splits.** Rows from one game are highly correlated, so
  splitting by row leaks trajectories across train/val/test and inflates
  validation accuracy. Whole games go to exactly one split.
* **Family balance reporting.** The action space is 77% property-for-property
  exchanges, so an unbalanced corpus teaches trade spam. This reports the
  distribution rather than silently accepting it.
* **Provenance.** Ruleset version, dimensions, weights, and a source hash,
  so a checkpoint can be traced to the exact teacher that produced it.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

import EXPOSURE_HEURISTIC  # noqa: F401
from monopoly_game_engine.actions import ACTION_SPACE_SIZE, OFFSETS
from monopoly_game_engine.constants import RULESET_VERSION
from monopoly_game_engine.state import STATE_DIM

from EXPOSURE_HEURISTIC import agent as expo_agent

SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}

#: Decimal places kept when hashing an observation for deduplication.
QUANTISE = 2

#: Hard ceiling on any one action family's share of the training split.
#: The raw corpus is ~70% forced-ish binary actions and ~24% trade offers,
#: which teaches a student to roll and spam trades while never learning to
#: build. Capping the dominant families is cheaper than reweighting the loss
#: and keeps the label distribution honest about what EXPO actually does.
FAMILY_CAP = 0.35


def _family_of(action: int) -> str:
    bounds = sorted(OFFSETS.items(), key=lambda kv: kv[1])
    for i, (name, start) in enumerate(bounds):
        end = bounds[i + 1][1] if i + 1 < len(bounds) else ACTION_SPACE_SIZE
        if start <= action < end:
            return name
    return "unknown"


def _source_hash() -> str:
    digest = hashlib.sha256()
    here = Path(__file__).parent
    for name in sorted(("agent.py", "board_model.py", "collect.py", "distill.py")):
        path = here / name
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def build_dataset(games, out_dir: Path) -> dict:
    """Assemble, dedupe, split and write the corpus. Returns a report."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Assign every row the id of the game it came from, so splits stay whole.
    obs, act, mask, game_id = [], [], [], []
    for index, game in enumerate(games):
        if not len(game.get("obs", ())):
            continue
        obs.append(game["obs"])
        act.append(game["act"])
        mask.append(game["mask"])
        game_id.append(np.full(len(game["act"]), index, dtype=np.int32))

    if not obs:
        return {"error": "no recorded decisions"}

    obs = np.concatenate(obs)
    act = np.concatenate(act)
    mask = np.concatenate(mask)
    game_id = np.concatenate(game_id)
    raw_rows = len(act)

    # Deduplicate on a *quantised* observation. Hashing the raw floats finds
    # nothing: the vector carries continuous cash and round counters, so two
    # strategically identical positions are essentially never bit-identical.
    # Rounding to QUANTISE decimals collapses "same situation, $3 different"
    # into one row, which is what we actually want to drop.
    quantised = np.round(obs, QUANTISE)
    keys = np.array([
        hashlib.blake2b(o.tobytes() + int(a).to_bytes(4, "little"),
                        digest_size=16).digest()
        for o, a in zip(quantised, act)
    ])
    _, keep = np.unique(keys, return_index=True)
    keep.sort()
    obs, act, mask, game_id = obs[keep], act[keep], mask[keep], game_id[keep]
    deduped_rows = len(act)

    # Cap dominant action families before splitting. Downsample only; never
    # duplicate rows, which would just re-weight the same states.
    families = np.array([_family_of(int(a)) for a in act])
    rng_bal = np.random.default_rng(1)
    cap = int(FAMILY_CAP * len(act))
    drop = np.zeros(len(act), dtype=bool)
    for family in np.unique(families):
        idx = np.flatnonzero(families == family)
        if len(idx) > cap:
            drop[rng_bal.permutation(idx)[cap:]] = True
    balanced_removed = int(drop.sum())
    obs, act, mask, game_id = (
        obs[~drop], act[~drop], mask[~drop], game_id[~drop]
    )

    # Split by game, never by row.
    unique_games = np.unique(game_id)
    rng = np.random.default_rng(0)
    rng.shuffle(unique_games)
    n_train = int(len(unique_games) * SPLITS["train"])
    n_val = int(len(unique_games) * SPLITS["val"])
    assignment = {
        "train": set(unique_games[:n_train].tolist()),
        "val": set(unique_games[n_train:n_train + n_val].tolist()),
        "test": set(unique_games[n_train + n_val:].tolist()),
    }

    report = {
        "ruleset": RULESET_VERSION,
        "state_dim": int(STATE_DIM),
        "action_dim": int(ACTION_SPACE_SIZE),
        "teacher": expo_agent.POLICY_ID,
        "teacher_weights": {n: getattr(expo_agent, n) for n in expo_agent.TUNABLE},
        "source_sha256": _source_hash(),
        "games": len(games),
        "rows_raw": int(raw_rows),
        "rows_deduped": int(len(act)),
        "rows_after_dedupe": int(deduped_rows),
        "duplicate_fraction": round(1 - deduped_rows / raw_rows, 4),
        "rows_dropped_for_balance": int(balanced_removed),
        "family_cap": FAMILY_CAP,
        "quantise_decimals": QUANTISE,
        "splits": {},
    }

    for split, ids in assignment.items():
        rows = np.isin(game_id, list(ids))
        path = out_dir / f"expo_teacher_{split}.npz"
        np.savez_compressed(
            path,
            obs=obs[rows], act=act[rows],
            mask=mask[rows], game_id=game_id[rows],
        )
        families = Counter(_family_of(int(a)) for a in act[rows])
        total = max(1, int(rows.sum()))
        report["splits"][split] = {
            "games": len(ids),
            "rows": total,
            "path": str(path),
            "action_families": {
                k: round(v / total, 4)
                for k, v in families.most_common()
            },
        }

    (out_dir / "provenance.json").write_text(json.dumps(report, indent=2))
    return report


__all__ = ["build_dataset"]
