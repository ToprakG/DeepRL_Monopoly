"""Checkpoint flush + resume for oracle label_gen."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from monopoly_bench.engine import ACTION_SPACE_SIZE, STATE_DIM
from oracle.agent import OracleConfig
from oracle.hybrid_config import HybridLabelConfig
from oracle.label_gen import _flush_checkpoint_part, _load_manifest, merge_checkpoint_dir


def _fake_game(game_id: int, seed: int, n_labels: int = 2) -> dict:
    records = []
    for step in range(n_labels):
        legal = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        legal[1] = True
        records.append(
            {
                "game_id": game_id,
                "seed": seed,
                "step": step,
                "actor": 0,
                "selected_action": 1,
                "state": np.zeros(STATE_DIM, dtype=np.float32),
                "legal_mask": legal,
                "visit_distribution": {1: 1.0},
                "backed_up_value_vector": [0.25, 0.25, 0.25, 0.25],
                "simulations": 128,
                "winner": 0,
                "checkpoint": True,
                "outcome": [1.0, 0.0, 0.0, 0.0],
            }
        )
    return {
        "game_id": game_id,
        "seed": seed,
        "winner": 0,
        "truncated": False,
        "records": records,
        "n_labels": n_labels,
        "checkpoints_seen": n_labels,
        "lineup_kind": "self",
        "elapsed_seconds": 0.1,
    }


def test_checkpoint_flush_and_resume_manifest(tmp_path: Path | None = None):
    root = tmp_path if tmp_path is not None else Path("/tmp/oracle_ckpt_test")
    if tmp_path is None:
        import tempfile

        root = Path(tempfile.mkdtemp())
    cfg = OracleConfig(simulations=8, rollout_horizon=4, rollouts_per_leaf=1)
    hybrid = HybridLabelConfig()
    ckpt = root / "ckpt"
    manifest = _load_manifest(ckpt)
    batch = [_fake_game(i, 1000 + i) for i in range(3)]
    _flush_checkpoint_part(
        checkpoint_dir=ckpt,
        batch=batch,
        games_requested=10,
        seed=1000,
        mode="hybrid",
        config=cfg,
        hybrid=hybrid,
        wall_elapsed=1.0,
        workers=1,
        manifest=manifest,
    )
    loaded = _load_manifest(ckpt)
    assert loaded["completed_seeds"] == [1000, 1001, 1002]
    assert len(loaded["parts"]) == 1
    assert (ckpt / f"{loaded['parts'][0]}.npz").exists()
    batch2 = [_fake_game(3, 1003)]
    _flush_checkpoint_part(
        checkpoint_dir=ckpt,
        batch=batch2,
        games_requested=10,
        seed=1000,
        mode="hybrid",
        config=cfg,
        hybrid=hybrid,
        wall_elapsed=2.0,
        workers=1,
        manifest=loaded,
    )
    loaded2 = _load_manifest(ckpt)
    assert loaded2["completed_seeds"] == [1000, 1001, 1002, 1003]


def test_merge_checkpoint_dir_covers_every_flushed_part(tmp_path: Path | None = None):
    root = tmp_path if tmp_path is not None else Path("/tmp/oracle_ckpt_merge_test")
    if tmp_path is None:
        import tempfile

        root = Path(tempfile.mkdtemp())
    cfg = OracleConfig(simulations=8, rollout_horizon=4, rollouts_per_leaf=1)
    hybrid = HybridLabelConfig()
    ckpt = root / "ckpt"
    manifest = _load_manifest(ckpt)

    # Simulate a kill + resume: two separate flushes, as if from two processes.
    batch1 = [_fake_game(i, 2000 + i) for i in range(3)]
    _flush_checkpoint_part(
        checkpoint_dir=ckpt, batch=batch1, games_requested=5, seed=2000,
        mode="hybrid", config=cfg, hybrid=hybrid, wall_elapsed=1.0, workers=1,
        manifest=manifest,
    )
    manifest = _load_manifest(ckpt)
    batch2 = [_fake_game(i, 2000 + i) for i in range(3, 5)]
    _flush_checkpoint_part(
        checkpoint_dir=ckpt, batch=batch2, games_requested=5, seed=2000,
        mode="hybrid", config=cfg, hybrid=hybrid, wall_elapsed=1.0, workers=1,
        manifest=manifest,
    )

    output = root / "labels_merged.json"
    merged = merge_checkpoint_dir(ckpt, output)
    assert merged["games_completed"] == 5
    assert merged["n_labels"] == 5 * 2  # 2 labels per fake game
    assert sorted(g["seed"] for g in merged["game_summaries"]) == [2000, 2001, 2002, 2003, 2004]

    with np.load(output.with_suffix(".npz")) as data:
        assert data["states"].shape[0] == merged["n_labels"]
    jsonl_lines = output.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()
    assert len(jsonl_lines) == merged["n_labels"]


if __name__ == "__main__":
    test_checkpoint_flush_and_resume_manifest()
    test_merge_checkpoint_dir_covers_every_flushed_part()
    print("ok")
