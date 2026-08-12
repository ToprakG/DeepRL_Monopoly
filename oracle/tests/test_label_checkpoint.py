"""Checkpoint flush + resume for oracle label_gen."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from monopoly_bench.engine import ACTION_SPACE_SIZE, STATE_DIM
from oracle.agent import OracleConfig
from oracle.hybrid_config import HybridLabelConfig
from oracle.label_gen import _flush_checkpoint_part, _load_manifest


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


if __name__ == "__main__":
    test_checkpoint_flush_and_resume_manifest()
    print("ok")
