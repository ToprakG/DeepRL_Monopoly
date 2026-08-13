"""Value-coverage toggles: denser routine labels + backup/winner blend."""

from __future__ import annotations

import numpy as np
import pytest

from oracle.hybrid_config import (
    BROAD_VALUE_OUTCOME_MIX,
    DEFAULT_VALUE_OUTCOME_MIX,
    blend_value_vectors,
    resolve_value_outcome_mix,
)


def test_resolve_value_outcome_mix_toggle_order():
    assert resolve_value_outcome_mix() == DEFAULT_VALUE_OUTCOME_MIX
    assert resolve_value_outcome_mix(broad_value=True) == BROAD_VALUE_OUTCOME_MIX
    assert resolve_value_outcome_mix(blend_outcomes=True) == BROAD_VALUE_OUTCOME_MIX
    assert resolve_value_outcome_mix(blend_outcomes=False, broad_value=True) == 0.0
    assert resolve_value_outcome_mix(mix=0.3, blend_outcomes=True) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        resolve_value_outcome_mix(mix=1.5)


def test_blend_mix_zero_keeps_backups():
    backups = np.array([[0.7, 0.1, 0.1, 0.1], [0.2, 0.5, 0.2, 0.1]], dtype=np.float32)
    outcomes = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    blended = blend_value_vectors(backups, outcomes, 0.0)
    np.testing.assert_allclose(blended, backups)


def test_blend_mix_one_uses_winners():
    backups = np.array([[0.7, 0.1, 0.1, 0.1]], dtype=np.float32)
    outcomes = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    blended = blend_value_vectors(backups, outcomes, 1.0)
    np.testing.assert_allclose(blended, outcomes)


def test_blend_halfway_and_truncated_keeps_backup():
    backups = np.array(
        [[0.7, 0.1, 0.1, 0.1], [0.4, 0.3, 0.2, 0.1]],
        dtype=np.float32,
    )
    outcomes = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    blended = blend_value_vectors(backups, outcomes, 0.5)
    np.testing.assert_allclose(blended[0], [0.85, 0.05, 0.05, 0.05])
    np.testing.assert_allclose(blended[1], backups[1])
