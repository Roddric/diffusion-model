"""Joint standardized-innovation reconstruction models."""

import numpy as np
import pandas as pd
import pytest

from reconstruction.dependent_innovations import DependentInnovationModel


def _innovations(seed=7, n_days=800, n_assets=6):
    rng = np.random.default_rng(seed)
    common = rng.standard_t(7, size=(n_days, 1))
    noise = rng.normal(size=(n_days, n_assets))
    values = 0.65 * common + 0.35 * noise
    return pd.DataFrame(
        values, columns=[f"stock_{index}" for index in range(n_assets)]
    )


def test_joint_student_t_is_seeded_and_preserves_shape():
    model = DependentInnovationModel().fit(_innovations())
    first = model.sample_student_t(4, 12, seed=11)
    second = model.sample_student_t(4, 12, seed=11)

    assert first.shape == (4, 12, 6)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first, second)
    assert 4.5 <= model.student_df_ <= 30.0


def test_joint_student_t_retains_positive_cross_asset_dependence():
    model = DependentInnovationModel().fit(_innovations())
    draws = model.sample_student_t(2000, 2, seed=3).reshape(-1, 6)
    correlation = np.corrcoef(draws, rowvar=False)

    assert correlation[np.triu_indices(6, 1)].mean() > 0.25


def test_moving_blocks_are_observed_centered_sequences():
    innovations = _innovations(n_days=100)
    model = DependentInnovationModel().fit(innovations)
    paths = model.sample_blocks(3, 10, block_length=5, seed=19)

    assert paths.shape == (3, 10, 6)
    for path in paths:
        for offset in (0, 5):
            block = path[offset:offset + 5]
            assert any(
                np.allclose(block, model.values_[start:start + 5])
                for start in range(len(model.values_) - 4)
            )


def test_sampling_validates_inputs():
    model = DependentInnovationModel()
    with pytest.raises(RuntimeError):
        model.sample_student_t(1, 2)
    model.fit(_innovations(n_days=20))
    with pytest.raises(ValueError):
        model.sample_blocks(1, 2, block_length=100)
