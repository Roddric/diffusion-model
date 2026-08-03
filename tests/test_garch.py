"""GARCH residual sampling."""

import numpy as np
import pandas as pd
import pytest

from residuals.garch import GARCHModeler


@pytest.fixture
def fitted(config):
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2021-01-01", periods=400)
    resid = pd.DataFrame(
        rng.standard_t(6, (400, 3)) * 0.01,
        index=idx,
        columns=["600000", "600001", "600002"],
    )
    gm = GARCHModeler(config)
    gm.fit_all(resid)
    return gm


def test_sample_returns_full_path(fitted):
    """Regression: indexing the arch simulation array as [0, :, 0] returns a single
    scalar instead of the horizon-length path."""
    path = fitted.sample("600000", horizon=50)

    assert path.shape == (50,)
    assert len(np.unique(path)) > 1, "path is constant -- likely a broadcast scalar"


def test_sample_all_has_one_row_per_sample(fitted):
    """Regression: a 1-row frame silently broadcasts, giving every generated sample the
    same idiosyncratic residual (which kills fat tails and biases the mean)."""
    draws = fitted.sample_all(horizon=200)

    assert len(draws) == 200
    assert (draws.nunique() > 1).all(), "residuals are constant across samples"


def test_sampled_residuals_are_plausibly_scaled(fitted):
    draws = fitted.sample_all(horizon=500)
    # Fitted on ~1% daily residuals; sampled residuals should be the same order.
    assert 0.001 < draws.std().mean() < 0.1


def test_sampling_seed_is_fully_reproducible(fitted):
    first = fitted.sample_all(horizon=40, seed=17)
    second = fitted.sample_all(horizon=40, seed=17)
    different = fitted.sample_all(horizon=40, seed=18)

    np.testing.assert_allclose(first.values, second.values)
    assert not np.allclose(first.values, different.values)
