"""Tests for the reconstruction oracle-ceiling attribution."""

import numpy as np
import pandas as pd
import pytest

from research.oracle_ceiling_attribution import (
    _metric_means_match,
    _realized_innovations,
    _shapley_values,
    _summarize_attribution,
)
from reconstruction.reconstructor import ReturnReconstructor


def test_shapley_attribution_is_efficient_for_additive_losses():
    component_reductions = {1: 0.2, 2: 0.3, 4: 0.4}
    losses = {
        mask: 1.0
        - sum(value for bit, value in component_reductions.items() if mask & bit)
        for mask in range(8)
    }

    values = _shapley_values(losses)

    assert values == pytest.approx({
        "mean_state": 0.2,
        "volatility_state": 0.3,
        "innovation": 0.4,
    })
    assert sum(values.values()) == pytest.approx(losses[0] - losses[7])


def test_realized_innovations_exactly_reconstruct_target(config):
    stocks = ["A", "B"]
    mean = {
        "loadings": pd.DataFrame(
            [[0.2], [-0.1]], index=stocks, columns=["market"]
        ),
        "intercepts": pd.Series([0.01, -0.02], index=stocks),
    }
    vol = {
        "loadings": pd.DataFrame(
            [[0.5], [-0.25]], index=stocks, columns=["logvol_pc1"]
        ),
        "log_variance_mean": pd.Series([-4.0, -3.0], index=stocks),
    }
    reconstructor = ReturnReconstructor(config, mean, vol)
    states = np.array([[0.1, -0.2], [0.2, 0.3], [-0.1, 0.0]])
    target = np.array([[0.03, -0.01], [-0.02, 0.04], [0.01, -0.03]])

    innovations = _realized_innovations(reconstructor, states, target)
    reconstructed = reconstructor.reconstruct(
        states, pd.DataFrame(innovations, columns=stocks), verbose=False
    )

    np.testing.assert_allclose(reconstructed.to_numpy(), target, atol=1e-12)


def test_bootstrap_accepts_origin_scores_as_lists():
    metrics = (
        "return_energy_score",
        "return_variogram_score",
        "daily_volatility_mae",
        "tail_quantile_error",
        "max_drawdown_error",
    )
    per_origin = {
        mask: {
            metric: [1.0 - 0.1 * bin(mask).count("1")] * 4
            for metric in metrics
        }
        for mask in range(8)
    }

    result = _summarize_attribution(
        per_origin, bootstrap_samples=10, seed=7
    )

    assert result["full_oracle_normalized_loss_reduction"] == pytest.approx(0.3)
    assert result["efficiency_check_sum_of_components"] == pytest.approx(0.3)


def test_reproduction_check_allows_only_floating_point_noise():
    metrics = (
        "return_energy_score",
        "return_variogram_score",
        "daily_volatility_mae",
        "tail_quantile_error",
        "max_drawdown_error",
    )
    expected = {metric: {"mean": 1.0} for metric in metrics}
    actual = {metric: {"mean": 1.0 + 1e-15} for metric in metrics}

    assert _metric_means_match(actual, expected)
    actual["return_energy_score"]["mean"] = 1.0 + 1e-9
    assert not _metric_means_match(actual, expected)
