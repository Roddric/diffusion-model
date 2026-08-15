"""Tests for the post-hoc cross-market synthesis."""

import math

import numpy as np
import pytest

from research.cross_market_meta_analysis import (
    _composite_ratio,
    _dersimonian_laird,
    _equal_market_summary,
    _origin_bootstrap,
)


def test_composite_ratio_is_geometric_mean_of_coprimary_ratios():
    pool = {
        "state_energy_score": [1.0, 3.0],
        "state_rmse": [2.0, 4.0],
    }
    baseline = {
        "state_energy_score": [2.0, 6.0],
        "state_rmse": [4.0, 8.0],
    }

    assert _composite_ratio(pool, baseline) == pytest.approx(0.5)


def test_paired_origin_bootstrap_is_reproducible_and_log_scaled():
    pool = {
        "state_energy_score": [1.0, 2.0, 3.0],
        "state_rmse": [2.0, 3.0, 4.0],
    }
    baseline = {
        "state_energy_score": [2.0, 4.0, 6.0],
        "state_rmse": [4.0, 6.0, 8.0],
    }

    first = _origin_bootstrap(pool, baseline, samples=100, seed=7)
    second = _origin_bootstrap(pool, baseline, samples=100, seed=7)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first, math.log(0.5))


def test_equal_market_summary_uses_market_level_student_t_interval():
    result = _equal_market_summary(np.log([0.98, 1.00, 1.02]))

    assert result["k_markets"] == 3
    assert result["composite_ratio"] == pytest.approx(
        (0.98 * 1.00 * 1.02) ** (1 / 3)
    )
    assert result["ci_95_composite_ratio"][0] < result["composite_ratio"]
    assert result["ci_95_composite_ratio"][1] > result["composite_ratio"]


def test_random_effects_reports_heterogeneity_for_separated_effects():
    result = _dersimonian_laird(
        np.log([0.98, 0.98, 1.04]),
        np.array([0.005, 0.006, 0.007]),
    )

    assert result["tau_squared"] > 0
    assert result["i_squared"] > 0
    assert result["ci_95_composite_ratio"][0] < result["composite_ratio"]
    assert result["ci_95_composite_ratio"][1] > result["composite_ratio"]
