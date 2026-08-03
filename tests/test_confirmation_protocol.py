"""Immutable confirmation statistics and multiple-testing helpers."""

import numpy as np

from research.confirm_phase2f import _bh_adjust
from research.robustness import (
    circular_block_bootstrap_mean,
    dependence_robust_comparison,
    hac_mean_test,
)


def test_benjamini_hochberg_adjustment_is_monotone_and_bounded():
    adjusted = _bh_adjust({"a": 0.01, "b": 0.04, "c": 0.20})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]
    assert adjusted["a"] == 0.03
    assert all(0.0 <= value <= 1.0 for value in adjusted.values())


def test_circular_block_bootstrap_is_reproducible_and_reports_loss_scale():
    difference = np.array([-0.4, -0.1, 0.2, -0.3, 0.1])
    first = circular_block_bootstrap_mean(
        difference, block_length=2, seed=9, draws=500
    )
    second = circular_block_bootstrap_mean(
        difference, block_length=2, seed=9, draws=500
    )
    assert first == second
    assert first["mean_difference"] == np.mean(difference)
    assert first["ci_95"][0] <= first["ci_95"][1]


def test_hac_and_dependence_summary_preserve_candidate_minus_baseline_sign():
    baseline = np.ones(12)
    candidate = baseline - np.linspace(0.01, 0.12, 12)
    summary = dependence_robust_comparison(
        candidate, baseline, block_lengths=(2, 3), draws=500
    )
    assert summary["n_origins"] == 12
    assert all(value < 0 for value in summary["origin_loss_differences"])
    assert summary["hac"]["mean_difference"] < 0
    assert set(summary["circular_block_bootstrap"]) == {"2", "3"}

    hac = hac_mean_test(np.asarray(summary["origin_loss_differences"]), max_lag=2)
    assert hac["newey_west_standard_error"] >= 0
    assert 0 <= hac["one_sided_normal_p_candidate_not_better"] <= 1
