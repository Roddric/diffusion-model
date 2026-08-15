"""Mean-state diagnostic tests."""

import numpy as np

from research.mean_state_diagnostic import _aggregate, _univariate_path_diagnostics


def test_univariate_diagnostic_is_exact_for_oracle_paths():
    target = np.arange(12, dtype=float).reshape(4, 3)
    paths = np.repeat(target[None, :, :], 5, axis=0)

    scores = _univariate_path_diagnostics(paths, target)

    np.testing.assert_allclose(scores["squared_error"], 0.0)
    np.testing.assert_allclose(scores["crps"], 0.0)
    np.testing.assert_allclose(scores["interval_90_covered"], 1.0)


def test_aggregate_mse_shares_sum_to_one():
    target = np.zeros((20, 2))
    paths = np.ones((4, 20, 2))
    row = _univariate_path_diagnostics(paths, target)
    summary = _aggregate({"method": [row]}, ["a", "b"])["method"]

    shares = [factor["mse_share"] for factor in summary["factors"].values()]
    assert sum(shares) == 1.0
