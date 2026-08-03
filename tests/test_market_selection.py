"""Statistical helpers for the prespecified market-selection protocol."""

import numpy as np

from research.market_selection import (
    _aggregate_seed_scores,
    _bootstrap_difference,
)


def test_paired_bootstrap_detects_consistent_improvement():
    result = _bootstrap_difference(
        diffusion=[0.5, 0.6, 0.7, 0.8],
        baseline=[1.0, 1.0, 1.0, 1.0],
        draws=2000,
    )
    assert result["mean_difference"] < 0
    assert result["ci_95"][1] < 0
    assert result["origin_win_rate"] == 1.0


def test_seed_scores_are_averaged_by_origin():
    seed_scores = [
        [{"loss": 1.0}, {"loss": 3.0}],
        [{"loss": 3.0}, {"loss": 5.0}],
    ]
    aggregated = _aggregate_seed_scores(seed_scores)
    np.testing.assert_allclose(aggregated["loss"], [2.0, 4.0])

