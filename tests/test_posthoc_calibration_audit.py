"""Unit tests for the post-hoc calibration and power-audit diagnostics."""

import numpy as np
import pytest

from research.posthoc_calibration_power_audit import (
    _energy_terms,
    _observation_ranks,
    _rank_histogram,
)


def test_observation_ranks_span_full_range():
    target = np.zeros((1, 1))
    paths = np.array([[[-1.0]], [[1.0]]])
    ranks, ensemble_size = _observation_ranks(paths, target)
    assert ensemble_size == 2
    assert ranks.tolist() == [[1]]


def test_observation_ranks_extreme_observations():
    target = np.array([[-10.0], [10.0]])
    paths = np.stack(
        [np.array([[-1.0], [1.0]]), np.array([[-2.0], [2.0]])]
    )
    ranks, ensemble_size = _observation_ranks(paths, target)
    assert ensemble_size == 2
    assert ranks[:, 0].tolist() == [0, 2]


def test_observation_ranks_ties_use_half_count():
    target = np.zeros((1, 1))
    paths = np.array([[[0.0]], [[0.0]]])
    ranks, _ = _observation_ranks(paths, target)
    assert ranks.tolist() == [[1]]


def test_rank_histogram_calibrated_ensemble_is_uniformish():
    rng = np.random.default_rng(0)
    n_origins, horizon, dim, ensemble_size = 400, 5, 2, 10
    ranks = []
    for _ in range(n_origins):
        draw = rng.normal(size=(ensemble_size + 1, horizon, dim))
        ranks.append(
            _observation_ranks(draw[1:], draw[0])[0].reshape(-1)
        )
    histogram = _rank_histogram(np.concatenate(ranks), ensemble_size)
    assert histogram["bins"] == ensemble_size + 1
    assert histogram["n_values"] == n_origins * horizon * dim
    assert histogram["chi_square_p_uniform"] > 0.01
    assert histogram["edge_bin_share"] == pytest.approx(
        histogram["uniform_edge_bin_share"], abs=0.02
    )


def test_rank_histogram_underdispersed_ensemble_loads_edges():
    rng = np.random.default_rng(1)
    n_origins, horizon, dim, ensemble_size = 400, 5, 2, 10
    ranks = []
    for _ in range(n_origins):
        ensemble = rng.normal(
            scale=0.1, size=(ensemble_size, horizon, dim)
        )
        observation = rng.normal(scale=1.0, size=(horizon, dim))
        ranks.append(
            _observation_ranks(ensemble, observation)[0].reshape(-1)
        )
    histogram = _rank_histogram(np.concatenate(ranks), ensemble_size)
    assert histogram["chi_square_p_uniform"] < 1e-6
    assert (
        histogram["edge_bin_share"]
        > 3.0 * histogram["uniform_edge_bin_share"]
    )


def test_energy_terms_recompose_energy_score():
    from sequences.evaluation import _energy_score

    rng = np.random.default_rng(2)
    paths = rng.normal(size=(7, 4, 3))
    target = rng.normal(size=(4, 3))
    first_term, spread_term = _energy_terms(paths, target)
    assert first_term - 0.5 * spread_term == pytest.approx(
        _energy_score(paths, target)
    )
