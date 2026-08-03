"""Exploratory state baseline invariants."""

import numpy as np

from research.expanded_baseline_audit import (
    DiagonalARGaussian,
    GradientBoostedStateAR,
    RidgeVARGaussian,
)


def _states(seed=4):
    rng = np.random.default_rng(seed)
    values = [np.zeros(3)]
    transition = np.diag([0.8, 0.4, -0.2])
    for _ in range(159):
        values.append(transition @ values[-1] + rng.normal(scale=0.2, size=3))
    return np.asarray(values)


def test_linear_exploratory_baselines_are_stable_and_reproducible():
    states = _states()
    for model in (DiagonalARGaussian().fit(states), RidgeVARGaussian().fit(states)):
        first = model.simulate(8, seed=3, initial_state=states[-1])
        second = model.simulate(8, seed=3, initial_state=states[-1])
        assert first.shape == (8, 3)
        assert np.isfinite(first).all()
        np.testing.assert_allclose(first, second)
        assert max(abs(np.linalg.eigvals(model.transition_))) <= 0.980001


def test_gradient_boosted_state_baseline_is_reproducible():
    states = _states()
    model = GradientBoostedStateAR().fit(states)
    first = model.simulate(6, seed=7, initial_state=states[-1])
    second = model.simulate(6, seed=7, initial_state=states[-1])
    assert first.shape == (6, 3)
    np.testing.assert_allclose(first, second)
