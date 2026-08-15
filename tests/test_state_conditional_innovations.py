"""State-conditional innovation sampling tests."""

import numpy as np
import pandas as pd
import pytest

from reconstruction.state_conditional_innovations import (
    StateConditionalInnovationModel,
)


def _training_panel():
    rng = np.random.default_rng(12)
    n = 200
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    regimes = np.repeat([-2.0, 2.0], n // 2)
    states = pd.DataFrame(
        {
            "mean": rng.normal(size=n),
            "vol1": regimes + rng.normal(0.0, 0.05, n),
            "vol2": regimes + rng.normal(0.0, 0.05, n),
        },
        index=index,
    )
    scales = np.where(regimes < 0, 0.2, 3.0)
    innovations = pd.DataFrame(
        rng.normal(size=(n, 3)) * scales[:, None],
        index=index,
        columns=["A", "B", "C"],
    )
    return states, innovations


def test_sampler_is_reproducible_and_has_requested_shape():
    states, innovations = _training_panel()
    model = StateConditionalInnovationModel().fit(states, innovations, 1)
    forecast = np.zeros((7, 5, 3))

    first = model.sample(forecast, neighbors=32, seed=9)
    second = model.sample(forecast, neighbors=32, seed=9)

    assert first.shape == (7, 5, 3)
    np.testing.assert_array_equal(first, second)


def test_sampler_conditions_scale_on_volatility_state():
    states, innovations = _training_panel()
    model = StateConditionalInnovationModel().fit(states, innovations, 1)
    low = np.zeros((100, 10, 3))
    low[:, :, 1:] = -2.0
    high = low.copy()
    high[:, :, 1:] = 2.0

    low_sample = model.sample(low, neighbors=64, seed=3)
    high_sample = model.sample(high, neighbors=64, seed=3)

    assert high_sample.std() > 5.0 * low_sample.std()


def test_sampler_rejects_invalid_neighbor_count():
    states, innovations = _training_panel()
    model = StateConditionalInnovationModel().fit(states, innovations, 1)

    with pytest.raises(ValueError, match="neighbors"):
        model.sample(np.zeros((2, 3, 3)), neighbors=1)
