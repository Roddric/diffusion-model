"""Dynamic factor-state VAR/GARCH baseline."""

import numpy as np

from dynamics.var import (
    DynamicFactorVARGARCH,
    LatentVAR,
    RegimeStudentTLatentVAR,
)


def test_latent_var_is_stable_and_reproducible():
    rng = np.random.default_rng(4)
    states = np.zeros((500, 3))
    transition = np.diag([0.9, 0.6, 0.2])
    for i in range(1, len(states)):
        states[i] = transition @ states[i - 1] + rng.normal(0, 0.2, 3)

    model = LatentVAR().fit(states)
    first = model.simulate(100, seed=9)
    second = model.simulate(100, seed=9)

    assert model.spectral_radius_ <= 0.980001
    np.testing.assert_allclose(first, second)
    assert np.corrcoef(first[:-1, 0], first[1:, 0])[0, 1] > 0.5


def test_heavy_tail_and_residual_bootstrap_var_paths_are_reproducible():
    rng = np.random.default_rng(91)
    states = rng.standard_t(6, size=(500, 4))
    model = LatentVAR().fit(states)

    first_t = model.simulate_student_t(12, seed=3)
    second_t = model.simulate_student_t(12, seed=3)
    first_boot = model.simulate_residual_bootstrap(12, seed=4)
    second_boot = model.simulate_residual_bootstrap(12, seed=4)

    assert 3.0 <= model.student_df_ <= 30.0
    assert model.innovations_.shape == (499, 4)
    np.testing.assert_allclose(first_t, second_t)
    np.testing.assert_allclose(first_boot, second_boot)


def test_regime_student_t_var_is_causal_stable_and_reproducible():
    rng = np.random.default_rng(23)
    states = np.zeros((900, 3))
    for index in range(1, len(states)):
        scale = 0.08 if states[index - 1, 2] < 0 else 0.30
        states[index] = (
            np.diag([0.8, 0.5, 0.9]) @ states[index - 1]
            + rng.standard_t(7, 3) * scale
        )
    weights = np.array([0.0, 0.0, 1.0])
    signal = states @ weights
    model = RegimeStudentTLatentVAR().fit(
        states, signal, weights
    )

    first = model.simulate(20, seed=8, initial_state=states[-1])
    second = model.simulate(20, seed=8, initial_state=states[-1])

    assert model.spectral_radius_ <= 0.980001
    assert model.regime_counts_.sum() == len(states) - 1
    assert len(model.regime_dfs_) == 3
    np.testing.assert_allclose(first, second)
    assert model.classify(np.array([0.0, 0.0, -10.0])) == 0
    assert model.classify(np.array([0.0, 0.0, 10.0])) == 2


def test_regime_student_t_var_validates_signal_shape():
    states = np.zeros((100, 2))
    with np.testing.assert_raises(ValueError):
        RegimeStudentTLatentVAR().fit(
            states,
            np.zeros(99),
            np.ones(2),
        )


def test_dynamic_factor_var_garch_samples_returns(config, returns, market):
    config.factors.n_vol_factors = 2
    config.residuals.min_obs = 30
    model = DynamicFactorVARGARCH(config).fit(returns, market)

    generated = model.sample(40, seed=3)

    assert generated.shape == (40, returns.shape[1])
    assert np.isfinite(generated.values).all()
    assert generated.std().mean() > 0
