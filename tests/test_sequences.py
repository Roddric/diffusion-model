"""Leak-free Phase 2A sequence protocol and path baselines."""

import numpy as np

from data.loader import SyntheticDataPipeline
from sequences.baselines import Phase2ABaselines
from sequences.dataset import FactorStateSequenceBuilder
from sequences.evaluation import PathForecastEvaluator, _energy_score


def _synthetic(config):
    config.data.source = "synthetic"
    config.data.synthetic_n_stocks = 8
    config.data.synthetic_n_days = 700
    config.data.random_seed = 17
    config.factors.n_vol_factors = 3
    config.factors.vol_window = 10
    config.residuals.min_obs = 30
    return SyntheticDataPipeline(config).load_all_data()


def _builder(config):
    return FactorStateSequenceBuilder(
        config,
        context_length=20,
        horizon=5,
        train_fraction=0.65,
        validation_fraction=0.15,
        evaluation_stride=5,
    )


def test_sequence_windows_are_split_contained_and_scaled_on_train(config):
    returns, market = _synthetic(config)
    splits = _builder(config).build(returns, market)

    assert splits.train.context.shape[1:] == (20, 8)
    assert splits.train.target.shape[1:] == (5, 8)
    assert len(splits.validation) > 0
    assert len(splits.test) > 0

    train_end = splits.train_returns.index[-1].to_datetime64()
    validation_start = splits.validation_returns.index[0].to_datetime64()
    validation_end = splits.validation_returns.index[-1].to_datetime64()
    test_start = splits.test_returns.index[0].to_datetime64()

    assert splits.train.target_dates.max() <= train_end
    assert splits.validation.context_dates.min() >= validation_start
    assert splits.validation.target_dates.max() <= validation_end
    assert splits.test.context_dates.min() >= test_start
    np.testing.assert_allclose(
        splits.train_states.mean().values, 0.0, atol=1e-6
    )
    np.testing.assert_allclose(
        splits.train_states.std(ddof=0).values, 1.0, atol=1e-6
    )


def test_future_changes_do_not_refit_training_transformations(config):
    returns, market = _synthetic(config)
    builder = _builder(config)
    original = builder.build(returns, market)

    tampered = returns.copy()
    tampered.iloc[-60:] *= 20
    after = builder.build(tampered, market)

    np.testing.assert_allclose(
        original.train_states.values, after.train_states.values
    )
    np.testing.assert_allclose(
        original.latent_metadata["loc"], after.latent_metadata["loc"]
    )
    np.testing.assert_allclose(
        original.extractor.vol_pca_.components_,
        after.extractor.vol_pca_.components_,
    )


def test_temporal_baselines_return_finite_path_ensembles(config):
    returns, market = _synthetic(config)
    splits = _builder(config).build(returns, market)
    baselines = Phase2ABaselines(config, splits)
    forecast = baselines.forecast(
        splits.test.context[0], horizon=5, n_paths=3, seed=4
    )
    evaluator = PathForecastEvaluator(
        splits.latent_metadata["n_mean_factors"]
    )
    target_dates = splits.test.target_dates[0]
    target_returns = (
        splits.test_returns.reindex(target_dates)[baselines.stocks].values
    )

    for method in baselines.METHODS:
        assert forecast.states[method].shape == (3, 5, 8)
        assert forecast.returns[method].shape == (3, 5, 8)
        assert np.isfinite(forecast.returns[method]).all()
        metrics = evaluator.score(
            forecast.states[method],
            forecast.returns[method],
            splits.test.target[0],
            target_returns,
        )
        assert all(np.isfinite(value) for value in metrics.values())
        assert metrics["portfolio_path_energy_score"] >= 0
        assert metrics["portfolio_var_05_pinball"] >= 0
        assert 0 <= metrics["portfolio_var_05_coverage_error"] <= 0.95


def test_large_ensemble_energy_matches_direct_empirical_formula():
    rng = np.random.default_rng(19)
    paths = rng.normal(size=(70, 5, 3))
    target = rng.normal(size=(5, 3))
    samples = paths.reshape(len(paths), -1)
    observed = target.reshape(-1)
    scale = np.sqrt(len(observed))
    expected = (
        np.linalg.norm(samples - observed, axis=1).mean()
        - 0.5
        * np.linalg.norm(
            samples[:, None, :] - samples[None, :, :], axis=2
        ).mean()
    ) / scale
    np.testing.assert_allclose(_energy_score(paths, target), expected)
