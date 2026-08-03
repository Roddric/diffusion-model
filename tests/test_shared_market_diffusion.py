"""Shared cross-market residual-path diffusion."""

import numpy as np

from diffusion.shared_market import SharedMarketResidualDiffusion


def _market_data(seed):
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(80, 4))
    contexts = np.stack([states[index:index + 6] for index in range(60)])
    targets = np.stack(
        [states[index + 6:index + 9] for index in range(60)]
    )
    return {
        "train_states": states[:65],
        "train_context": contexts[:45],
        "train_target": targets[:45],
        "validation_context": contexts[45:],
        "validation_target": targets[45:],
    }


def test_shared_market_diffusion_trains_and_samples(config):
    config.temporal.hidden_dim = 8
    config.temporal.n_layers = 1
    config.temporal.dropout = 0.0
    config.temporal.batch_size = 8
    config.temporal.validation_interval = 1
    config.temporal.early_stopping_patience = 2
    config.temporal.sampling_steps = 2
    config.temporal.path_validation_paths = 2
    config.temporal.path_validation_steps = 2
    config.temporal.residual_scale_grid = [1.0]
    data = {"a": _market_data(1), "b": _market_data(2)}
    model = SharedMarketResidualDiffusion(
        config,
        market_names=["a", "b"],
        state_dim=4,
        horizon=3,
        n_mean_factors=2,
    )
    history = model.fit(data, seed=4, training_steps=2)
    first = model.sample(
        "a", data["a"]["validation_context"][0], n_paths=3, seed=5
    )
    second = model.sample(
        "a", data["a"]["validation_context"][0], n_paths=3, seed=5
    )

    assert history
    assert first.shape == (3, 3, 4)
    np.testing.assert_allclose(first, second)
    assert model.network.context_dim == 6
    assert model.market_window_counts_["a"]["train"] == 45


def test_shared_market_diffusion_rejects_unknown_market(config):
    model = SharedMarketResidualDiffusion(
        config, ["a", "b"], state_dim=4, horizon=3
    )
    with np.testing.assert_raises(KeyError):
        model._augment_context(np.zeros((6, 4)), "c")


def test_shared_market_diffusion_builds_one_adapter_per_market(config):
    model = SharedMarketResidualDiffusion(
        config,
        ["sp500", "csi300"],
        state_dim=4,
        horizon=3,
        adapter_rank=2,
    )

    assert model.adapter_rank == 2
    assert model.network.n_markets == 2
    assert len(model.network.market_adapters) == 2
