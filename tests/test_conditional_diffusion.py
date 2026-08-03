"""Conditional temporal path diffusion."""

import numpy as np
import torch

from diffusion.conditional_path import (
    ConditionalPathDiffusion,
    ConditionalTemporalDenoiser,
    VARInnovationPathDiffusion,
    VARResidualPathDiffusion,
)


def test_temporal_denoiser_preserves_path_shape_and_uses_context():
    torch.manual_seed(3)
    network = ConditionalTemporalDenoiser(
        state_dim=4,
        max_horizon=5,
        hidden_dim=16,
        n_layers=2,
        dropout=0.0,
    ).eval()
    noisy = torch.randn(3, 5, 4)
    time = torch.full((3,), 0.5)
    zeros = torch.zeros(3, 10, 4)
    ones = torch.ones(3, 10, 4)

    first = network(noisy, time, zeros)
    second = network(noisy, time, ones)

    assert first.shape == noisy.shape
    assert not torch.allclose(first, second)


def test_temporal_denoiser_accepts_market_augmented_context():
    torch.manual_seed(13)
    network = ConditionalTemporalDenoiser(
        state_dim=4,
        context_dim=6,
        max_horizon=3,
        hidden_dim=12,
        n_layers=1,
        dropout=0.0,
    ).eval()
    noisy = torch.randn(2, 3, 4)
    time = torch.full((2,), 0.5)
    state_context = torch.randn(2, 8, 4)
    first_market = torch.cat(
        [
            state_context,
            torch.tensor([1.0, 0.0]).view(1, 1, 2).expand(2, 8, 2),
        ],
        dim=-1,
    )
    second_market = torch.cat(
        [
            state_context,
            torch.tensor([0.0, 1.0]).view(1, 1, 2).expand(2, 8, 2),
        ],
        dim=-1,
    )

    first = network(noisy, time, first_market)
    second = network(noisy, time, second_market)

    assert first.shape == noisy.shape
    assert not torch.allclose(first, second)


def test_market_adapters_start_as_zero_residual_branches():
    network = ConditionalTemporalDenoiser(
        state_dim=4,
        context_dim=6,
        n_markets=2,
        market_adapter_rank=3,
        max_horizon=3,
        hidden_dim=12,
        n_layers=1,
        dropout=0.0,
    )

    assert len(network.market_adapters) == 2
    for adapter in network.market_adapters:
        torch.testing.assert_close(
            adapter[-1].weight, torch.zeros_like(adapter[-1].weight)
        )

    network.reset_parameters()
    for adapter in network.market_adapters:
        torch.testing.assert_close(
            adapter[-1].weight, torch.zeros_like(adapter[-1].weight)
        )


def test_temporal_denoiser_uses_separate_factor_family_heads():
    network = ConditionalTemporalDenoiser(
        state_dim=6,
        max_horizon=4,
        hidden_dim=16,
        n_layers=1,
        dropout=0.0,
        n_mean_factors=2,
        split_output_heads=True,
    )
    noisy = torch.randn(3, 4, 6)
    output = network(
        noisy,
        torch.full((3,), 0.4),
        torch.randn(3, 8, 6),
    )

    assert network.split_output_heads
    assert network.mean_output_projection.out_features == 2
    assert network.volatility_output_projection.out_features == 4
    assert output.shape == noisy.shape


def test_volatility_aware_loss_applies_configured_weight(config):
    config.temporal.mean_loss_weight = 1.0
    config.temporal.volatility_loss_weight = 3.0
    model = ConditionalPathDiffusion(
        config,
        state_dim=4,
        horizon=2,
        n_mean_factors=2,
    )
    prediction = torch.zeros(1, 2, 4)
    noise = torch.zeros_like(prediction)
    noise[..., :2] = 1.0
    noise[..., 2:] = 2.0

    total, mean, volatility = model._loss_components(prediction, noise)

    assert mean.item() == 1.0
    assert volatility.item() == 4.0
    assert total.item() == 3.25


def test_conditional_diffusion_trains_and_samples_reproducibly(config):
    rng = np.random.default_rng(8)
    config.temporal.hidden_dim = 16
    config.temporal.n_layers = 2
    config.temporal.dropout = 0.0
    config.temporal.batch_size = 8
    config.temporal.validation_interval = 2
    config.temporal.early_stopping_patience = 5
    config.temporal.sampling_steps = 4
    context = rng.normal(size=(40, 10, 4)).astype(np.float32)
    target = np.stack(
        [
            0.7 * row[-1] + rng.normal(0, 0.2, (5, 4))
            for row in context
        ]
    ).astype(np.float32)

    model = ConditionalPathDiffusion(
        config,
        state_dim=4,
        horizon=5,
        n_mean_factors=2,
    )
    history = model.fit(
        context[:32],
        target[:32],
        context[32:],
        target[32:],
        seed=2,
        training_steps=4,
    )
    first = model.sample(context[0], n_paths=3, seed=11)
    second = model.sample(context[0], n_paths=3, seed=11)

    assert history
    assert model.best_step_ is not None
    assert model.best_validation_components_["volatility"] >= 0
    assert first.shape == (3, 5, 4)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first, second)


def test_fit_seed_controls_network_initialization(config):
    rng = np.random.default_rng(19)
    config.temporal.hidden_dim = 8
    config.temporal.n_layers = 1
    config.temporal.dropout = 0.0
    config.temporal.batch_size = 4
    config.temporal.validation_interval = 1
    config.temporal.early_stopping_patience = 3
    context = rng.normal(size=(12, 6, 4)).astype(np.float32)
    target = rng.normal(size=(12, 3, 4)).astype(np.float32)
    first = ConditionalPathDiffusion(
        config, state_dim=4, horizon=3, n_mean_factors=2
    )
    second = ConditionalPathDiffusion(
        config, state_dim=4, horizon=3, n_mean_factors=2
    )

    first.fit(
        context[:8], target[:8], context[8:], target[8:],
        seed=23, training_steps=2,
    )
    second.fit(
        context[:8], target[:8], context[8:], target[8:],
        seed=23, training_steps=2,
    )

    for first_value, second_value in zip(
        first.network.state_dict().values(),
        second.network.state_dict().values(),
    ):
        torch.testing.assert_close(first_value, second_value)


def test_batched_context_sampling_and_path_selection(config):
    rng = np.random.default_rng(29)
    config.temporal.hidden_dim = 8
    config.temporal.n_layers = 1
    config.temporal.dropout = 0.0
    config.temporal.batch_size = 4
    config.temporal.validation_interval = 1
    config.temporal.early_stopping_patience = 3
    config.temporal.sampling_steps = 3
    config.temporal.path_validation_paths = 3
    config.temporal.path_validation_steps = 2
    context = rng.normal(size=(12, 6, 4)).astype(np.float32)
    target = rng.normal(size=(12, 3, 4)).astype(np.float32)
    model = ConditionalPathDiffusion(
        config, state_dim=4, horizon=3, n_mean_factors=2
    )

    history = model.fit(
        context[:8],
        target[:8],
        context[8:],
        target[8:],
        seed=31,
        training_steps=2,
        selection_metric="sampled_path_energy",
    )
    paths = model.sample_contexts(
        context[8:10], n_paths=3, seed=32, sampling_steps=2
    )

    assert paths.shape == (2, 3, 3, 4)
    assert np.isfinite(paths).all()
    assert model.selection_metric_ == "sampled_path_energy"
    assert model.best_selection_score_ is not None
    assert history[0]["validation_path_energy"] is not None
    assert history[0]["selection_score"] == (
        history[0]["validation_path_energy"]
    )


def test_var_residual_diffusion_anchors_paths_to_linear_forecast(config):
    rng = np.random.default_rng(12)
    config.temporal.hidden_dim = 16
    config.temporal.n_layers = 1
    config.temporal.dropout = 0.0
    config.temporal.batch_size = 8
    config.temporal.validation_interval = 2
    config.temporal.early_stopping_patience = 5
    config.temporal.sampling_steps = 3
    states = rng.normal(size=(80, 3)).astype(np.float32)
    context = np.stack([states[i:i + 8] for i in range(50)])
    target = np.stack([states[i + 8:i + 12] for i in range(50)])

    model = VARResidualPathDiffusion(config, state_dim=3, horizon=4)
    model.fit(
        states[:65],
        context[:40],
        target[:40],
        context[40:],
        target[40:],
        seed=5,
        training_steps=4,
    )
    paths = model.sample(context[0], n_paths=2, seed=6)

    assert paths.shape == (2, 4, 3)
    assert np.isfinite(paths).all()
    assert model.residual_scale_.shape == (4, 3)


def test_residual_scale_calibration_is_validation_only_and_applied(config):
    rng = np.random.default_rng(81)
    config.temporal.hidden_dim = 8
    config.temporal.n_layers = 1
    config.temporal.dropout = 0.0
    config.temporal.batch_size = 4
    config.temporal.validation_interval = 1
    config.temporal.early_stopping_patience = 2
    config.temporal.sampling_steps = 2
    config.temporal.path_validation_paths = 2
    config.temporal.path_validation_steps = 2
    config.temporal.residual_scale_grid = [0.0, 1.0]
    states = rng.normal(size=(60, 4)).astype(np.float32)
    context = np.stack([states[i:i + 6] for i in range(40)])
    target = np.stack([states[i + 6:i + 9] for i in range(40)])

    model = VARResidualPathDiffusion(
        config, state_dim=4, horizon=3, n_mean_factors=2
    )
    model.fit(
        states[:50],
        context[:30],
        target[:30],
        context[30:],
        target[30:],
        seed=13,
        training_steps=2,
    )
    assert model.calibration_["criterion"] == (
        "validation_state_energy_score"
    )
    assert set(model.residual_multiplier_) <= {0.0, 1.0}
    assert len(model.calibration_["candidates"]) == 4


def test_var_innovation_diffusion_integrates_one_step_shocks(config):
    rng = np.random.default_rng(101)
    config.temporal.hidden_dim = 8
    config.temporal.n_layers = 1
    config.temporal.dropout = 0.0
    config.temporal.batch_size = 4
    config.temporal.validation_interval = 1
    config.temporal.early_stopping_patience = 2
    config.temporal.sampling_steps = 2
    config.temporal.path_validation_paths = 2
    config.temporal.path_validation_steps = 2
    states = rng.normal(size=(70, 4)).astype(np.float32)
    context = np.stack([states[i:i + 6] for i in range(50)])
    target = np.stack([states[i + 6:i + 9] for i in range(50)])

    model = VARInnovationPathDiffusion(
        config, state_dim=4, horizon=3, n_mean_factors=2
    )
    history = model.fit(
        states[:60],
        context[:40],
        target[:40],
        context[40:],
        target[40:],
        seed=17,
        training_steps=2,
    )
    paths = model.sample(context[0], n_paths=3, seed=18)

    assert history
    assert paths.shape == (3, 3, 4)
    assert model.innovation_location_.shape == (4,)
    assert model.innovation_scale_.shape == (4,)
    assert np.isfinite(paths).all()
