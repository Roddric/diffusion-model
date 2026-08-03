"""Conditional diffusion over complete future factor-state paths."""

import copy

import numpy as np
import torch
import torch.nn as nn

from .score_net import SinusoidalPositionalEmbedding

if __package__ and __package__.startswith("diffusion_factor_model."):
    from ..dynamics.var import LatentVAR
else:
    from dynamics.var import LatentVAR


def _group_count(hidden_dim):
    for groups in (8, 4, 2, 1):
        if hidden_dim % groups == 0:
            return groups
    return 1


class TemporalResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        groups = _group_count(hidden_dim)
        self.norm1 = nn.GroupNorm(groups, hidden_dim)
        self.norm2 = nn.GroupNorm(groups, hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.condition = nn.Linear(hidden_dim, hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, condition):
        scale, shift = self.condition(condition).chunk(2, dim=-1)
        value = self.norm1(hidden)
        value = value * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        value = self.conv1(torch.nn.functional.silu(value))
        value = self.conv2(torch.nn.functional.silu(self.norm2(value)))
        return hidden + self.dropout(value)


class ConditionalTemporalDenoiser(nn.Module):
    """Predict future-path diffusion noise conditioned on observed state history."""

    def __init__(
        self,
        state_dim,
        max_horizon,
        hidden_dim=64,
        n_layers=4,
        dropout=0.05,
        n_mean_factors=None,
        split_output_heads=False,
        context_dim=None,
        n_markets=None,
        market_adapter_rank=None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.context_dim = context_dim or state_dim
        self.n_markets = int(n_markets or 0)
        self.market_adapter_rank = market_adapter_rank
        if self.n_markets and (
            market_adapter_rank is None or market_adapter_rank < 1
        ):
            raise ValueError(
                "market_adapter_rank must be positive when n_markets is set."
            )
        if self.n_markets and self.context_dim < state_dim + self.n_markets:
            raise ValueError(
                "Market-conditioned context lacks the requested market tokens."
            )
        self.max_horizon = max_horizon
        self.n_mean_factors = n_mean_factors
        self.split_output_heads = bool(
            split_output_heads
            and n_mean_factors is not None
            and 0 < n_mean_factors < state_dim
        )
        self.context_encoder = nn.GRU(
            self.context_dim, hidden_dim, batch_first=True
        )
        self.input_projection = nn.Linear(state_dim, hidden_dim)
        self.position = nn.Parameter(
            torch.randn(max_horizon, hidden_dim) * 0.02
        )
        self.time_embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_projection = nn.Linear(hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                TemporalResidualBlock(hidden_dim, dropout)
                for _ in range(n_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        if self.split_output_heads:
            self.mean_output_norm = nn.LayerNorm(hidden_dim)
            self.volatility_output_norm = nn.LayerNorm(hidden_dim)
            self.mean_output_projection = nn.Linear(
                hidden_dim, n_mean_factors
            )
            self.volatility_output_projection = nn.Linear(
                hidden_dim, state_dim - n_mean_factors
            )
        else:
            self.output_projection = nn.Linear(hidden_dim, state_dim)
        if self.n_markets:
            self.market_adapters = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(
                            hidden_dim,
                            market_adapter_rank,
                            bias=False,
                        ),
                        nn.SiLU(),
                        nn.Linear(
                            market_adapter_rank,
                            state_dim,
                            bias=False,
                        ),
                    )
                    for _ in range(self.n_markets)
                ]
            )
            self._zero_market_adapter_outputs()

    def _zero_market_adapter_outputs(self):
        if not self.n_markets:
            return
        for adapter in self.market_adapters:
            nn.init.zeros_(adapter[-1].weight)

    def reset_parameters(self):
        """Reset every learned layer so the fit seed defines initialization."""
        for module in self.modules():
            if module is self:
                continue
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
        nn.init.normal_(self.position, mean=0.0, std=0.02)
        self._zero_market_adapter_outputs()

    def forward(self, noisy_future, diffusion_time, context):
        if noisy_future.ndim != 3 or context.ndim != 3:
            raise ValueError("noisy_future and context must both be 3D tensors.")
        horizon = noisy_future.shape[1]
        if horizon > self.max_horizon:
            raise ValueError("Future path exceeds configured maximum horizon.")

        _, context_hidden = self.context_encoder(context)
        condition = self.context_projection(context_hidden[-1])
        condition = condition + self.time_projection(
            self.time_embedding(diffusion_time)
        )

        hidden = self.input_projection(noisy_future)
        hidden = hidden + self.position[:horizon].unsqueeze(0)
        hidden = hidden.transpose(1, 2)
        for block in self.blocks:
            hidden = block(hidden, condition)
        hidden = hidden.transpose(1, 2)
        if self.split_output_heads:
            mean = self.mean_output_projection(
                self.mean_output_norm(hidden)
            )
            volatility = self.volatility_output_projection(
                self.volatility_output_norm(hidden)
            )
            output = torch.cat([mean, volatility], dim=-1)
        else:
            output = self.output_projection(self.output_norm(hidden))
        if self.n_markets:
            tokens = context[:, 0, -self.n_markets:]
            adapter_values = torch.stack(
                [adapter(hidden) for adapter in self.market_adapters],
                dim=2,
            )
            output = output + torch.einsum(
                "bm,bhms->bhs", tokens, adapter_values
            )
        return output


class ConditionalPathDiffusion:
    """Train, select, and sample a context-conditioned path denoiser."""

    def __init__(
        self,
        config,
        state_dim,
        horizon=None,
        n_mean_factors=None,
        context_dim=None,
        n_markets=None,
        market_adapter_rank=None,
    ):
        temporal = config.temporal
        self.config = config
        self.horizon = horizon or temporal.horizon
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.network = ConditionalTemporalDenoiser(
            state_dim=state_dim,
            max_horizon=self.horizon,
            hidden_dim=temporal.hidden_dim,
            n_layers=temporal.n_layers,
            dropout=temporal.dropout,
            n_mean_factors=n_mean_factors,
            split_output_heads=temporal.split_output_heads,
            context_dim=context_dim,
            n_markets=n_markets,
            market_adapter_rank=market_adapter_rank,
        ).to(self.device)
        self.n_mean_factors = n_mean_factors
        self.beta_min = config.diffusion.beta_min
        self.beta_max = config.diffusion.beta_max
        self.T = config.diffusion.T
        self.history_ = []
        self.best_step_ = None
        self.best_validation_loss_ = None
        self.best_validation_components_ = None
        self.best_selection_score_ = None
        self.selection_metric_ = None

    def _loss_components(self, prediction, noise):
        squared_error = (prediction - noise) ** 2
        if (
            self.n_mean_factors is None
            or self.n_mean_factors <= 0
            or self.n_mean_factors >= squared_error.shape[-1]
        ):
            total = torch.mean(squared_error)
            return total, total, total

        temporal = self.config.temporal
        mean_loss = torch.mean(
            squared_error[..., : self.n_mean_factors]
        )
        volatility_loss = torch.mean(
            squared_error[..., self.n_mean_factors :]
        )
        mean_weight = temporal.mean_loss_weight
        volatility_weight = temporal.volatility_loss_weight
        if mean_weight <= 0 or volatility_weight <= 0:
            raise ValueError("Temporal loss weights must be positive.")
        total = (
            mean_weight * mean_loss
            + volatility_weight * volatility_loss
        ) / (mean_weight + volatility_weight)
        return total, mean_loss, volatility_loss

    def _alpha_sigma(self, time):
        log_alpha = (
            -0.25 * time ** 2 * (self.beta_max - self.beta_min)
            - 0.5 * time * self.beta_min
        )
        alpha = torch.exp(log_alpha).view(-1, 1, 1)
        sigma = torch.sqrt(
            torch.clamp(1.0 - alpha ** 2, min=1e-8)
        )
        return alpha, sigma

    def fit(
        self,
        train_context,
        train_target,
        validation_context,
        validation_target,
        seed=0,
        training_steps=None,
        selection_metric="denoising_loss",
        validation_path_transform=None,
    ):
        temporal = self.config.temporal
        steps = training_steps or temporal.training_steps
        supported_selection = {
            "denoising_loss",
            "sampled_path_energy",
        }
        if selection_metric not in supported_selection:
            raise ValueError(
                "selection_metric must be one of "
                f"{sorted(supported_selection)}."
            )
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.network.reset_parameters()
        self.selection_metric_ = selection_metric

        train_context = torch.as_tensor(
            train_context, dtype=torch.float32, device=self.device
        )
        train_target = torch.as_tensor(
            train_target, dtype=torch.float32, device=self.device
        )
        validation_context = torch.as_tensor(
            validation_context, dtype=torch.float32, device=self.device
        )
        validation_target = torch.as_tensor(
            validation_target, dtype=torch.float32, device=self.device
        )
        if len(train_context) == 0 or len(validation_context) == 0:
            raise ValueError("Training and validation windows must be non-empty.")

        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=temporal.learning_rate,
            weight_decay=temporal.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps
        )
        generator = torch.Generator(device=self.device).manual_seed(seed)
        validation_generator = torch.Generator(
            device=self.device
        ).manual_seed(seed + 10000)
        validation_time = (
            1e-3
            + torch.rand(
                len(validation_context),
                generator=validation_generator,
                device=self.device,
            )
            * (self.T - 1e-3)
        )
        validation_noise = torch.randn(
            validation_target.shape,
            generator=validation_generator,
            device=self.device,
        )

        best_state = copy.deepcopy(self.network.state_dict())
        best_score = float("inf")
        stale_checks = 0
        batch_size = min(temporal.batch_size, len(train_context))
        self.history_ = []

        for step in range(1, steps + 1):
            self.network.train()
            indices = torch.randint(
                0,
                len(train_context),
                (batch_size,),
                generator=generator,
                device=self.device,
            )
            context = train_context[indices]
            target = train_target[indices]
            time = (
                1e-3
                + torch.rand(
                    batch_size, generator=generator, device=self.device
                )
                * (self.T - 1e-3)
            )
            noise = torch.randn(
                target.shape, generator=generator, device=self.device
            )
            alpha, sigma = self._alpha_sigma(time)
            prediction = self.network(
                alpha * target + sigma * noise, time, context
            )
            loss, train_mean_loss, train_volatility_loss = (
                self._loss_components(prediction, noise)
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), max_norm=1.0
            )
            optimizer.step()
            scheduler.step()

            should_validate = (
                step == 1
                or step % temporal.validation_interval == 0
                or step == steps
            )
            if not should_validate:
                continue
            validation = self._validation_loss(
                validation_context,
                validation_target,
                validation_time,
                validation_noise,
            )
            validation_loss = validation["total"]
            path_validation = None
            if selection_metric == "sampled_path_energy":
                path_validation = self._validation_path_metrics(
                    validation_context,
                    validation_target,
                    seed=seed + 20000,
                    n_paths=temporal.path_validation_paths,
                    sampling_steps=temporal.path_validation_steps,
                    path_transform=validation_path_transform,
                )
                selection_score = path_validation["energy"]
            else:
                selection_score = validation_loss
            self.history_.append(
                {
                    "step": step,
                    "train_loss": float(loss.item()),
                    "train_mean_loss": float(train_mean_loss.item()),
                    "train_volatility_loss": float(
                        train_volatility_loss.item()
                    ),
                    "validation_loss": validation_loss,
                    "validation_mean_loss": validation["mean"],
                    "validation_volatility_loss": validation["volatility"],
                    "validation_path_energy": (
                        None
                        if path_validation is None
                        else path_validation["energy"]
                    ),
                    "validation_path_rmse": (
                        None
                        if path_validation is None
                        else path_validation["rmse"]
                    ),
                    "selection_metric": selection_metric,
                    "selection_score": selection_score,
                }
            )
            if selection_score < best_score - 1e-5:
                best_score = selection_score
                best_state = copy.deepcopy(self.network.state_dict())
                self.best_step_ = step
                self.best_validation_components_ = validation
                self.best_validation_loss_ = validation_loss
                self.best_selection_score_ = selection_score
                stale_checks = 0
            else:
                stale_checks += 1
            if stale_checks >= temporal.early_stopping_patience:
                break

        self.network.load_state_dict(best_state)
        return self.history_

    def _validation_loss(self, context, target, time, noise):
        self.network.eval()
        with torch.no_grad():
            alpha, sigma = self._alpha_sigma(time)
            prediction = self.network(
                alpha * target + sigma * noise, time, context
            )
            total, mean, volatility = self._loss_components(
                prediction, noise
            )
            return {
                "total": float(total.item()),
                "mean": float(mean.item()),
                "volatility": float(volatility.item()),
            }

    def _sample_expanded_context(
        self,
        context,
        seed,
        sampling_steps,
    ):
        temporal = self.config.temporal
        steps = sampling_steps or temporal.sampling_steps
        n_samples = len(context)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        state = torch.randn(
            (n_samples, self.horizon, self.network.state_dim),
            generator=generator,
            device=self.device,
        )
        times = torch.linspace(
            self.T, 1e-3, steps + 1, device=self.device
        )
        self.network.eval()
        with torch.no_grad():
            for index in range(steps):
                current = times[index].expand(n_samples)
                following = times[index + 1].expand(n_samples)
                alpha, sigma = self._alpha_sigma(current)
                next_alpha, next_sigma = self._alpha_sigma(following)
                predicted_noise = torch.clamp(
                    self.network(state, current, context), -10.0, 10.0
                )
                clean = torch.clamp(
                    (state - sigma * predicted_noise) / alpha.clamp(min=1e-8),
                    -8.0,
                    8.0,
                )
                state = next_alpha * clean + next_sigma * predicted_noise
        return state

    def _sample_context_batch_tensor(
        self,
        contexts,
        n_paths,
        seed,
        sampling_steps,
    ):
        if n_paths < 1:
            raise ValueError("n_paths must be positive.")
        contexts = torch.as_tensor(
            contexts, dtype=torch.float32, device=self.device
        )
        if contexts.ndim != 3 or len(contexts) == 0:
            raise ValueError(
                "contexts must have shape "
                "(n_contexts, context_length, state_dim)."
            )
        n_contexts, context_length, state_dim = contexts.shape
        expanded = (
            contexts[:, None, :, :]
            .expand(n_contexts, n_paths, context_length, state_dim)
            .reshape(n_contexts * n_paths, context_length, state_dim)
        )
        samples = self._sample_expanded_context(
            expanded,
            seed=seed,
            sampling_steps=sampling_steps,
        )
        return samples.reshape(
            n_contexts,
            n_paths,
            self.horizon,
            self.network.state_dim,
        )

    def _validation_path_metrics(
        self,
        context,
        target,
        seed,
        n_paths,
        sampling_steps,
        path_transform=None,
    ):
        if n_paths < 2:
            raise ValueError(
                "At least two validation paths are required for energy score."
            )
        samples = self._sample_context_batch_tensor(
            context,
            n_paths=n_paths,
            seed=seed,
            sampling_steps=sampling_steps,
        )
        target = torch.as_tensor(
            target, dtype=torch.float32, device=self.device
        )
        if path_transform is not None:
            samples, target = path_transform(samples, target, context)
        flattened_samples = samples.flatten(start_dim=2)
        flattened_target = target.flatten(start_dim=1)
        scale = float(np.sqrt(flattened_target.shape[1]))
        first = torch.linalg.vector_norm(
            flattened_samples - flattened_target[:, None, :],
            dim=2,
        ).mean(dim=1) / scale
        pairwise = torch.cdist(
            flattened_samples, flattened_samples
        ).mean(dim=(1, 2)) / scale
        energy = (first - 0.5 * pairwise).mean()
        rmse = torch.sqrt(
            torch.mean((samples.mean(dim=1) - target) ** 2)
        )
        return {
            "energy": float(energy.item()),
            "rmse": float(rmse.item()),
        }

    def sample_contexts(
        self,
        contexts,
        n_paths=20,
        seed=0,
        sampling_steps=None,
    ):
        return self._sample_context_batch_tensor(
            contexts,
            n_paths=n_paths,
            seed=seed,
            sampling_steps=sampling_steps,
        ).cpu().numpy()

    def sample(self, context, n_paths=20, seed=0, sampling_steps=None):
        context = torch.as_tensor(
            context, dtype=torch.float32, device=self.device
        )
        if context.ndim == 2:
            context = context.unsqueeze(0).repeat(n_paths, 1, 1)
        elif context.ndim == 3 and len(context) == 1:
            context = context.repeat(n_paths, 1, 1)
        elif context.ndim != 3 or len(context) != n_paths:
            raise ValueError("Context batch must be one row or match n_paths.")
        return self._sample_expanded_context(
            context,
            seed=seed,
            sampling_steps=sampling_steps,
        ).cpu().numpy()


class VARResidualPathDiffusion:
    """Diffuse normalized nonlinear residual paths around a fitted VAR mean."""

    def __init__(
        self,
        config,
        state_dim,
        horizon=None,
        n_mean_factors=None,
    ):
        self.config = config
        self.horizon = horizon or config.temporal.horizon
        self.var = LatentVAR()
        self.diffusion = ConditionalPathDiffusion(
            config,
            state_dim=state_dim,
            horizon=self.horizon,
            n_mean_factors=n_mean_factors,
        )
        self.residual_location_ = None
        self.residual_scale_ = None
        self.residual_multiplier_ = None
        self.calibration_ = None

    @property
    def network(self):
        return self.diffusion.network

    @property
    def history_(self):
        return self.diffusion.history_

    @property
    def best_step_(self):
        return self.diffusion.best_step_

    @property
    def best_validation_loss_(self):
        return self.diffusion.best_validation_loss_

    @property
    def best_validation_components_(self):
        return self.diffusion.best_validation_components_

    @property
    def best_selection_score_(self):
        return self.diffusion.best_selection_score_

    @property
    def selection_metric_(self):
        return self.diffusion.selection_metric_

    def _var_mean_paths(self, contexts):
        return np.stack(
            [
                self.var.forecast_mean(
                    self.horizon, initial_state=context[-1]
                )
                for context in contexts
            ]
        )

    def fit(
        self,
        train_states,
        train_context,
        train_target,
        validation_context,
        validation_target,
        seed=0,
        training_steps=None,
        selection_metric="denoising_loss",
    ):
        self.var.fit(train_states)
        train_residual = (
            train_target - self._var_mean_paths(train_context)
        )
        validation_residual = (
            validation_target - self._var_mean_paths(validation_context)
        )
        self.residual_location_ = train_residual.mean(axis=0)
        self.residual_scale_ = train_residual.std(axis=0)
        self.residual_scale_[self.residual_scale_ < 1e-6] = 1.0
        normalized_train = (
            train_residual - self.residual_location_
        ) / self.residual_scale_
        normalized_validation = (
            validation_residual - self.residual_location_
        ) / self.residual_scale_
        history = self.diffusion.fit(
            train_context,
            normalized_train,
            validation_context,
            normalized_validation,
            seed=seed,
            training_steps=training_steps,
            selection_metric=selection_metric,
        )
        self.residual_multiplier_ = np.ones(train_target.shape[-1])
        grid = self.config.temporal.residual_scale_grid
        if len(grid) > 1:
            self.calibrate_residual_scale(
                validation_context,
                validation_target,
                seed=seed + 30000,
                grid=grid,
                n_paths=self.config.temporal.path_validation_paths,
                sampling_steps=self.config.temporal.path_validation_steps,
            )
        return history

    @staticmethod
    def _mean_energy_score(samples, targets):
        scores = []
        for paths, target in zip(samples, targets):
            flattened = paths.reshape(len(paths), -1)
            observed = target.reshape(-1)
            scale = np.sqrt(len(observed))
            first = np.linalg.norm(
                flattened - observed[None, :], axis=1
            ).mean() / scale
            pairwise = np.linalg.norm(
                flattened[:, None, :] - flattened[None, :, :], axis=2
            ).mean() / scale
            scores.append(first - 0.5 * pairwise)
        return float(np.mean(scores))

    def calibrate_residual_scale(
        self,
        validation_context,
        validation_target,
        seed,
        grid,
        n_paths=8,
        sampling_steps=10,
    ):
        """Select mean/vol residual amplitudes using validation only."""
        if self.residual_location_ is None:
            raise RuntimeError("fit must be called before calibration.")
        grid = sorted({float(value) for value in grid})
        if not grid or grid[0] < 0:
            raise ValueError("Residual scale grid must contain nonnegative values.")
        normalized = self.diffusion.sample_contexts(
            validation_context,
            n_paths=n_paths,
            seed=seed,
            sampling_steps=sampling_steps,
        )
        raw_residual = (
            normalized * self.residual_scale_[None, None, :, :]
            + self.residual_location_[None, None, :, :]
        )
        var_mean = self._var_mean_paths(validation_context)
        n_mean = self.diffusion.n_mean_factors
        if n_mean is None or not 0 < n_mean < raw_residual.shape[-1]:
            families = [(raw_residual.shape[-1],)]
            candidates = [(value,) for value in grid]
        else:
            families = [(n_mean,), (raw_residual.shape[-1] - n_mean,)]
            candidates = [
                (mean_value, vol_value)
                for mean_value in grid
                for vol_value in grid
            ]

        rows = []
        best = None
        for candidate in candidates:
            multiplier = np.concatenate(
                [
                    np.full(width[0], value)
                    for width, value in zip(families, candidate)
                ]
            )
            samples = (
                var_mean[:, None, :, :]
                + raw_residual * multiplier[None, None, None, :]
            )
            score = self._mean_energy_score(samples, validation_target)
            row = {
                "mean_multiplier": candidate[0],
                "volatility_multiplier": (
                    candidate[-1] if len(candidate) > 1 else candidate[0]
                ),
                "energy_score": score,
            }
            rows.append(row)
            if best is None or score < best["energy_score"] - 1e-12:
                best = row
                self.residual_multiplier_ = multiplier
        self.calibration_ = {
            "criterion": "validation_state_energy_score",
            "seed": seed,
            "n_paths": n_paths,
            "sampling_steps": sampling_steps,
            "grid": grid,
            "selected": best,
            "candidates": rows,
        }
        return self.calibration_

    def sample(self, context, n_paths=20, seed=0, sampling_steps=None):
        if self.residual_location_ is None:
            raise RuntimeError("fit must be called before sample.")
        normalized = self.diffusion.sample(
            context,
            n_paths=n_paths,
            seed=seed,
            sampling_steps=sampling_steps,
        )
        residual = (
            normalized * self.residual_scale_
            + self.residual_location_
        )
        multiplier = (
            np.ones(residual.shape[-1])
            if self.residual_multiplier_ is None
            else self.residual_multiplier_
        )
        residual = residual * multiplier[None, None, :]
        mean_path = self.var.forecast_mean(
            self.horizon, initial_state=np.asarray(context)[-1]
        )
        return mean_path[None, :, :] + residual


class VARInnovationPathDiffusion:
    """Diffuse one-step VAR innovations, then recursively integrate states."""

    def __init__(
        self,
        config,
        state_dim,
        horizon=None,
        n_mean_factors=None,
    ):
        self.config = config
        self.horizon = horizon or config.temporal.horizon
        self.var = LatentVAR()
        self.diffusion = ConditionalPathDiffusion(
            config,
            state_dim=state_dim,
            horizon=self.horizon,
            n_mean_factors=n_mean_factors,
        )
        self.innovation_location_ = None
        self.innovation_scale_ = None

    @property
    def network(self):
        return self.diffusion.network

    @property
    def history_(self):
        return self.diffusion.history_

    @property
    def best_step_(self):
        return self.diffusion.best_step_

    @property
    def best_validation_loss_(self):
        return self.diffusion.best_validation_loss_

    @property
    def best_validation_components_(self):
        return self.diffusion.best_validation_components_

    @property
    def best_selection_score_(self):
        return self.diffusion.best_selection_score_

    @property
    def selection_metric_(self):
        return self.diffusion.selection_metric_

    def _innovation_paths(self, contexts, targets):
        contexts = np.asarray(contexts, dtype=float)
        targets = np.asarray(targets, dtype=float)
        previous = np.concatenate(
            [contexts[:, -1:, :], targets[:, :-1, :]], axis=1
        )
        conditional_mean = (
            self.var.intercept_[None, None, :]
            + previous @ self.var.transition_.T
        )
        return targets - conditional_mean

    def _state_path_transform(self):
        intercept = torch.as_tensor(
            self.var.intercept_,
            dtype=torch.float32,
            device=self.diffusion.device,
        )
        transition = torch.as_tensor(
            self.var.transition_,
            dtype=torch.float32,
            device=self.diffusion.device,
        )
        location = torch.as_tensor(
            self.innovation_location_,
            dtype=torch.float32,
            device=self.diffusion.device,
        )
        scale = torch.as_tensor(
            self.innovation_scale_,
            dtype=torch.float32,
            device=self.diffusion.device,
        )

        def integrate(innovations, initial):
            previous = initial
            values = []
            for step in range(innovations.shape[-2]):
                current = (
                    intercept
                    + previous @ transition.T
                    + innovations[..., step, :]
                )
                values.append(current)
                previous = current
            return torch.stack(values, dim=-2)

        def transform(samples, targets, contexts):
            sample_innovations = (
                samples * scale[None, None, None, :]
                + location[None, None, None, :]
            )
            target_innovations = (
                targets * scale[None, None, :]
                + location[None, None, :]
            )
            sample_initial = contexts[:, None, -1, :].expand(
                -1, samples.shape[1], -1
            )
            target_initial = contexts[:, -1, :]
            return (
                integrate(sample_innovations, sample_initial),
                integrate(target_innovations, target_initial),
            )

        return transform

    def fit(
        self,
        train_states,
        train_context,
        train_target,
        validation_context,
        validation_target,
        seed=0,
        training_steps=None,
        selection_metric="sampled_path_energy",
    ):
        self.var.fit(train_states)
        train_innovations = self._innovation_paths(
            train_context, train_target
        )
        validation_innovations = self._innovation_paths(
            validation_context, validation_target
        )
        self.innovation_location_ = train_innovations.mean(axis=(0, 1))
        self.innovation_scale_ = train_innovations.std(axis=(0, 1))
        self.innovation_scale_[self.innovation_scale_ < 1e-6] = 1.0
        normalized_train = (
            train_innovations - self.innovation_location_[None, None, :]
        ) / self.innovation_scale_[None, None, :]
        normalized_validation = (
            validation_innovations
            - self.innovation_location_[None, None, :]
        ) / self.innovation_scale_[None, None, :]
        return self.diffusion.fit(
            train_context,
            normalized_train,
            validation_context,
            normalized_validation,
            seed=seed,
            training_steps=training_steps,
            selection_metric=selection_metric,
            validation_path_transform=self._state_path_transform(),
        )

    def _integrate_numpy(self, innovations, initial_state):
        previous = np.broadcast_to(
            np.asarray(initial_state, dtype=float),
            innovations.shape[:-2] + (innovations.shape[-1],),
        ).copy()
        values = []
        for step in range(innovations.shape[-2]):
            current = (
                self.var.intercept_
                + previous @ self.var.transition_.T
                + innovations[..., step, :]
            )
            values.append(current)
            previous = current
        return np.stack(values, axis=-2)

    def sample(self, context, n_paths=20, seed=0, sampling_steps=None):
        if self.innovation_location_ is None:
            raise RuntimeError("fit must be called before sample.")
        normalized = self.diffusion.sample(
            context,
            n_paths=n_paths,
            seed=seed,
            sampling_steps=sampling_steps,
        )
        innovations = (
            normalized * self.innovation_scale_[None, None, :]
            + self.innovation_location_[None, None, :]
        )
        return self._integrate_numpy(
            innovations, np.asarray(context)[-1]
        )
