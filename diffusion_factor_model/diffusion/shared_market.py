"""Shared residual-path diffusion with market-specific linear dynamics."""

import numpy as np
import torch

from .conditional_path import ConditionalPathDiffusion

if __package__ and __package__.startswith("diffusion_factor_model."):
    from ..dynamics.var import LatentVAR
else:
    from dynamics.var import LatentVAR


class SharedMarketResidualDiffusion:
    """Share nonlinear residual learning while retaining one VAR per market."""

    REQUIRED_KEYS = (
        "train_states",
        "train_context",
        "train_target",
        "validation_context",
        "validation_target",
    )

    def __init__(
        self,
        config,
        market_names,
        state_dim,
        horizon=None,
        n_mean_factors=None,
        adapter_rank=None,
    ):
        names = list(market_names)
        if len(names) < 2 or len(set(names)) != len(names):
            raise ValueError("At least two unique market names are required.")
        self.config = config
        self.market_names = names
        self.market_index = {
            name: index for index, name in enumerate(names)
        }
        self.state_dim = int(state_dim)
        self.horizon = horizon or config.temporal.horizon
        self.diffusion = ConditionalPathDiffusion(
            config,
            state_dim=state_dim,
            context_dim=state_dim + len(names),
            horizon=self.horizon,
            n_mean_factors=n_mean_factors,
            n_markets=(len(names) if adapter_rank is not None else None),
            market_adapter_rank=adapter_rank,
        )
        self.adapter_rank = adapter_rank
        self.vars = {}
        self.residual_locations = {}
        self.residual_scales = {}
        self.market_window_counts_ = None

    @property
    def network(self):
        return self.diffusion.network

    @property
    def best_step_(self):
        return self.diffusion.best_step_

    @property
    def best_validation_loss_(self):
        return self.diffusion.best_validation_loss_

    @property
    def best_selection_score_(self):
        return self.diffusion.best_selection_score_

    @property
    def history_(self):
        return self.diffusion.history_

    def _check_market(self, market):
        if market not in self.market_index:
            raise KeyError(f"Unknown market {market!r}.")

    def _augment_context(self, context, market):
        self._check_market(market)
        context = np.asarray(context, dtype=float)
        if context.ndim not in (2, 3) or context.shape[-1] != self.state_dim:
            raise ValueError("Context has the wrong state dimension.")
        token = np.zeros(context.shape[:-1] + (len(self.market_names),))
        token[..., self.market_index[market]] = 1.0
        return np.concatenate([context, token], axis=-1)

    def _var_mean_paths(self, contexts, market):
        return np.stack(
            [
                self.vars[market].forecast_mean(
                    self.horizon, initial_state=context[-1]
                )
                for context in contexts
            ]
        )

    @staticmethod
    def _even_indices(length, count):
        if count == length:
            return np.arange(length)
        return np.linspace(0, length - 1, count, dtype=int)

    def _state_path_transform(self):
        device = self.diffusion.device
        state_dim = self.state_dim
        parameters = {}
        for market in self.market_names:
            parameters[market] = {
                "intercept": torch.as_tensor(
                    self.vars[market].intercept_,
                    dtype=torch.float32,
                    device=device,
                ),
                "transition": torch.as_tensor(
                    self.vars[market].transition_,
                    dtype=torch.float32,
                    device=device,
                ),
                "location": torch.as_tensor(
                    self.residual_locations[market],
                    dtype=torch.float32,
                    device=device,
                ),
                "scale": torch.as_tensor(
                    self.residual_scales[market],
                    dtype=torch.float32,
                    device=device,
                ),
            }

        def transform(samples, targets, contexts):
            sample_states = torch.empty_like(samples)
            target_states = torch.empty_like(targets)
            market_tokens = contexts[:, 0, state_dim:]
            market_ids = torch.argmax(market_tokens, dim=1)
            for market, market_id in self.market_index.items():
                mask = market_ids == market_id
                if not torch.any(mask):
                    continue
                values = parameters[market]
                previous = contexts[mask, -1, :state_dim]
                means = []
                for _ in range(self.horizon):
                    previous = (
                        values["intercept"]
                        + previous @ values["transition"].T
                    )
                    means.append(previous)
                mean_paths = torch.stack(means, dim=1)
                sample_residual = (
                    samples[mask]
                    * values["scale"][None, None, :, :]
                    + values["location"][None, None, :, :]
                )
                target_residual = (
                    targets[mask] * values["scale"][None, :, :]
                    + values["location"][None, :, :]
                )
                sample_states[mask] = (
                    mean_paths[:, None, :, :] + sample_residual
                )
                target_states[mask] = mean_paths + target_residual
            return sample_states, target_states

        return transform

    def fit(
        self,
        market_data,
        seed=0,
        training_steps=None,
        selection_metric="sampled_path_energy",
    ):
        if set(market_data) != set(self.market_names):
            raise ValueError("market_data keys must match market_names exactly.")
        train_contexts = []
        train_targets = []
        validation_parts = []
        counts = {}
        for market in self.market_names:
            data = market_data[market]
            missing = [key for key in self.REQUIRED_KEYS if key not in data]
            if missing:
                raise ValueError(
                    f"{market} data are missing required keys: {missing}"
                )
            states = np.asarray(data["train_states"], dtype=float)
            train_context = np.asarray(data["train_context"], dtype=float)
            train_target = np.asarray(data["train_target"], dtype=float)
            validation_context = np.asarray(
                data["validation_context"], dtype=float
            )
            validation_target = np.asarray(
                data["validation_target"], dtype=float
            )
            self.vars[market] = LatentVAR().fit(states)
            train_residual = (
                train_target
                - self._var_mean_paths(train_context, market)
            )
            validation_residual = (
                validation_target
                - self._var_mean_paths(validation_context, market)
            )
            location = train_residual.mean(axis=0)
            scale = train_residual.std(axis=0)
            scale[scale < 1e-6] = 1.0
            self.residual_locations[market] = location
            self.residual_scales[market] = scale
            train_contexts.append(
                self._augment_context(train_context, market)
            )
            train_targets.append((train_residual - location) / scale)
            validation_parts.append(
                (
                    self._augment_context(validation_context, market),
                    (validation_residual - location) / scale,
                )
            )
            counts[market] = {
                "train": len(train_context),
                "validation": len(validation_context),
            }

        balanced_validation = min(
            len(context) for context, _ in validation_parts
        )
        validation_contexts = []
        validation_targets = []
        for context, target in validation_parts:
            indices = self._even_indices(len(context), balanced_validation)
            validation_contexts.append(context[indices])
            validation_targets.append(target[indices])
        self.market_window_counts_ = counts
        return self.diffusion.fit(
            np.concatenate(train_contexts),
            np.concatenate(train_targets),
            np.concatenate(validation_contexts),
            np.concatenate(validation_targets),
            seed=seed,
            training_steps=training_steps,
            selection_metric=selection_metric,
            validation_path_transform=self._state_path_transform(),
        )

    def sample(
        self,
        market,
        context,
        n_paths=20,
        seed=0,
        sampling_steps=None,
    ):
        if market not in self.vars:
            raise RuntimeError("fit must be called before sample.")
        context = np.asarray(context, dtype=float)
        normalized = self.diffusion.sample(
            self._augment_context(context, market),
            n_paths=n_paths,
            seed=seed,
            sampling_steps=sampling_steps,
        )
        residual = (
            normalized * self.residual_scales[market]
            + self.residual_locations[market]
        )
        mean_path = self.vars[market].forecast_mean(
            self.horizon, initial_state=context[-1]
        )
        return mean_path[None, :, :] + residual
