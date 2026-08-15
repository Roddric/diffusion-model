"""Training-only nearest-neighbor innovations conditional on log-volatility state."""

import numpy as np
import pandas as pd


class StateConditionalInnovationModel:
    """Resample full cross-sectional innovations near a forecast volatility state.

    The factor-state reconstruction already supplies asset-level conditional
    scales.  This model conditions the remaining standardized innovation shape
    and dependence on the forecast common log-volatility coordinates, using only
    aligned training observations.
    """

    def __init__(self):
        self.volatility_states_ = None
        self.innovations_ = None
        self.state_scale_ = None
        self.stocks = None
        self.n_mean_factors_ = None

    def fit(self, states, innovations, n_mean_factors):
        if not isinstance(states, pd.DataFrame) or not isinstance(
            innovations, pd.DataFrame
        ):
            raise TypeError("states and innovations must be pandas DataFrames.")
        if n_mean_factors < 0 or n_mean_factors >= states.shape[1]:
            raise ValueError("n_mean_factors leaves no volatility-state columns.")
        common = states.index.intersection(innovations.dropna(how="any").index)
        if len(common) < 3:
            raise ValueError("At least three aligned complete training rows are required.")
        self.n_mean_factors_ = int(n_mean_factors)
        self.stocks = list(innovations.columns)
        self.volatility_states_ = states.loc[common].iloc[
            :, self.n_mean_factors_:
        ].to_numpy(dtype=float)
        raw_innovations = innovations.loc[common].to_numpy(dtype=float)
        self.innovations_ = raw_innovations - raw_innovations.mean(axis=0)
        self.state_scale_ = np.std(self.volatility_states_, axis=0, ddof=0)
        self.state_scale_[self.state_scale_ < 1e-8] = 1.0
        return self

    def _check_fitted(self):
        if self.volatility_states_ is None:
            raise RuntimeError("fit must be called before sampling.")

    def sample(self, state_paths, neighbors, seed=0):
        """Sample locally centered innovation rows for each path and horizon."""
        self._check_fitted()
        state_paths = np.asarray(state_paths, dtype=float)
        if state_paths.ndim != 3:
            raise ValueError("state_paths must have shape (paths, horizon, state_dim).")
        if state_paths.shape[2] != self.n_mean_factors_ + self.volatility_states_.shape[1]:
            raise ValueError("state_paths dimension differs from the fitted state schema.")
        if neighbors < 2 or neighbors > len(self.innovations_):
            raise ValueError("neighbors must be between 2 and the training row count.")

        forecast_volatility = state_paths[:, :, self.n_mean_factors_:]
        flat = forecast_volatility.reshape(-1, forecast_volatility.shape[-1])
        scaled_difference = (
            flat[:, None, :] - self.volatility_states_[None, :, :]
        ) / self.state_scale_[None, None, :]
        squared_distance = np.sum(scaled_difference**2, axis=2)
        nearest = np.argpartition(
            squared_distance, kth=neighbors - 1, axis=1
        )[:, :neighbors]

        rng = np.random.default_rng(seed)
        choices = rng.integers(0, neighbors, size=len(flat))
        selected = self.innovations_[nearest[np.arange(len(flat)), choices]]
        local_center = self.innovations_[nearest].mean(axis=1)
        sampled = selected - local_center
        return sampled.reshape(
            state_paths.shape[0], state_paths.shape[1], len(self.stocks)
        )

    def sample_unconditional(self, n_paths, horizon, seed=0):
        """Resample globally centered full innovation rows without conditioning."""
        self._check_fitted()
        if n_paths < 1 or horizon < 1:
            raise ValueError("n_paths and horizon must be positive.")
        rng = np.random.default_rng(seed)
        indices = rng.integers(
            0, len(self.innovations_), size=(n_paths, horizon)
        )
        return self.innovations_[indices]
