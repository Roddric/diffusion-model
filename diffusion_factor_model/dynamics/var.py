"""Leak-free VAR/GARCH baseline for the factor-state representation."""

import numpy as np
from sklearn.covariance import LedoitWolf
from scipy.stats import t as student_t

if __package__ and __package__.startswith("diffusion_factor_model."):
    from ..factors.extractor import FactorExtractor
    from ..latent.parametrizer import LatentParametrizer
    from ..reconstruction.reconstructor import ReturnReconstructor
    from ..residuals.garch import GARCHModeler
else:
    from factors.extractor import FactorExtractor
    from latent.parametrizer import LatentParametrizer
    from reconstruction.reconstructor import ReturnReconstructor
    from residuals.garch import GARCHModeler


class LatentVAR:
    """Stable VAR(1) fitted to standardized latent factor states."""

    def __init__(self, max_spectral_radius=0.98):
        self.max_spectral_radius = max_spectral_radius
        self.intercept_ = None
        self.transition_ = None
        self.innovation_cov_ = None
        self.innovations_ = None
        self.student_df_ = None
        self.last_state_ = None

    def fit(self, states):
        states = np.asarray(states, dtype=float)
        if states.ndim != 2 or len(states) < 3:
            raise ValueError("LatentVAR requires at least three 2D state observations.")

        design = np.column_stack([np.ones(len(states) - 1), states[:-1]])
        coef = np.linalg.lstsq(design, states[1:], rcond=None)[0]
        intercept = coef[0]
        transition = coef[1:].T

        radius = max(abs(np.linalg.eigvals(transition)))
        if radius > self.max_spectral_radius:
            transition *= self.max_spectral_radius / radius
            # Preserve the fitted unconditional mean after stabilizing A.
            mean = states.mean(axis=0)
            intercept = mean - transition @ mean

        fitted = intercept + states[:-1] @ transition.T
        innovations = states[1:] - fitted
        if len(innovations) > innovations.shape[1]:
            covariance = LedoitWolf().fit(innovations).covariance_
        else:
            covariance = np.cov(innovations, rowvar=False)
        covariance = np.atleast_2d(covariance) + 1e-8 * np.eye(states.shape[1])

        self.intercept_ = intercept
        self.transition_ = transition
        self.innovation_cov_ = covariance
        self.innovations_ = innovations
        whitened = self._whiten(innovations, covariance).ravel()
        try:
            fitted_df, _, _ = student_t.fit(whitened, floc=0.0)
            self.student_df_ = float(np.clip(fitted_df, 3.0, 30.0))
        except Exception:
            self.student_df_ = 8.0
        self.last_state_ = states[-1].copy()
        return self

    @staticmethod
    def _whiten(values, covariance):
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        inverse_root = (
            eigenvectors
            @ np.diag(1.0 / np.sqrt(np.maximum(eigenvalues, 1e-10)))
            @ eigenvectors.T
        )
        return values @ inverse_root.T

    @property
    def spectral_radius_(self):
        if self.transition_ is None:
            raise RuntimeError("fit must be called first.")
        return float(max(abs(np.linalg.eigvals(self.transition_))))

    def simulate(self, n_steps, seed=0, initial_state=None, burn_in=0):
        if self.transition_ is None:
            raise RuntimeError("fit must be called before simulate.")
        if n_steps < 1:
            raise ValueError("n_steps must be positive.")

        rng = np.random.default_rng(seed)
        state = (
            self.last_state_.copy()
            if initial_state is None
            else np.asarray(initial_state, dtype=float).copy()
        )
        draws = []
        for step in range(n_steps + burn_in):
            innovation = rng.multivariate_normal(
                np.zeros(len(state)), self.innovation_cov_
            )
            state = self.intercept_ + self.transition_ @ state + innovation
            if step >= burn_in:
                draws.append(state.copy())
        return np.asarray(draws)

    def simulate_student_t(
        self, n_steps, seed=0, initial_state=None, burn_in=0
    ):
        """Simulate VAR paths with covariance-matched multivariate-t shocks."""
        if self.transition_ is None:
            raise RuntimeError("fit must be called before simulate_student_t.")
        if n_steps < 1:
            raise ValueError("n_steps must be positive.")

        rng = np.random.default_rng(seed)
        state = (
            self.last_state_.copy()
            if initial_state is None
            else np.asarray(initial_state, dtype=float).copy()
        )
        df = self.student_df_
        gaussian_cov = self.innovation_cov_ * (df - 2.0) / df
        draws = []
        for step in range(n_steps + burn_in):
            gaussian = rng.multivariate_normal(
                np.zeros(len(state)), gaussian_cov
            )
            innovation = gaussian / np.sqrt(rng.chisquare(df) / df)
            state = self.intercept_ + self.transition_ @ state + innovation
            if step >= burn_in:
                draws.append(state.copy())
        return np.asarray(draws)

    def simulate_residual_bootstrap(
        self,
        n_steps,
        seed=0,
        initial_state=None,
        block_length=5,
    ):
        """Simulate VAR paths from moving blocks of fitted innovation vectors."""
        if self.transition_ is None:
            raise RuntimeError(
                "fit must be called before simulate_residual_bootstrap."
            )
        if n_steps < 1 or block_length < 1:
            raise ValueError("n_steps and block_length must be positive.")

        rng = np.random.default_rng(seed)
        state = (
            self.last_state_.copy()
            if initial_state is None
            else np.asarray(initial_state, dtype=float).copy()
        )
        innovations = []
        while len(innovations) < n_steps:
            width = min(block_length, n_steps - len(innovations))
            max_start = len(self.innovations_) - width
            start = int(rng.integers(0, max_start + 1))
            innovations.extend(self.innovations_[start:start + width])

        path = []
        for innovation in innovations:
            state = self.intercept_ + self.transition_ @ state + innovation
            path.append(state.copy())
        return np.asarray(path)

    def forecast_mean(self, n_steps, initial_state=None):
        """Deterministic conditional mean path with innovations fixed at zero."""
        if self.transition_ is None:
            raise RuntimeError("fit must be called before forecast_mean.")
        state = (
            self.last_state_.copy()
            if initial_state is None
            else np.asarray(initial_state, dtype=float).copy()
        )
        path = []
        for _ in range(n_steps):
            state = self.intercept_ + self.transition_ @ state
            path.append(state.copy())
        return np.asarray(path)


class RegimeStudentTLatentVAR:
    """Stable VAR with regime-specific covariance and Student-t tails.

    The transition remains common across regimes. Only the innovation
    distribution changes, limiting parameter growth while allowing volatility
    states available at the forecast origin to control forecast dispersion.
    """

    def __init__(
        self,
        n_regimes=3,
        min_regime_obs=30,
        max_spectral_radius=0.98,
    ):
        if n_regimes < 2:
            raise ValueError("n_regimes must be at least two.")
        if min_regime_obs < 3:
            raise ValueError("min_regime_obs must be at least three.")
        self.n_regimes = int(n_regimes)
        self.min_regime_obs = int(min_regime_obs)
        self.var = LatentVAR(max_spectral_radius=max_spectral_radius)
        self.regime_weights_ = None
        self.regime_intercept_ = 0.0
        self.thresholds_ = None
        self.regime_covariances_ = None
        self.regime_dfs_ = None
        self.regime_counts_ = None

    @staticmethod
    def _fit_df(innovations, covariance):
        whitened = LatentVAR._whiten(innovations, covariance).ravel()
        try:
            fitted_df, _, _ = student_t.fit(whitened, floc=0.0)
            return float(np.clip(fitted_df, 3.0, 30.0))
        except Exception:
            return 8.0

    def fit(
        self,
        states,
        regime_signal,
        regime_weights,
        regime_intercept=0.0,
    ):
        states = np.asarray(states, dtype=float)
        signal = np.asarray(regime_signal, dtype=float)
        weights = np.asarray(regime_weights, dtype=float)
        if states.ndim != 2 or len(states) < 3:
            raise ValueError("states must be a two-dimensional history.")
        if signal.shape != (len(states),):
            raise ValueError("regime_signal must align one-for-one with states.")
        if weights.shape != (states.shape[1],):
            raise ValueError("regime_weights must match the state dimension.")
        if not np.isfinite(signal).all() or not np.isfinite(weights).all():
            raise ValueError("Regime inputs must be finite.")

        self.var.fit(states)
        self.regime_weights_ = weights.copy()
        self.regime_intercept_ = float(regime_intercept)
        probabilities = np.arange(1, self.n_regimes) / self.n_regimes
        self.thresholds_ = np.quantile(signal, probabilities)
        origin_regimes = np.digitize(signal[:-1], self.thresholds_)
        covariances = []
        dfs = []
        counts = []
        global_innovations = self.var.innovations_
        for regime in range(self.n_regimes):
            selected = global_innovations[origin_regimes == regime]
            counts.append(int(len(selected)))
            if len(selected) < max(self.min_regime_obs, states.shape[1] + 1):
                covariance = self.var.innovation_cov_.copy()
                fit_values = global_innovations
            else:
                covariance = LedoitWolf().fit(selected).covariance_
                covariance += 1e-8 * np.eye(states.shape[1])
                fit_values = selected
            covariances.append(covariance)
            dfs.append(self._fit_df(fit_values, covariance))
        self.regime_covariances_ = np.asarray(covariances)
        self.regime_dfs_ = np.asarray(dfs)
        self.regime_counts_ = np.asarray(counts)
        return self

    @property
    def spectral_radius_(self):
        return self.var.spectral_radius_

    def classify(self, state):
        if self.thresholds_ is None:
            raise RuntimeError("fit must be called before classify.")
        state = np.asarray(state, dtype=float)
        signal = float(state @ self.regime_weights_ + self.regime_intercept_)
        return int(np.digitize(signal, self.thresholds_))

    def simulate(self, n_steps, seed=0, initial_state=None):
        if self.thresholds_ is None:
            raise RuntimeError("fit must be called before simulate.")
        if n_steps < 1:
            raise ValueError("n_steps must be positive.")
        rng = np.random.default_rng(seed)
        state = (
            self.var.last_state_.copy()
            if initial_state is None
            else np.asarray(initial_state, dtype=float).copy()
        )
        draws = []
        for _ in range(n_steps):
            regime = self.classify(state)
            df = self.regime_dfs_[regime]
            covariance = self.regime_covariances_[regime]
            gaussian_cov = covariance * (df - 2.0) / df
            gaussian = rng.multivariate_normal(
                np.zeros(len(state)), gaussian_cov
            )
            innovation = gaussian / np.sqrt(rng.chisquare(df) / df)
            state = (
                self.var.intercept_
                + self.var.transition_ @ state
                + innovation
            )
            draws.append(state.copy())
        return np.asarray(draws)


class DynamicFactorVARGARCH:
    """Factor-state VAR plus GARCH-filtered standardized residual innovations."""

    def __init__(self, config):
        self.config = config
        self.extractor = FactorExtractor(config)
        self.parametrizer = LatentParametrizer(config)
        self.var = LatentVAR()
        self.garch = GARCHModeler(config)
        self.reconstructor = None
        self.latent_metadata = None
        self.latent_ = None

    def fit(self, returns, market_returns):
        mean_results = self.extractor.fit_mean_factors(returns, market_returns)
        residuals = self.extractor.compute_residuals(returns, mean_results)
        vol_results = self.extractor.fit_volatility_factors(residuals)
        latent, metadata = self.parametrizer.fit_transform(
            mean_results, vol_results
        )
        innovations = self.extractor.compute_idiosyncratic(
            residuals, vol_results
        )
        self.garch.fit_all(innovations)
        self.var.fit(latent)
        self.latent_ = latent
        self.latent_metadata = metadata
        self.reconstructor = ReturnReconstructor(
            self.config,
            mean_results,
            vol_results,
            parametrizer=self.parametrizer,
            latent_metadata=metadata,
        )
        return self

    def sample(self, n_samples, seed=0):
        if self.reconstructor is None:
            raise RuntimeError("fit must be called before sample.")
        latent = self.var.simulate(n_samples, seed=seed)
        innovations = self.garch.sample_all(n_samples, seed=seed)
        return self.reconstructor.reconstruct(latent, innovations)
