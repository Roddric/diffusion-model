"""Leak-free mean-factor and log-volatility-factor extraction."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class FactorExtractor:
    """Fit factor mappings on training data and reuse them on later windows."""

    def __init__(self, config):
        self.config = config
        self.n_vol_factors = config.factors.n_vol_factors
        self.vol_window = config.factors.vol_window
        self.vol_floor = config.factors.vol_floor
        self.vol_lower_quantile = config.factors.vol_lower_quantile
        self.vol_upper_quantile = config.factors.vol_upper_quantile

        self.mean_loadings_ = None
        self.mean_intercepts_ = None
        self.vol_pca_ = None
        self.vol_assets_ = None
        self.vol_bounds_ = None

    @staticmethod
    def _mean_factor_frame(returns, market_returns):
        factors = {
            "market": market_returns.reindex(returns.index),
            # 12-month momentum, skipping the latest month.
            "momentum": returns.rolling(231).mean().shift(21).mean(axis=1),
            "reversal": -returns.rolling(21).mean().mean(axis=1),
            "dispersion": np.log(returns.std(axis=1) + 1e-8),
            "illiquidity": np.log(returns.abs().mean(axis=1) + 1e-8),
        }
        return pd.DataFrame(factors).dropna()

    @staticmethod
    def _combine_history(current, history):
        if history is None or history.empty:
            return current
        combined = pd.concat([history, current])
        return combined.loc[~combined.index.duplicated(keep="last")].sort_index()

    def fit_mean_factors(self, returns, market_returns):
        """Fit cross-asset factor loadings using only the supplied window."""
        factors = self._mean_factor_frame(returns, market_returns)
        common_idx = returns.index.intersection(factors.index)
        aligned_returns = returns.loc[common_idx]
        factors = factors.loc[common_idx]

        loadings = {}
        intercepts = {}
        names = list(factors.columns)
        for stock in aligned_returns.columns:
            y = aligned_returns[stock].dropna()
            common = y.index.intersection(factors.dropna().index)
            if len(common) < 50:
                intercepts[stock] = 0.0
                loadings[stock] = np.zeros(len(names))
                continue

            x = factors.loc[common].values
            design = np.column_stack([np.ones(len(x)), x])
            try:
                beta = np.linalg.lstsq(design, y.loc[common].values, rcond=None)[0]
                intercepts[stock] = beta[0]
                loadings[stock] = beta[1:]
            except np.linalg.LinAlgError:
                intercepts[stock] = 0.0
                loadings[stock] = np.zeros(len(names))

        self.mean_loadings_ = pd.DataFrame(loadings, index=names).T
        self.mean_intercepts_ = pd.Series(intercepts, name="intercept")
        return {
            "factors": factors,
            "loadings": self.mean_loadings_.copy(),
            "intercepts": self.mean_intercepts_.copy(),
        }

    def transform_mean_factors(
        self, returns, market_returns, history_returns=None, history_market=None
    ):
        """Construct later factor states while retaining train-fitted loadings."""
        if self.mean_loadings_ is None:
            raise RuntimeError("fit_mean_factors must be called before transform.")

        all_returns = self._combine_history(returns, history_returns)
        all_market = self._combine_history(
            market_returns.to_frame("market"),
            None if history_market is None else history_market.to_frame("market"),
        )["market"]
        factors = self._mean_factor_frame(all_returns, all_market)
        dates = returns.index.intersection(factors.index)
        return {
            "factors": factors.loc[dates],
            "loadings": self.mean_loadings_.copy(),
            "intercepts": self.mean_intercepts_.copy(),
        }

    def extract_mean_factors(self, returns, market_returns):
        """Backward-compatible alias for fitting and transforming one window."""
        return self.fit_mean_factors(returns, market_returns)

    def compute_residuals(self, returns, mean_factors):
        factors = mean_factors["factors"]
        loadings = mean_factors["loadings"]
        intercepts = mean_factors.get(
            "intercepts", pd.Series(0.0, index=loadings.index)
        )
        common_idx = returns.index.intersection(factors.index)
        stocks = [stock for stock in returns.columns if stock in loadings.index]

        fitted = (
            factors.loc[common_idx].values @ loadings.loc[stocks].values.T
            + intercepts.reindex(stocks).fillna(0.0).values
        )
        return returns.loc[common_idx, stocks] - pd.DataFrame(
            fitted, index=common_idx, columns=stocks
        )

    def _log_variance_features(self, residuals, history_residuals=None):
        """Causal rolling log realized variance for each residual series."""
        all_residuals = self._combine_history(residuals, history_residuals)
        log_variance = np.log(
            all_residuals.pow(2)
            .rolling(self.vol_window, min_periods=self.vol_window)
            .mean()
            + self.vol_floor
        )
        dates = residuals.index.intersection(log_variance.dropna().index)
        return log_variance.loc[dates]

    def fit_volatility_factors(self, residuals):
        """Fit PCA to rolling log residual variance, never to signed returns."""
        valid = residuals.dropna(axis=1, how="all")
        features = self._log_variance_features(valid).dropna(axis=1, how="any")
        if features.empty:
            raise ValueError(
                f"Need at least {self.vol_window} residual observations to fit "
                "log-volatility factors."
            )

        n_components = min(
            self.n_vol_factors, features.shape[0], features.shape[1]
        )
        if n_components < 1:
            raise ValueError("No usable residual series for volatility PCA.")

        self.vol_pca_ = PCA(n_components=n_components)
        scores = self.vol_pca_.fit_transform(features)
        self.vol_assets_ = list(features.columns)
        self.vol_bounds_ = pd.DataFrame(
            {
                "lower": features.quantile(self.vol_lower_quantile),
                "upper": features.quantile(self.vol_upper_quantile),
            }
        )
        names = [f"logvol_pc{i + 1}" for i in range(n_components)]
        explained = self.vol_pca_.explained_variance_ratio_.cumsum()
        print(
            f"Log-volatility PCA explained variance: {explained[-1]:.2%} "
            f"with {n_components} factors"
        )
        return self._vol_result(scores, features.index, names, explained)

    def transform_volatility_factors(self, residuals, history_residuals=None):
        """Transform later residual volatility using the train-fitted PCA."""
        if self.vol_pca_ is None:
            raise RuntimeError(
                "fit_volatility_factors must be called before transform."
            )
        aligned = residuals.reindex(columns=self.vol_assets_)
        history = (
            None
            if history_residuals is None
            else history_residuals.reindex(columns=self.vol_assets_)
        )
        features = self._log_variance_features(aligned, history).dropna(how="any")
        scores = self.vol_pca_.transform(features)
        names = [f"logvol_pc{i + 1}" for i in range(scores.shape[1])]
        explained = self.vol_pca_.explained_variance_ratio_.cumsum()
        return self._vol_result(scores, features.index, names, explained)

    def _vol_result(self, scores, index, names, explained):
        return {
            "factors": pd.DataFrame(scores, index=index, columns=names),
            "loadings": pd.DataFrame(
                self.vol_pca_.components_.T,
                index=self.vol_assets_,
                columns=names,
            ),
            "log_variance_mean": pd.Series(
                self.vol_pca_.mean_, index=self.vol_assets_
            ),
            "log_variance_bounds": self.vol_bounds_.copy(),
            "explained_variance": explained,
            "window": self.vol_window,
        }

    def extract_volatility_factors(self, residuals):
        """Backward-compatible alias for fitting one training window."""
        return self.fit_volatility_factors(residuals)

    @staticmethod
    def reconstruct_log_variance(vol_factors):
        scores = vol_factors["factors"]
        loadings = vol_factors["loadings"]
        center = vol_factors["log_variance_mean"].reindex(loadings.index)
        values = scores.values @ loadings.values.T + center.values
        reconstructed = pd.DataFrame(
            values, index=scores.index, columns=loadings.index
        )
        bounds = vol_factors.get("log_variance_bounds")
        if bounds is not None:
            reconstructed = reconstructed.clip(
                lower=bounds["lower"], upper=bounds["upper"], axis=1
            )
        return reconstructed

    def compute_idiosyncratic(self, residuals, vol_factors):
        """Remove the common volatility surface and return standardized innovations."""
        log_variance = self.reconstruct_log_variance(vol_factors)
        dates = residuals.index.intersection(log_variance.index)
        stocks = residuals.columns.intersection(log_variance.columns)
        scale = np.exp(0.5 * log_variance.loc[dates, stocks]).clip(lower=1e-6)
        innovations = residuals.loc[dates, stocks] / scale
        print(
            "Standardized residual innovations: "
            f"mean std={innovations.std().mean():.3f}"
        )
        return innovations
