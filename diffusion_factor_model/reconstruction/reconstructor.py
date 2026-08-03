"""Reconstruct returns from mean factors, log-volatility states, and innovations."""

import numpy as np
import pandas as pd


class ReturnReconstructor:
    """Convert factor-state samples back to stock-return samples."""

    def __init__(
        self,
        config,
        mean_factor_results,
        vol_factor_results,
        parametrizer=None,
        latent_metadata=None,
    ):
        self.config = config
        self.mean_loadings = mean_factor_results["loadings"]
        self.mean_intercepts = mean_factor_results.get(
            "intercepts", pd.Series(0.0, index=self.mean_loadings.index)
        )
        self.vol_loadings = vol_factor_results["loadings"]
        self.log_variance_mean = vol_factor_results["log_variance_mean"]
        self.log_variance_bounds = vol_factor_results.get("log_variance_bounds")
        self.parametrizer = parametrizer
        self.latent_metadata = latent_metadata
        self.stocks = [
            stock
            for stock in self.mean_loadings.index
            if stock in self.vol_loadings.index
        ]

    def reconstruct(self, latent_samples, residual_samples, verbose=True):
        """Reconstruct ``r = intercept + Bf + exp(logvar/2) * epsilon``."""
        if self.parametrizer is not None and self.latent_metadata is not None:
            mean_factors, vol_factors = self.parametrizer.inverse_transform(
                latent_samples, self.latent_metadata
            )
        else:
            n_mean = len(self.mean_loadings.columns)
            mean_factors = latent_samples[:, :n_mean]
            vol_factors = latent_samples[:, n_mean:]

        beta_mean = self.mean_loadings.loc[self.stocks].values
        beta_vol = self.vol_loadings.loc[self.stocks].values
        intercept = self.mean_intercepts.reindex(self.stocks).fillna(0.0).values

        mean_returns = mean_factors @ beta_mean.T + intercept
        log_variance = (
            vol_factors @ beta_vol.T
            + self.log_variance_mean.reindex(self.stocks).values
        )
        if self.log_variance_bounds is not None:
            lower = self.log_variance_bounds.loc[self.stocks, "lower"].values
            upper = self.log_variance_bounds.loc[self.stocks, "upper"].values
            log_variance = np.clip(log_variance, lower, upper)
        conditional_scale = np.exp(0.5 * log_variance)

        returns = mean_returns
        if residual_samples is not None:
            innovations = residual_samples.reindex(columns=self.stocks).fillna(0.0)
            returns = returns + conditional_scale * innovations.values[: len(returns)]

        result = pd.DataFrame(returns, columns=self.stocks)
        if verbose:
            print(f"Reconstructed returns: {result.shape}")
        return result
