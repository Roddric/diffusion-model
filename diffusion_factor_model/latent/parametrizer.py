"""Leak-free factor-state standardization."""

import warnings

import numpy as np
import pandas as pd


class LatentParametrizer:
    """Fit latent scaling once on training states, then reuse it."""

    def __init__(self, config):
        self.config = config
        self.standardize = getattr(config.latent, "standardize", True)
        self.metadata_ = None
        if getattr(config.latent, "use_log_variance", False):
            warnings.warn(
                "latent.use_log_variance is deprecated: volatility PCA already "
                "operates in log-variance space.",
                FutureWarning,
                stacklevel=2,
            )

    @staticmethod
    def _join(mean_factors, vol_factors):
        mean_df = mean_factors["factors"]
        vol_df = vol_factors["factors"]
        dates = mean_df.index.intersection(vol_df.index)
        mean_df = mean_df.loc[dates]
        vol_df = vol_df.loc[dates]
        return np.hstack([mean_df.values, vol_df.values]), mean_df, vol_df, dates

    def fit(self, mean_factors, vol_factors):
        latent, mean_df, vol_df, dates = self._join(mean_factors, vol_factors)
        if self.standardize:
            loc = latent.mean(axis=0)
            scale = latent.std(axis=0)
            scale[scale < 1e-12] = 1.0
        else:
            loc = np.zeros(latent.shape[1])
            scale = np.ones(latent.shape[1])

        self.metadata_ = {
            "n_mean_factors": mean_df.shape[1],
            "n_vol_factors": vol_df.shape[1],
            "mean_names": list(mean_df.columns),
            "vol_names": list(vol_df.columns),
            "dates": dates,
            "loc": loc,
            "scale": scale,
        }
        return self

    def transform(self, mean_factors, vol_factors):
        if self.metadata_ is None:
            raise RuntimeError("fit must be called before transform.")
        latent, mean_df, vol_df, dates = self._join(mean_factors, vol_factors)
        expected = self.metadata_["mean_names"] + self.metadata_["vol_names"]
        actual = list(mean_df.columns) + list(vol_df.columns)
        if actual != expected:
            raise ValueError("Factor columns differ from the fitted latent schema.")

        transformed = (latent - self.metadata_["loc"]) / self.metadata_["scale"]
        metadata = dict(self.metadata_)
        metadata["dates"] = dates
        return transformed, metadata

    def fit_transform(self, mean_factors, vol_factors):
        self.fit(mean_factors, vol_factors)
        latent, metadata = self.transform(mean_factors, vol_factors)
        print(
            f"Latent space: {latent.shape} "
            f"(mean: {metadata['n_mean_factors']}, "
            f"vol: {metadata['n_vol_factors']})"
        )
        print(
            "  train scale before standardization: "
            f"{metadata['scale'].round(5).tolist()}"
        )
        return latent, metadata

    def parametrize(self, mean_factors, vol_factors):
        """Backward-compatible alias for fit_transform."""
        return self.fit_transform(mean_factors, vol_factors)

    @staticmethod
    def inverse_transform(latent, metadata):
        latent = latent * metadata["scale"] + metadata["loc"]
        n_mean = metadata["n_mean_factors"]
        n_vol = metadata["n_vol_factors"]
        return latent[:, :n_mean], latent[:, n_mean:n_mean + n_vol]

    def deparametrize(self, latent, metadata):
        mean_values, vol_values = self.inverse_transform(latent, metadata)
        mean_df = pd.DataFrame(
            mean_values,
            index=metadata["dates"],
            columns=metadata["mean_names"],
        )
        vol_df = pd.DataFrame(
            vol_values,
            index=metadata["dates"],
            columns=metadata["vol_names"],
        )
        return mean_df, vol_df
