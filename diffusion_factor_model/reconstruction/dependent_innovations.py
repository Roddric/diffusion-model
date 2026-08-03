"""Joint heavy-tailed and block-resampled idiosyncratic innovations."""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from sklearn.covariance import LedoitWolf


class DependentInnovationModel:
    """Model cross-asset dependence in standardized return innovations.

    The factor-state forecast already supplies each asset's conditional scale.
    This class models the remaining standardized innovations jointly instead of
    fitting one independent simulator per asset.
    """

    def __init__(self, min_df=4.5, max_df=30.0):
        if min_df <= 2 or max_df < min_df:
            raise ValueError("Student-t degree-of-freedom bounds are invalid.")
        self.min_df = float(min_df)
        self.max_df = float(max_df)
        self.stocks = None
        self.values_ = None
        self.center_ = None
        self.covariance_ = None
        self.covariance_root_ = None
        self.student_df_ = None

    def fit(self, innovations):
        """Fit using a complete training-only standardized innovation panel."""
        if not isinstance(innovations, pd.DataFrame):
            raise TypeError("innovations must be a pandas DataFrame.")
        complete = innovations.dropna(axis=0, how="any")
        if complete.shape[0] < 3 or complete.shape[1] < 1:
            raise ValueError("At least three complete innovation rows are required.")

        self.stocks = list(complete.columns)
        raw = complete.to_numpy(dtype=float)
        self.center_ = raw.mean(axis=0)
        self.values_ = raw - self.center_
        self.covariance_ = LedoitWolf().fit(self.values_).covariance_

        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance_)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        self.covariance_root_ = (
            eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
        )
        inverse_root = (
            eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
        )
        whitened = self.values_ @ inverse_root.T
        excess = float(
            np.nanmedian(kurtosis(whitened, axis=0, fisher=True, bias=False))
        )
        if not np.isfinite(excess) or excess <= 0:
            estimated_df = self.max_df
        else:
            estimated_df = 4.0 + 6.0 / excess
        self.student_df_ = float(
            np.clip(estimated_df, self.min_df, self.max_df)
        )
        return self

    def _check_fitted(self):
        if self.values_ is None:
            raise RuntimeError("fit must be called before sampling.")

    @staticmethod
    def _validate_shape(n_paths, horizon):
        if n_paths < 1 or horizon < 1:
            raise ValueError("n_paths and horizon must be positive.")

    def sample_student_t(self, n_paths, horizon, seed=0):
        """Draw covariance-matched multivariate-t innovation paths."""
        self._check_fitted()
        self._validate_shape(n_paths, horizon)
        rng = np.random.default_rng(seed)
        shape = (n_paths, horizon, len(self.stocks))
        gaussian = rng.standard_normal(shape) @ self.covariance_root_.T
        df = self.student_df_
        gaussian *= np.sqrt((df - 2.0) / df)
        common_scale = np.sqrt(
            rng.chisquare(df, size=(n_paths, horizon, 1)) / df
        )
        return gaussian / common_scale

    def sample_blocks(self, n_paths, horizon, block_length=5, seed=0):
        """Draw moving blocks of full cross-sectional innovation vectors."""
        self._check_fitted()
        self._validate_shape(n_paths, horizon)
        if block_length < 1:
            raise ValueError("block_length must be positive.")
        if block_length > len(self.values_):
            raise ValueError("block_length exceeds the fitted history.")

        rng = np.random.default_rng(seed)
        result = np.empty(
            (n_paths, horizon, len(self.stocks)), dtype=float
        )
        for path in range(n_paths):
            offset = 0
            while offset < horizon:
                width = min(block_length, horizon - offset)
                max_start = len(self.values_) - width
                start = int(rng.integers(0, max_start + 1))
                result[path, offset:offset + width] = self.values_[
                    start:start + width
                ]
                offset += width
        return result

