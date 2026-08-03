"""Phase 2A temporal path-forecasting baselines."""

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf

if __package__ and __package__.startswith("diffusion_factor_model."):
    from ..dynamics.var import LatentVAR
    from ..residuals.garch import GARCHModeler
else:
    from dynamics.var import LatentVAR
    from residuals.garch import GARCHModeler


@dataclass
class ForecastEnsemble:
    states: dict
    returns: dict
    innovations: dict


class Phase2ABaselines:
    """Gaussian VAR, VAR-GARCH, and historical block-bootstrap forecasts."""

    METHODS = (
        "Gaussian-VAR",
        "Student-t-VAR",
        "Residual-Bootstrap-VAR",
        "VAR-GARCH",
        "Block-Bootstrap",
    )

    def __init__(self, config, splits):
        self.config = config
        self.splits = splits
        self.var = LatentVAR().fit(splits.train_states.values)
        self.garch = GARCHModeler(config)
        self.stocks = splits.reconstructor.stocks

        innovations = splits.innovations_train.reindex(
            columns=self.stocks
        ).dropna()
        if len(innovations) < 3:
            raise ValueError("Not enough training innovations for path baselines.")
        self.innovations_train = innovations
        self.gaussian_residual_cov = LedoitWolf().fit(
            innovations.values
        ).covariance_
        self.garch.fit_all(innovations)

        common_dates = splits.train_states.index.intersection(
            innovations.index
        )
        self.block_states = splits.train_states.loc[common_dates]
        self.block_innovations = innovations.loc[common_dates]

    def forecast(self, context, horizon, n_paths=20, seed=0):
        if context.ndim != 2:
            raise ValueError("context must have shape (context_length, state_dim).")
        if context.shape[1] != self.splits.train_states.shape[1]:
            raise ValueError("context state dimension does not match training states.")
        if horizon < 1 or n_paths < 1:
            raise ValueError("horizon and n_paths must be positive.")
        if len(self.block_states) < horizon:
            raise ValueError("Training state history is shorter than the horizon.")

        rng = np.random.default_rng(seed)
        var_states = np.stack(
            [
                self.var.simulate(
                    horizon,
                    seed=seed + path,
                    initial_state=context[-1],
                )
                for path in range(n_paths)
            ]
        )
        student_states = np.stack(
            [
                self.var.simulate_student_t(
                    horizon,
                    seed=seed + 10000 + path,
                    initial_state=context[-1],
                )
                for path in range(n_paths)
            ]
        )
        residual_bootstrap_states = np.stack(
            [
                self.var.simulate_residual_bootstrap(
                    horizon,
                    seed=seed + 20000 + path,
                    initial_state=context[-1],
                    block_length=min(5, horizon),
                )
                for path in range(n_paths)
            ]
        )
        gaussian_innovations = rng.multivariate_normal(
            np.zeros(len(self.stocks)),
            self.gaussian_residual_cov,
            size=(n_paths, horizon),
        )

        garch_innovations = (
            self.garch.sample_all(n_paths * horizon, seed=seed)
            .reindex(columns=self.stocks)
            .fillna(0.0)
            .values.reshape(n_paths, horizon, len(self.stocks))
        )

        max_start = len(self.block_states) - horizon
        starts = rng.integers(0, max_start + 1, size=n_paths)
        block_states = np.stack(
            [
                self.block_states.iloc[start:start + horizon].values
                for start in starts
            ]
        )
        block_innovations = np.stack(
            [
                self.block_innovations.iloc[start:start + horizon].values
                for start in starts
            ]
        )

        state_paths = {
            "Gaussian-VAR": var_states,
            "Student-t-VAR": student_states,
            "Residual-Bootstrap-VAR": residual_bootstrap_states,
            "VAR-GARCH": var_states.copy(),
            "Block-Bootstrap": block_states,
        }
        innovation_paths = {
            "Gaussian-VAR": gaussian_innovations,
            "Student-t-VAR": garch_innovations,
            "Residual-Bootstrap-VAR": garch_innovations,
            "VAR-GARCH": garch_innovations,
            "Block-Bootstrap": block_innovations,
        }
        return_paths = {}
        for method in self.METHODS:
            reconstructed = []
            for path in range(n_paths):
                residuals = self.innovations_train.iloc[:horizon].copy()
                residuals.loc[:, :] = innovation_paths[method][path]
                reconstructed.append(
                    self.splits.reconstructor.reconstruct(
                        state_paths[method][path], residuals, verbose=False
                    )[self.stocks].values
                )
            return_paths[method] = np.stack(reconstructed)

        return ForecastEnsemble(
            states=state_paths,
            returns=return_paths,
            innovations=innovation_paths,
        )
