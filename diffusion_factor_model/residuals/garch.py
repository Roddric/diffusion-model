"""Residual modeling with GJR-GARCH for idiosyncratic returns."""

import numpy as np
import pandas as pd
from arch import arch_model
from typing import Dict
from tqdm import tqdm


class GARCHModeler:
    """Model idiosyncratic residuals with GJR-GARCH(1,1) + Student-t."""

    def __init__(self, config):
        self.config = config
        self.models = {}
        self.scales = {}
        # The small-N regime we want to probe has fewer than 100 residual observations.
        # Fits get noisier, but every method shares the same GARCH, so the comparison is fair.
        self.min_obs = getattr(config.residuals, "min_obs", 60)
        self.p = config.residuals.p
        self.q = config.residuals.q
        self.o = (
            0
            if config.residuals.model_type.lower() == "garch"
            else config.residuals.o
        )
        distribution = config.residuals.distribution.lower()
        self.distribution = {
            "student_t": "studentst",
            "student-t": "studentst",
            "t": "studentst",
            "gaussian": "normal",
        }.get(distribution, distribution)

    def fit_all(self, residuals: pd.DataFrame) -> Dict:
        """Fit GJR-GARCH to each stock's residual series."""
        results = {}
        for col in tqdm(residuals.columns, desc="Fitting GARCH"):
            series = residuals[col].dropna()
            if len(series) < self.min_obs:
                continue
            # ARCH is numerically happiest around unit scale. Raw returns are ~1%;
            # volatility-standardized innovations are already O(1).
            scale = 100.0 if series.std() < 0.1 else 1.0
            scaled = series * scale
            try:
                model = arch_model(
                    scaled,
                    vol="Garch",
                    p=self.p, q=self.q, o=self.o,
                    dist=self.distribution,
                    rescale=False
                )
                res = model.fit(disp="off")
                results[col] = res
                self.scales[col] = scale
            except Exception:
                continue

        print(f"Fitted GARCH on {len(results)}/{len(residuals.columns)} stocks")
        self.models = results
        return results

    def sample(self, stock: str, horizon: int, seed: int = 0) -> np.ndarray:
        """Sample future residuals from fitted model."""
        if stock not in self.models:
            raise KeyError(f"No model for {stock}")
        res = self.models[stock]
        distribution = type(res.model.distribution)(seed=seed)
        n_distribution_parameters = distribution.num_params
        if n_distribution_parameters:
            distribution_parameters = res.params.iloc[
                -n_distribution_parameters:
            ].values
        else:
            distribution_parameters = np.empty(0)
        rng = distribution.simulate(distribution_parameters)
        sim = res.forecast(
            horizon=horizon,
            method="simulation",
            simulations=1,
            rng=rng,
            reindex=False,
        )
        # values is (n_origins, n_simulations, horizon): take the full simulated path
        # for origin 0 / simulation 0. Indexing [0, :, 0] would return a single scalar,
        # which pandas then broadcasts to every sample.
        return sim.simulations.values[0, 0, :] / self.scales[stock]

    def sample_all(self, horizon: int, seed: int = 0) -> pd.DataFrame:
        """Sample residuals for all stocks."""
        data = {}
        stock_seeds = np.random.SeedSequence(seed).spawn(len(self.models))
        for stock, stock_seed in zip(self.models, stock_seeds):
            data[stock] = self.sample(
                stock,
                horizon,
                seed=int(stock_seed.generate_state(1)[0]),
            )
        return pd.DataFrame(data)
