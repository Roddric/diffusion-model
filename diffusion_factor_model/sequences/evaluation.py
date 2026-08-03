"""Path-level scoring for rolling conditional forecasts."""

import numpy as np
from scipy.spatial.distance import pdist


def _energy_score(paths, target):
    samples = paths.reshape(len(paths), -1)
    observed = target.reshape(-1)
    scale = np.sqrt(len(observed))
    first = np.linalg.norm(samples - observed, axis=1).mean() / scale
    if len(samples) <= 64:
        pairwise = np.linalg.norm(
            samples[:, None, :] - samples[None, :, :], axis=2
        ).mean() / scale
    else:
        # The empirical score includes both directions and the zero diagonal.
        # pdist avoids the O(M^2 * H * d) temporary used by broadcasting and
        # makes high-path-count Monte Carlo sensitivity checks practical.
        pairwise = (
            2.0 * pdist(samples, metric="euclidean").sum()
            / (len(samples) ** 2)
            / scale
        )
    return float(first - 0.5 * pairwise)


def _variogram_score(paths, target):
    target_increment = np.abs(np.diff(target, axis=0)) ** 0.5
    sample_increment = np.abs(np.diff(paths, axis=1)) ** 0.5
    expected_increment = sample_increment.mean(axis=0)
    return float(np.mean((target_increment - expected_increment) ** 2))


def _maximum_drawdown(returns):
    wealth = np.cumprod(1.0 + returns)
    running_peak = np.maximum.accumulate(wealth)
    return float(np.min(wealth / running_peak - 1.0))


class PathForecastEvaluator:
    def __init__(self, n_mean_factors):
        self.n_mean_factors = n_mean_factors

    def score(self, state_paths, return_paths, target_states, target_returns):
        state_mean = state_paths.mean(axis=0)
        predicted_daily_vol = return_paths.std(axis=2).mean(axis=0)
        target_daily_vol = target_returns.std(axis=1)

        target_portfolio = target_returns.mean(axis=1)
        portfolio_paths = return_paths.mean(axis=2)
        predicted_drawdowns = np.asarray(
            [
                _maximum_drawdown(path.mean(axis=1))
                for path in return_paths
            ]
        )
        target_drawdown = _maximum_drawdown(target_portfolio)

        state_error = state_mean - target_states
        return_quantile = np.quantile(return_paths, 0.05)
        target_quantile = np.quantile(target_returns, 0.05)
        return_mean = return_paths.mean(axis=0)
        return_scale = max(target_returns.std(), 1e-12)
        return_rmse = np.sqrt(np.mean((return_mean - target_returns) ** 2))
        portfolio_var = np.quantile(portfolio_paths, 0.05, axis=0)
        portfolio_var_error = target_portfolio - portfolio_var
        portfolio_var_pinball = np.maximum(
            0.05 * portfolio_var_error,
            -0.95 * portfolio_var_error,
        )
        predicted_portfolio_volatility = np.std(
            portfolio_paths, axis=1, ddof=1
        ).mean()
        realized_portfolio_volatility = np.std(
            target_portfolio, ddof=1
        )
        flat_generated = return_paths.reshape(-1, return_paths.shape[-1])
        predicted_covariance = np.cov(flat_generated, rowvar=False)
        realized_covariance = np.cov(target_returns, rowvar=False)
        covariance_scale = max(
            np.linalg.norm(realized_covariance, ord="fro"), 1e-12
        )

        return {
            "state_rmse": float(np.sqrt(np.mean(state_error ** 2))),
            "mean_factor_rmse": float(
                np.sqrt(
                    np.mean(state_error[:, : self.n_mean_factors] ** 2)
                )
            ),
            "logvol_factor_rmse": float(
                np.sqrt(
                    np.mean(state_error[:, self.n_mean_factors:] ** 2)
                )
            ),
            "state_energy_score": _energy_score(state_paths, target_states),
            "state_variogram_score": _variogram_score(
                state_paths, target_states
            ),
            "return_rmse_scaled": float(return_rmse / return_scale),
            "daily_volatility_mae": float(
                np.mean(np.abs(predicted_daily_vol - target_daily_vol))
            ),
            "tail_quantile_error": float(
                abs(return_quantile - target_quantile)
            ),
            "max_drawdown_error": float(
                abs(predicted_drawdowns.mean() - target_drawdown)
            ),
            "portfolio_path_energy_score": _energy_score(
                portfolio_paths, target_portfolio
            ),
            "portfolio_volatility_error": float(
                abs(
                    predicted_portfolio_volatility
                    - realized_portfolio_volatility
                )
            ),
            "portfolio_var_05_pinball": float(
                np.mean(portfolio_var_pinball)
            ),
            "portfolio_var_05_coverage_error": float(
                abs(np.mean(target_portfolio < portfolio_var) - 0.05)
            ),
            "covariance_frobenius_scaled_error": float(
                np.linalg.norm(
                    predicted_covariance - realized_covariance, ord="fro"
                )
                / covariance_scale
            ),
        }

    @staticmethod
    def aggregate(origin_scores):
        methods = sorted(origin_scores)
        output = {}
        for method in methods:
            keys = origin_scores[method][0]
            output[method] = {
                key: {
                    "mean": float(
                        np.mean([row[key] for row in origin_scores[method]])
                    ),
                    "std": float(
                        np.std([row[key] for row in origin_scores[method]])
                    ),
                }
                for key in keys
            }
        return output
