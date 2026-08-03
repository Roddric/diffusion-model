"""Post-hoc comparison with additional classical and nonlinear state baselines."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from config.config import load_config
from diffusion.conditional_path import VARResidualPathDiffusion
from dynamics.var import LatentVAR
from phase2c_benchmark import _restore_checkpoint
from research.confirm_phase2f import (
    _load_frozen_history_with_current_extension,
    _sha256,
)
from research.freeze_phase2f import _panel_fingerprint
from research.phase2f_pooling import _linear_pool
from research.robustness import dependence_robust_comparison
from sequences.dataset import FactorStateSequenceBuilder
from sequences.evaluation import _energy_score


MODEL_NAME = "Phase2F-Validation-Pooled-Diffusion"
METRICS = ("state_energy_score", "state_rmse")


def _stabilize(transition, intercept, states, cap=0.98):
    radius = float(max(abs(np.linalg.eigvals(transition))))
    if radius > cap:
        transition = transition * (cap / radius)
        mean = states.mean(axis=0)
        intercept = mean - transition @ mean
    return transition, intercept


class DiagonalARGaussian:
    """Dimension-wise AR(1) mean with correlated Gaussian innovations."""

    def fit(self, states):
        states = np.asarray(states, dtype=float)
        x, y = states[:-1], states[1:]
        coefficients = []
        intercepts = []
        for dimension in range(states.shape[1]):
            design = np.column_stack([np.ones(len(x)), x[:, dimension]])
            beta = np.linalg.lstsq(design, y[:, dimension], rcond=None)[0]
            intercepts.append(beta[0])
            coefficients.append(float(np.clip(beta[1], -0.98, 0.98)))
        self.transition_ = np.diag(coefficients)
        self.intercept_ = np.asarray(intercepts)
        fitted = self.intercept_ + x @ self.transition_.T
        self.innovation_cov_ = LedoitWolf().fit(y - fitted).covariance_
        self.innovation_cov_ += 1e-8 * np.eye(states.shape[1])
        return self

    def simulate(self, n_steps, seed, initial_state):
        rng = np.random.default_rng(seed)
        state = np.asarray(initial_state, dtype=float).copy()
        path = []
        for _ in range(n_steps):
            state = (
                self.intercept_
                + self.transition_ @ state
                + rng.multivariate_normal(
                    np.zeros(len(state)), self.innovation_cov_
                )
            )
            path.append(state.copy())
        return np.asarray(path)


class RidgeVARGaussian(DiagonalARGaussian):
    """Ridge VAR(1) with alpha selected on a training-only terminal block."""

    def fit(self, states, alphas=(0.01, 0.1, 1.0, 10.0, 100.0)):
        states = np.asarray(states, dtype=float)
        x, y = states[:-1], states[1:]
        split = max(int(len(x) * 0.8), 2)
        validation_scores = {}
        for alpha in alphas:
            candidate = Ridge(alpha=alpha).fit(x[:split], y[:split])
            validation_scores[str(alpha)] = float(
                np.sqrt(np.mean((candidate.predict(x[split:]) - y[split:]) ** 2))
            )
        self.selected_alpha_ = min(
            alphas, key=lambda value: (validation_scores[str(value)], value)
        )
        self.validation_scores_ = validation_scores
        model = Ridge(alpha=self.selected_alpha_).fit(x, y)
        transition, intercept = _stabilize(
            model.coef_.copy(), model.intercept_.copy(), states
        )
        self.transition_ = transition
        self.intercept_ = intercept
        fitted = self.intercept_ + x @ self.transition_.T
        self.innovation_cov_ = LedoitWolf().fit(y - fitted).covariance_
        self.innovation_cov_ += 1e-8 * np.eye(states.shape[1])
        return self


class GradientBoostedStateAR:
    """Fixed-complexity nonlinear AR(1) with joint residual resampling."""

    def fit(self, states):
        states = np.asarray(states, dtype=float)
        x, y = states[:-1], states[1:]
        self.models_ = []
        fitted = []
        for dimension in range(states.shape[1]):
            model = HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=200,
                max_depth=3,
                l2_regularization=1.0,
                random_state=7000 + dimension,
            ).fit(x, y[:, dimension])
            self.models_.append(model)
            fitted.append(model.predict(x))
        self.residuals_ = y - np.column_stack(fitted)
        return self

    def simulate(self, n_steps, seed, initial_state):
        rng = np.random.default_rng(seed)
        state = np.asarray(initial_state, dtype=float).copy()
        path = []
        for _ in range(n_steps):
            conditional_mean = np.asarray(
                [model.predict(state[None, :])[0] for model in self.models_]
            )
            innovation = self.residuals_[rng.integers(0, len(self.residuals_))]
            state = conditional_mean + innovation
            path.append(state.copy())
        return np.asarray(path)


def _score(paths, target):
    return {
        "state_energy_score": _energy_score(paths, target),
        "state_rmse": float(
            np.sqrt(np.mean((paths.mean(axis=0) - target) ** 2))
        ),
    }


def run(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite baseline audit: {output}")
    frozen = json.loads(Path(args.frozen_protocol).read_text(encoding="utf-8"))
    config = load_config(args.config)
    frozen_config = load_config(args.freeze_config)
    returns, market = _load_frozen_history_with_current_extension(
        frozen_config, config, args.max_stocks
    )
    pre2024 = returns.loc[returns.index < "2024-01-01"]
    if _panel_fingerprint(
        pre2024, market.reindex(pre2024.index)
    ) != frozen["protocol"]["pre2024_panel_fingerprint"]:
        raise ValueError("Frozen pre-2024 panel fingerprint mismatch.")
    splits = FactorStateSequenceBuilder(
        config,
        context_length=config.temporal.context_length,
        horizon=config.temporal.horizon,
        evaluation_stride=config.temporal.horizon,
        train_end_date=frozen["protocol"]["train_end"],
        validation_end_date=frozen["protocol"][
            "pool_and_checkpoint_validation_end"
        ],
    ).build(returns, market)
    states = splits.train_states.values
    gaussian_var = LatentVAR().fit(states)
    diagonal_ar = DiagonalARGaussian().fit(states)
    ridge_var = RidgeVARGaussian().fit(states)
    boosted_ar = GradientBoostedStateAR().fit(states)
    models = []
    for frozen_seed in frozen["seed_reports"]:
        checkpoint = Path(frozen_seed["checkpoint"])
        if _sha256(checkpoint) != frozen_seed["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
        model = VARResidualPathDiffusion(
            config,
            state_dim=states.shape[1],
            horizon=config.temporal.horizon,
            n_mean_factors=splits.latent_metadata["n_mean_factors"],
        )
        _restore_checkpoint(model, splits, checkpoint)
        models.append((frozen_seed, model))

    method_names = (
        MODEL_NAME,
        "Gaussian-VAR",
        "Student-t-VAR",
        "Diagonal-AR-Gaussian",
        "Ridge-VAR-Gaussian",
        "Gradient-Boosted-AR-Residual-Bootstrap",
        "Persistence",
    )
    origin_scores = {name: [] for name in method_names}
    for origin in range(len(splits.test)):
        context = splits.test.context[origin]
        target = splits.test.target[origin]
        seed_root = 1200000 + origin * 1000
        simulators = {
            "Gaussian-VAR": gaussian_var.simulate,
            "Student-t-VAR": gaussian_var.simulate_student_t,
            "Diagonal-AR-Gaussian": diagonal_ar.simulate,
            "Ridge-VAR-Gaussian": ridge_var.simulate,
            "Gradient-Boosted-AR-Residual-Bootstrap": boosted_ar.simulate,
        }
        paths = {}
        for method_index, (name, simulator) in enumerate(simulators.items()):
            paths[name] = np.stack(
                [
                    simulator(
                        config.temporal.horizon,
                        seed=seed_root + method_index * 100000 + path,
                        initial_state=context[-1],
                    )
                    for path in range(args.paths)
                ]
            )
            origin_scores[name].append(_score(paths[name], target))
        persistence = np.repeat(
            context[-1][None, None, :],
            args.paths,
            axis=0,
        )
        persistence = np.repeat(
            persistence, config.temporal.horizon, axis=1
        )
        origin_scores["Persistence"].append(_score(persistence, target))

        seed_scores = []
        for frozen_seed, model in models:
            diffusion = model.sample(
                context,
                n_paths=args.paths,
                seed=frozen_seed["seed"] * 1000000 + seed_root,
            )
            pooled = _linear_pool(
                diffusion,
                paths["Gaussian-VAR"],
                frozen_seed["selected_diffusion_weight"],
            )
            seed_scores.append(_score(pooled, target))
        origin_scores[MODEL_NAME].append(
            {
                metric: float(
                    np.mean([row[metric] for row in seed_scores])
                )
                for metric in METRICS
            }
        )

    metrics = {
        method: {
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in METRICS
        }
        for method, rows in origin_scores.items()
    }
    paired = {
        baseline: {
            metric: dependence_robust_comparison(
                [row[metric] for row in origin_scores[MODEL_NAME]],
                [row[metric] for row in origin_scores[baseline]],
                seed=20260804 + baseline_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(METRICS)
        }
        for baseline_index, baseline in enumerate(method_names[1:])
    }
    report = {
        "analyzed_at": datetime.now().isoformat(),
        "status": "posthoc_expanded_baseline_audit",
        "interpretation_boundary": (
            "Additional baselines specified after the primary score was known; "
            "exploratory comparison on a consumed holdout."
        ),
        "protocol": {
            "n_paths": args.paths,
            "n_origins": len(splits.test),
            "horizon": config.temporal.horizon,
            "weights_reselected": False,
            "ridge_alpha_selected_on": "terminal 20% of training transitions",
            "ridge_alpha": ridge_var.selected_alpha_,
            "ridge_validation_rmse": ridge_var.validation_scores_,
            "boosted_model_tuning": "fixed, no holdout selection",
        },
        "metrics": metrics,
        "paired_pool_minus_baseline": paired,
        "origin_level_scores": origin_scores,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_sp500_confirmation.yaml")
    parser.add_argument("--freeze-config", default="research_sp500_freeze.yaml")
    parser.add_argument(
        "--frozen-protocol",
        default="research_output/sp500_frozen/frozen_protocol.json",
    )
    parser.add_argument(
        "--output",
        default=(
            "research_output/sp500_confirmation/"
            "posthoc_expanded_baselines.json"
        ),
    )
    parser.add_argument("--paths", type=int, default=100)
    parser.add_argument("--max-stocks", type=int, default=100)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
