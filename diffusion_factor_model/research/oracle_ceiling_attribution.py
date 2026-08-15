"""Pre-2024 oracle-ceiling attribution for return reconstruction.

This experiment is diagnostic only.  It replaces forecast components with
their realized future values inside the already consumed 2022--2023 S&P 500
development period.  It must never load the 2024-current confirmation panel,
and none of its oracle interventions is a feasible forecasting method.
"""

import argparse
import hashlib
import itertools
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from config.config import load_config
from data.yf_loader import YFinanceDataPipeline
from diffusion.conditional_path import VARResidualPathDiffusion
from phase2b_benchmark import reconstruct_paths
from phase2c_benchmark import _restore_checkpoint
from research.market_selection import _aggregate_seed_scores
from research.phase2f_pooling import _linear_pool
from sequences.baselines import Phase2ABaselines
from sequences.dataset import FactorStateSequenceBuilder
from sequences.evaluation import PathForecastEvaluator, _energy_score, _variogram_score


MEAN_STATE = 1
VOL_STATE = 2
INNOVATION = 4
COMPONENTS = {
    MEAN_STATE: "mean_state",
    VOL_STATE: "volatility_state",
    INNOVATION: "innovation",
}
RETURN_METRICS = (
    "return_energy_score",
    "return_variogram_score",
    "daily_volatility_mae",
    "tail_quantile_error",
    "max_drawdown_error",
)
COALITION_NAMES = {
    0: "current_forecast",
    MEAN_STATE: "oracle_mean_state",
    VOL_STATE: "oracle_volatility_state",
    MEAN_STATE | VOL_STATE: "oracle_all_states",
    INNOVATION: "oracle_innovation",
    MEAN_STATE | INNOVATION: "oracle_mean_and_innovation",
    VOL_STATE | INNOVATION: "oracle_volatility_and_innovation",
    MEAN_STATE | VOL_STATE | INNOVATION: "full_oracle",
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _metric_means_match(actual, expected, tolerance=1e-12):
    return all(
        abs(actual[metric]["mean"] - expected[metric]["mean"]) <= tolerance
        for metric in RETURN_METRICS
    )


def _realized_innovations(reconstructor, target_states, target_returns):
    """Recover the exact innovation path under the fixed training mapping."""
    stocks = reconstructor.stocks
    conditional_mean = reconstructor.reconstruct(
        target_states, None, verbose=False
    )[stocks].to_numpy()
    unit = pd.DataFrame(
        np.ones((len(target_states), len(stocks))), columns=stocks
    )
    mean_plus_scale = reconstructor.reconstruct(
        target_states, unit, verbose=False
    )[stocks].to_numpy()
    scale = np.maximum(mean_plus_scale - conditional_mean, 1e-12)
    return (np.asarray(target_returns) - conditional_mean) / scale


def _shapley_values(losses):
    """Return loss reductions allocated across the three oracle components."""
    expected = set(range(8))
    if set(losses) != expected:
        raise ValueError("losses must contain every coalition mask from 0 to 7.")
    n_components = len(COMPONENTS)
    values = {}
    for bit, name in COMPONENTS.items():
        contribution = 0.0
        other_bits = [candidate for candidate in COMPONENTS if candidate != bit]
        for width in range(len(other_bits) + 1):
            weight = (
                math.factorial(width)
                * math.factorial(n_components - width - 1)
                / math.factorial(n_components)
            )
            for subset in itertools.combinations(other_bits, width):
                mask = sum(subset)
                contribution += weight * (losses[mask] - losses[mask | bit])
        values[name] = float(contribution)
    return values


def _normalized_losses(metric_means, metrics=RETURN_METRICS):
    """Create ratio-scale coalition losses, with the current forecast equal to 1."""
    baseline = metric_means[0]
    return {
        mask: float(
            np.mean(
                [
                    values[metric] / max(baseline[metric], 1e-12)
                    for metric in metrics
                ]
            )
        )
        for mask, values in metric_means.items()
    }


def _summarize_attribution(per_origin, bootstrap_samples=10000, seed=20260815):
    """Compute point and paired-origin bootstrap Shapley attribution."""
    metric_means = {
        mask: {
            metric: float(np.mean(values[metric]))
            for metric in RETURN_METRICS
        }
        for mask, values in per_origin.items()
    }
    point_losses = _normalized_losses(metric_means)
    point = _shapley_values(point_losses)
    total = point_losses[0] - point_losses[7]

    n_origins = len(next(iter(per_origin.values()))[RETURN_METRICS[0]])
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in COMPONENTS.values()}
    for _ in range(bootstrap_samples):
        indices = rng.integers(0, n_origins, n_origins)
        sampled_means = {
            mask: {
                metric: float(np.mean(np.asarray(values[metric])[indices]))
                for metric in RETURN_METRICS
            }
            for mask, values in per_origin.items()
        }
        sampled = _shapley_values(_normalized_losses(sampled_means))
        for name, value in sampled.items():
            draws[name].append(value)

    components = {}
    for name, value in point.items():
        sample = np.asarray(draws[name])
        components[name] = {
            "normalized_loss_reduction": value,
            "share_of_full_oracle_reduction": float(value / max(total, 1e-12)),
            "paired_origin_bootstrap_ci_95": [
                float(np.quantile(sample, 0.025)),
                float(np.quantile(sample, 0.975)),
            ],
            "bootstrap_probability_positive": float(np.mean(sample > 0.0)),
        }

    by_metric = {}
    for metric in RETURN_METRICS:
        losses = {
            mask: values[metric] / max(metric_means[0][metric], 1e-12)
            for mask, values in metric_means.items()
        }
        contributions = _shapley_values(losses)
        metric_total = losses[0] - losses[7]
        by_metric[metric] = {
            "coalition_loss_ratios": {
                COALITION_NAMES[mask]: value for mask, value in losses.items()
            },
            "component_loss_reductions": contributions,
            "component_shares": {
                name: float(value / max(metric_total, 1e-12))
                for name, value in contributions.items()
            },
        }

    return {
        "loss_definition": (
            "arithmetic mean of the five return-metric loss ratios versus the "
            "current Phase 2F pool; lower is better"
        ),
        "coalition_normalized_losses": {
            COALITION_NAMES[mask]: value for mask, value in point_losses.items()
        },
        "full_oracle_normalized_loss_reduction": total,
        "efficiency_check_sum_of_components": float(sum(point.values())),
        "components": components,
        "by_metric": by_metric,
        "bootstrap": {
            "method": "paired resampling of development origins",
            "samples": bootstrap_samples,
            "seed": seed,
            "n_origins": n_origins,
        },
    }


def _score_return_paths(
    evaluator, states, returns, target_states, target_returns, return_scale
):
    scores = evaluator.score(states, returns, target_states, target_returns)
    scores["return_energy_score"] = _energy_score(
        returns / return_scale[None, None, :],
        target_returns / return_scale[None, :],
    )
    scores["return_variogram_score"] = _variogram_score(
        returns / return_scale[None, None, :],
        target_returns / return_scale[None, :],
    )
    return scores


def _origin_rows(config, splits, baselines, returns, n_paths):
    rows = []
    origins = []
    for origin in range(len(splits.test)):
        context = splits.test.context[origin]
        forecast = baselines.forecast(
            context,
            config.temporal.horizon,
            n_paths=n_paths,
            seed=600000 + origin * n_paths,
        )
        target_dates = splits.test.target_dates[origin]
        target_returns = returns.reindex(target_dates)[baselines.stocks].to_numpy()
        target_states = splits.test.target[origin]
        rows.append(
            {
                "context": context,
                "forecast_states": forecast.states["Gaussian-VAR"],
                "forecast_innovations": forecast.innovations["VAR-GARCH"],
                "target_states": target_states,
                "target_returns": target_returns,
                "realized_innovations": _realized_innovations(
                    splits.reconstructor, target_states, target_returns
                ),
                "target_dates": target_dates,
            }
        )
        origins.append(
            {
                "context_end": str(
                    np.datetime_as_string(
                        splits.test.context_dates[origin, -1], unit="D"
                    )
                ),
                "target_start": str(np.datetime_as_string(target_dates[0], unit="D")),
                "target_end": str(np.datetime_as_string(target_dates[-1], unit="D")),
            }
        )
    return rows, origins


def _coalition_scores(
    config, splits, evaluator, models, rows, return_scale, n_paths
):
    scores = {mask: [[] for _ in models] for mask in range(8)}
    n_mean = splits.latent_metadata["n_mean_factors"]
    stocks = splits.reconstructor.stocks
    for model_index, (seed, weight, model) in enumerate(models):
        for origin, row in enumerate(rows):
            diffusion = model.sample(
                row["context"],
                n_paths=n_paths,
                seed=seed * 10000 + 700000 + origin * n_paths,
            )
            current_states = _linear_pool(
                diffusion, row["forecast_states"], weight
            )
            oracle_innovations = np.repeat(
                row["realized_innovations"][None, :, :], n_paths, axis=0
            )
            for mask in range(8):
                states = current_states.copy()
                if mask & MEAN_STATE:
                    states[:, :, :n_mean] = row["target_states"][None, :, :n_mean]
                if mask & VOL_STATE:
                    states[:, :, n_mean:] = row["target_states"][None, :, n_mean:]
                innovations = (
                    oracle_innovations
                    if mask & INNOVATION
                    else row["forecast_innovations"]
                )
                reconstructed = reconstruct_paths(
                    splits, states, innovations, stocks
                )
                scores[mask][model_index].append(
                    _score_return_paths(
                        evaluator,
                        states,
                        reconstructed,
                        row["target_states"],
                        row["target_returns"],
                        return_scale,
                    )
                )
    return scores


def _aggregate_by_origin(scores):
    return {
        mask: _aggregate_seed_scores(seed_scores) for mask, seed_scores in scores.items()
    }


def _coalition_metrics(per_origin):
    return {
        COALITION_NAMES[mask]: {
            metric: {
                "mean": float(np.mean(values)),
                "std_across_origins": float(np.std(values)),
            }
            for metric, values in metrics.items()
        }
        for mask, metrics in per_origin.items()
    }


def _blocked_origin_folds(n_origins, n_folds=5):
    return [fold for fold in np.array_split(np.arange(n_origins), n_folds) if len(fold)]


def _representation_drift_diagnostic(splits, rows, returns, market, n_folds=5):
    """Cross-fit outcome-informed loadings to measure fixed-map representation error."""
    target_dates = np.concatenate([row["target_dates"] for row in rows])
    target_states = np.concatenate([row["target_states"] for row in rows])
    target_returns = np.concatenate([row["target_returns"] for row in rows])
    n_mean = splits.latent_metadata["n_mean_factors"]
    mean_factors, _ = splits.parametrizer.inverse_transform(
        target_states, splits.latent_metadata
    )
    fixed_mean = splits.reconstructor.reconstruct(
        target_states, None, verbose=False
    )[splits.reconstructor.stocks].to_numpy()
    origin_ids = np.repeat(np.arange(len(rows)), len(rows[0]["target_dates"]))
    folds = _blocked_origin_folds(len(rows), n_folds=n_folds)

    crossfit_mean = np.empty_like(target_returns)
    design = np.column_stack([np.ones(len(mean_factors)), mean_factors[:, :n_mean]])
    for fold in folds:
        held_out = np.isin(origin_ids, fold)
        beta = np.linalg.lstsq(
            design[~held_out], target_returns[~held_out], rcond=None
        )[0]
        crossfit_mean[held_out] = design[held_out] @ beta
    scale = np.std(target_returns, axis=0, ddof=0).clip(min=1e-6)
    fixed_mean_rmse = float(
        np.sqrt(np.mean(((fixed_mean - target_returns) / scale) ** 2))
    )
    crossfit_mean_rmse = float(
        np.sqrt(np.mean(((crossfit_mean - target_returns) / scale) ** 2))
    )

    train_returns = splits.train_returns[splits.reconstructor.stocks]
    train_mean_result = {
        "factors": splits.extractor._mean_factor_frame(
            train_returns, splits.train_market
        ),
        "loadings": splits.extractor.mean_loadings_,
        "intercepts": splits.extractor.mean_intercepts_,
    }
    train_residuals = splits.extractor.compute_residuals(
        train_returns, train_mean_result
    )
    future_returns = returns.loc[returns.index > train_returns.index[-1], splits.reconstructor.stocks]
    future_market = market.reindex(future_returns.index)
    future_mean_result = splits.extractor.transform_mean_factors(
        future_returns,
        future_market,
        history_returns=train_returns,
        history_market=splits.train_market,
    )
    future_residuals = splits.extractor.compute_residuals(
        future_returns, future_mean_result
    )
    log_variance = splits.extractor._log_variance_features(
        future_residuals, history_residuals=train_residuals
    ).reindex(target_dates)
    if log_variance.isna().any().any():
        raise ValueError("Development target log-variance features are incomplete.")
    log_variance_values = log_variance.to_numpy()
    fixed_scores = splits.extractor.vol_pca_.transform(log_variance)
    fixed_log_variance = splits.extractor.vol_pca_.inverse_transform(fixed_scores)
    crossfit_log_variance = np.empty_like(log_variance_values)
    n_components = splits.latent_metadata["n_vol_factors"]
    for fold in folds:
        held_out = np.isin(origin_ids, fold)
        pca = PCA(n_components=n_components).fit(log_variance_values[~held_out])
        crossfit_log_variance[held_out] = pca.inverse_transform(
            pca.transform(log_variance_values[held_out])
        )
    fixed_vol_rmse = float(
        np.sqrt(np.mean((fixed_log_variance - log_variance_values) ** 2))
    )
    crossfit_vol_rmse = float(
        np.sqrt(np.mean((crossfit_log_variance - log_variance_values) ** 2))
    )

    return {
        "description": (
            "Five-block cross-fitted, outcome-informed representation test on "
            "2022-2023 development origins; diagnostic only and not feasible forecasting."
        ),
        "fold_unit": "non-overlapping 20-session forecast origin",
        "n_folds": len(folds),
        "n_rows": len(target_dates),
        "mean_loading_scaled_rmse": {
            "fixed_2015_2020_mapping": fixed_mean_rmse,
            "crossfit_development_mapping": crossfit_mean_rmse,
            "ratio_crossfit_vs_fixed": crossfit_mean_rmse / max(fixed_mean_rmse, 1e-12),
        },
        "volatility_loading_logvariance_rmse": {
            "fixed_2015_2020_pca": fixed_vol_rmse,
            "crossfit_development_pca": crossfit_vol_rmse,
            "ratio_crossfit_vs_fixed": crossfit_vol_rmse / max(fixed_vol_rmse, 1e-12),
        },
        "interpretation_guardrail": (
            "Ratios below one indicate representation drift/headroom, not a valid "
            "out-of-sample forecasting improvement."
        ),
    }


def run(args):
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Oracle attribution output already exists: {output}. "
            "Pass --overwrite only for deterministic development reruns."
        )
    config = load_config(args.config)
    if pd.to_datetime(config.data.end_date) >= pd.Timestamp("2024-01-02"):
        raise ValueError(
            "Oracle attribution config must end before 2024; confirmation data "
            "cannot be used."
        )
    returns, market = YFinanceDataPipeline(config).load_all_data(
        max_stocks=args.max_stocks
    )
    splits = FactorStateSequenceBuilder(
        config,
        context_length=config.temporal.context_length,
        horizon=config.temporal.horizon,
        evaluation_stride=config.temporal.horizon,
        train_end_date=args.train_end,
        validation_end_date=args.validation_end,
    ).build(returns, market)
    baselines = Phase2ABaselines(config, splits)
    evaluator = PathForecastEvaluator(splits.latent_metadata["n_mean_factors"])
    n_paths = args.paths or config.temporal.ensemble_paths
    return_scale = (
        splits.train_returns[baselines.stocks]
        .std(ddof=0)
        .clip(lower=1e-6)
        .to_numpy()
    )

    phase2f_path = Path(args.phase2f_report)
    phase2f = json.loads(phase2f_path.read_text(encoding="utf-8"))
    models = []
    for seed_report in phase2f["seed_reports"]:
        model = VARResidualPathDiffusion(
            config,
            state_dim=splits.train_states.shape[1],
            horizon=config.temporal.horizon,
            n_mean_factors=splits.latent_metadata["n_mean_factors"],
        )
        _restore_checkpoint(model, splits, Path(seed_report["checkpoint"]))
        models.append(
            (
                seed_report["seed"],
                seed_report["selected_diffusion_weight"],
                model,
            )
        )

    rows, origins = _origin_rows(config, splits, baselines, returns, n_paths)
    scores = _coalition_scores(
        config, splits, evaluator, models, rows, return_scale, n_paths
    )
    per_origin = _aggregate_by_origin(scores)
    coalition_metrics = _coalition_metrics(per_origin)
    attribution = _summarize_attribution(
        per_origin,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    representation = _representation_drift_diagnostic(
        splits, rows, returns, market, n_folds=args.loading_folds
    )

    phase3a_path = Path(args.phase3a_report)
    phase3a = json.loads(phase3a_path.read_text(encoding="utf-8"))
    incumbent_expected = phase3a["development"]["metrics"]["Independent-GARCH"]
    oracle_expected = phase3a["oracle_state_diagnostic"]["metrics"]
    reproduction = {
        "phase3a_report": str(phase3a_path),
        "phase3a_report_sha256": _sha256(phase3a_path),
        "floating_tolerance": 1e-12,
        "current_forecast_return_metrics_match": _metric_means_match(
            coalition_metrics["current_forecast"], incumbent_expected
        ),
        "oracle_all_states_return_metrics_match": _metric_means_match(
            coalition_metrics["oracle_all_states"], oracle_expected
        ),
    }

    report = {
        "created_at": datetime.now().isoformat(),
        "status": "diagnostic_complete",
        "protocol": {
            "phase": "reconstruction_oracle_attribution",
            "purpose": "pre-2024 diagnostic; no candidate selection",
            "train_end": args.train_end,
            "validation_end": args.validation_end,
            "development_period": (
                f"{origins[0]['target_start']} through {origins[-1]['target_end']}"
            ),
            "confirmation_data_loaded": False,
            "consumed_hsi_data_loaded": False,
            "n_assets": len(baselines.stocks),
            "n_origins": len(rows),
            "n_paths_per_seed": n_paths,
            "seeds": [seed for seed, _, _ in models],
            "components": list(COMPONENTS.values()),
            "coalitions": list(COALITION_NAMES.values()),
            "return_metrics": list(RETURN_METRICS),
            "phase2f_report": str(phase2f_path),
            "phase2f_report_sha256": _sha256(phase2f_path),
        },
        "coalition_metrics": coalition_metrics,
        "shapley_attribution": attribution,
        "loading_representation_diagnostic": representation,
        "reproduction_checks": reproduction,
        "origins": origins,
        "decision": {
            "model_modified": False,
            "external_sample_reused": False,
            "candidate_selected": False,
            "next_experiment_must_use_only_pre_2024_data": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("ORACLE-CEILING ATTRIBUTION")
    for name, values in attribution["components"].items():
        print(
            f"  {name}: share={values['share_of_full_oracle_reduction']:.3f}; "
            f"reduction={values['normalized_loss_reduction']:.3f}"
        )
    print(
        "  loading ratios: mean="
        f"{representation['mean_loading_scaled_rmse']['ratio_crossfit_vs_fixed']:.3f}; "
        "vol="
        f"{representation['volatility_loading_logvariance_rmse']['ratio_crossfit_vs_fixed']:.3f}"
    )
    print(f"Saved {output}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_sp500_selection.yaml")
    parser.add_argument(
        "--phase2f-report",
        default="research_output/sp500/phase2f_development.json",
    )
    parser.add_argument(
        "--phase3a-report",
        default="research_output/sp500/phase3a_reconstruction.json",
    )
    parser.add_argument(
        "--output",
        default="research_output/sp500/oracle_ceiling_attribution.json",
    )
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--validation-end", default="2021-12-31")
    parser.add_argument("--paths", type=int, default=None)
    parser.add_argument("--max-stocks", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260815)
    parser.add_argument("--loading-folds", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
