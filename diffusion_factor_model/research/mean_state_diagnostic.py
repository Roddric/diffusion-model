"""Pre-2024 mean-state error decomposition for the Phase 2F pool."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import load_config
from data.yf_loader import YFinanceDataPipeline
from diffusion.conditional_path import VARResidualPathDiffusion
from phase2c_benchmark import _restore_checkpoint
from research.phase2f_pooling import MODEL_NAME, _linear_pool
from sequences.baselines import Phase2ABaselines
from sequences.dataset import FactorStateSequenceBuilder


METHODS = (MODEL_NAME, "Gaussian-VAR", "Student-t-VAR")
HORIZON_BANDS = {"days_1_5": (0, 5), "days_6_10": (5, 10), "days_11_20": (10, 20)}


def _univariate_path_diagnostics(paths, target):
    """Return proper-score and calibration components by horizon and dimension."""
    paths = np.asarray(paths, dtype=float)
    target = np.asarray(target, dtype=float)
    mean = paths.mean(axis=0)
    bias = mean - target
    first = np.mean(np.abs(paths - target[None, :, :]), axis=0)
    pairwise = np.mean(
        np.abs(paths[:, None, :, :] - paths[None, :, :, :]), axis=(0, 1)
    )
    lower = np.quantile(paths, 0.05, axis=0)
    upper = np.quantile(paths, 0.95, axis=0)
    return {
        "squared_error": bias**2,
        "bias": bias,
        "crps": first - 0.5 * pairwise,
        "ensemble_spread": paths.std(axis=0, ddof=0),
        "interval_90_covered": ((target >= lower) & (target <= upper)).astype(float),
        "interval_90_width": upper - lower,
    }


def _aggregate(rows, names):
    output = {}
    for method, method_rows in rows.items():
        stacked = {
            metric: np.stack([row[metric] for row in method_rows])
            for metric in method_rows[0]
        }
        factors = {}
        mse = stacked["squared_error"].mean(axis=(0, 1))
        total_mse = max(float(mse.sum()), 1e-12)
        for index, name in enumerate(names):
            factor = {
                "rmse": float(np.sqrt(mse[index])),
                "mse_share": float(mse[index] / total_mse),
                "mean_bias": float(stacked["bias"][:, :, index].mean()),
                "mean_crps": float(stacked["crps"][:, :, index].mean()),
                "mean_ensemble_spread": float(
                    stacked["ensemble_spread"][:, :, index].mean()
                ),
                "interval_90_coverage": float(
                    stacked["interval_90_covered"][:, :, index].mean()
                ),
                "interval_90_mean_width": float(
                    stacked["interval_90_width"][:, :, index].mean()
                ),
                "horizon_bands": {},
            }
            for band, (start, stop) in HORIZON_BANDS.items():
                band_mse = stacked["squared_error"][:, start:stop, index].mean()
                factor["horizon_bands"][band] = {
                    "rmse": float(np.sqrt(band_mse)),
                    "crps": float(stacked["crps"][:, start:stop, index].mean()),
                    "bias": float(stacked["bias"][:, start:stop, index].mean()),
                    "spread": float(
                        stacked["ensemble_spread"][:, start:stop, index].mean()
                    ),
                }
            factors[name] = factor
        output[method] = {
            "n_scored_ensembles": len(method_rows),
            "mean_factor_rmse": float(np.sqrt(mse.mean())),
            "factors": factors,
        }
    return output


def _ratios(summary, names):
    pooled = summary[MODEL_NAME]["factors"]
    return {
        comparator: {
            name: {
                "rmse_ratio": float(
                    pooled[name]["rmse"]
                    / max(summary[comparator]["factors"][name]["rmse"], 1e-12)
                ),
                "crps_ratio": float(
                    pooled[name]["mean_crps"]
                    / max(
                        summary[comparator]["factors"][name]["mean_crps"], 1e-12
                    )
                ),
            }
            for name in names
        }
        for comparator in ("Gaussian-VAR", "Student-t-VAR")
    }


def _score_period(config, splits, baselines, models, windows, seed_offset, sample_offset, n_paths):
    n_mean = splits.latent_metadata["n_mean_factors"]
    rows = {method: [] for method in METHODS}
    origins = []
    for origin in range(len(windows)):
        context = windows.context[origin]
        forecast = baselines.forecast(
            context,
            config.temporal.horizon,
            n_paths=n_paths,
            seed=seed_offset + origin * n_paths,
        )
        target = windows.target[origin][:, :n_mean]
        rows["Gaussian-VAR"].append(
            _univariate_path_diagnostics(
                forecast.states["Gaussian-VAR"][:, :, :n_mean], target
            )
        )
        rows["Student-t-VAR"].append(
            _univariate_path_diagnostics(
                forecast.states["Student-t-VAR"][:, :, :n_mean], target
            )
        )
        for seed, weight, model in models:
            diffusion = model.sample(
                context,
                n_paths=n_paths,
                seed=seed * 10000 + sample_offset + origin * n_paths,
            )
            pooled = _linear_pool(
                diffusion, forecast.states["Gaussian-VAR"], weight
            )
            rows[MODEL_NAME].append(
                _univariate_path_diagnostics(pooled[:, :, :n_mean], target)
            )
        origins.append(
            {
                "context_end": str(
                    np.datetime_as_string(windows.context_dates[origin, -1], unit="D")
                ),
                "target_start": str(
                    np.datetime_as_string(windows.target_dates[origin, 0], unit="D")
                ),
                "target_end": str(
                    np.datetime_as_string(windows.target_dates[origin, -1], unit="D")
                ),
            }
        )
    names = splits.latent_metadata["mean_names"]
    summary = _aggregate(rows, names)
    return {
        "summary": summary,
        "pooled_ratios": _ratios(summary, names),
        "origins": origins,
    }


def run(args):
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Mean-state diagnostic exists: {output}")
    config = load_config(args.config)
    if pd.to_datetime(config.data.end_date) >= pd.Timestamp("2024-01-02"):
        raise ValueError("Mean-state diagnostic must end before 2024.")
    returns, market = YFinanceDataPipeline(config).load_all_data(max_stocks=100)
    splits = FactorStateSequenceBuilder(
        config,
        context_length=config.temporal.context_length,
        horizon=config.temporal.horizon,
        evaluation_stride=config.temporal.horizon,
        train_end_date="2020-12-31",
        validation_end_date="2021-12-31",
    ).build(returns, market)
    baselines = Phase2ABaselines(config, splits)
    phase2f = json.loads(Path(args.phase2f_report).read_text(encoding="utf-8"))
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
            (seed_report["seed"], seed_report["selected_diffusion_weight"], model)
        )
    n_paths = args.paths or config.temporal.ensemble_paths
    validation = _score_period(
        config,
        splits,
        baselines,
        models,
        splits.validation,
        300000,
        400000,
        n_paths,
    )
    development = _score_period(
        config,
        splits,
        baselines,
        models,
        splits.test,
        600000,
        700000,
        n_paths,
    )
    pooled_factors = development["summary"][MODEL_NAME]["factors"]
    priority = max(pooled_factors, key=lambda name: pooled_factors[name]["mse_share"])
    report = {
        "created_at": datetime.now().isoformat(),
        "status": "diagnostic_complete",
        "protocol": {
            "purpose": "pre-2024 mean-state diagnosis; no candidate selection",
            "train_end": "2020-12-31",
            "validation_end": "2021-12-31",
            "confirmation_data_loaded": False,
            "consumed_external_samples_loaded": False,
            "n_paths_per_seed": n_paths,
            "seeds": [seed for seed, _, _ in models],
            "mean_factor_names": splits.latent_metadata["mean_names"],
        },
        "validation": validation,
        "development": development,
        "priority_factor": {
            "name": priority,
            "mse_share": pooled_factors[priority]["mse_share"],
            "selection_rule": "largest Phase2F development mean-state MSE share",
        },
        "decision": {
            "candidate_selected": False,
            "next_intervention_must_be_prespecified_after_this_diagnostic": True,
            "external_sample_reused": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"MEAN-STATE PRIORITY={priority}; "
        f"MSE share={pooled_factors[priority]['mse_share']:.3f}"
    )
    print(f"Saved {output}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_sp500_selection.yaml")
    parser.add_argument(
        "--phase2f-report", default="research_output/sp500/phase2f_development.json"
    )
    parser.add_argument(
        "--output", default="research_output/sp500/mean_state_diagnostic.json"
    )
    parser.add_argument("--paths", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
