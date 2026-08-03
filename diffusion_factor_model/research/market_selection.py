"""Matched pre-2024 CSI 300 versus S&P 500 development comparison."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

from config.config import load_config
from data.loader import DataPipeline
from data.yf_loader import YFinanceDataPipeline
from diffusion.conditional_path import VARResidualPathDiffusion
from phase2c_benchmark import (
    _checkpoint_payload,
    _evaluate_diffusion,
    _restore_checkpoint,
    _score_baselines,
)
from sequences.baselines import Phase2ABaselines
from sequences.dataset import FactorStateSequenceBuilder
from sequences.evaluation import PathForecastEvaluator


MODEL_NAME = "Phase2D-PathSelected-Diffusion"
PRIMARY_SELECTION_METRICS = ("state_energy_score", "state_rmse")


def _pipeline(config):
    if config.data.source == "akshare":
        return DataPipeline(config)
    if config.data.source == "yfinance":
        return YFinanceDataPipeline(config)
    raise ValueError(
        "Market selection requires source='akshare' or source='yfinance'."
    )


def _bootstrap_difference(diffusion, baseline, seed=20260730, draws=10000):
    """Paired origin bootstrap for a lower-is-better loss difference."""
    diffusion = np.asarray(diffusion, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if diffusion.shape != baseline.shape or diffusion.ndim != 1:
        raise ValueError("Paired score vectors must be one-dimensional and aligned.")
    difference = diffusion - baseline
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(draws, len(difference)))
    bootstrap_means = difference[indices].mean(axis=1)
    return {
        "mean_difference": float(difference.mean()),
        "ci_95": [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ],
        "one_sided_p_diffusion_not_better": float(
            np.mean(bootstrap_means >= 0.0)
        ),
        "origin_win_rate": float(np.mean(difference < 0.0)),
        "n_origins": len(difference),
    }


def _aggregate_seed_scores(seed_scores):
    """Average diffusion losses across seeds separately at each origin."""
    n_origins = len(seed_scores[0])
    keys = seed_scores[0][0]
    return {
        key: [
            float(np.mean([scores[origin][key] for scores in seed_scores]))
            for origin in range(n_origins)
        ]
        for key in keys
    }


def run(args):
    config = load_config(args.config)
    if args.residual_scale_grid is not None:
        config.temporal.residual_scale_grid = args.residual_scale_grid
    if pd_timestamp(config.data.end_date) >= pd_timestamp("2024-01-02"):
        raise ValueError(
            "Market selection config must end before 2024; confirmation data "
            "cannot be used to choose the market."
        )
    returns, market = _pipeline(config).load_all_data(
        max_stocks=args.max_stocks
    )
    builder = FactorStateSequenceBuilder(
        config,
        context_length=config.temporal.context_length,
        horizon=config.temporal.horizon,
        evaluation_stride=config.temporal.horizon,
        train_end_date=args.train_end,
        validation_end_date=args.validation_end,
    )
    splits = builder.build(returns, market)
    baselines = Phase2ABaselines(config, splits)
    evaluator = PathForecastEvaluator(
        splits.latent_metadata["n_mean_factors"]
    )
    n_paths = args.paths or config.temporal.ensemble_paths
    n_origins = len(splits.test)
    if args.max_origins is not None:
        n_origins = min(n_origins, args.max_origins)
    baseline_scores, evaluation_rows, origins = _score_baselines(
        config, splits, baselines, evaluator, n_paths, n_origins
    )

    seeds = args.seeds or config.temporal.validation_seeds
    diffusion_seed_scores = []
    seed_reports = []
    output = Path(args.output)
    checkpoint_dir = output.parent / "checkpoints"
    previous_report = None
    if args.reuse_checkpoints:
        if not output.exists():
            raise FileNotFoundError(
                "--reuse-checkpoints requires an existing selection report."
            )
        previous_report = json.loads(output.read_text(encoding="utf-8"))
    for seed in seeds:
        model = VARResidualPathDiffusion(
            config,
            state_dim=splits.train_states.shape[1],
            horizon=config.temporal.horizon,
            n_mean_factors=splits.latent_metadata["n_mean_factors"],
        )
        checkpoint = checkpoint_dir / f"selection_seed{seed}.pt"
        if args.reuse_checkpoints:
            print(f"Recalibrating market-selection seed {seed}")
            _restore_checkpoint(model, splits, checkpoint)
            model.calibrate_residual_scale(
                splits.validation.context,
                splits.validation.target,
                seed=seed + 30000,
                grid=config.temporal.residual_scale_grid,
                n_paths=config.temporal.path_validation_paths,
                sampling_steps=config.temporal.path_validation_steps,
            )
            history = next(
                row["training_history"]
                for row in previous_report["seed_reports"]
                if row["seed"] == seed
            )
        else:
            print(f"Training market-selection seed {seed}")
            history = model.fit(
                splits.train_states.values,
                splits.train.context,
                splits.train.target,
                splits.validation.context,
                splits.validation.target,
                seed=seed,
                training_steps=args.steps,
                selection_metric="sampled_path_energy",
            )
        seed_scores = _evaluate_diffusion(
            model,
            seed,
            splits,
            baselines,
            evaluator,
            evaluation_rows,
            n_paths,
        )
        diffusion_seed_scores.append(seed_scores)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(_checkpoint_payload(model, splits, seed), checkpoint)
        seed_reports.append(
            {
                "seed": seed,
                "best_step": model.best_step_,
                "best_validation_loss": model.best_validation_loss_,
                "best_selection_score": model.best_selection_score_,
                "residual_calibration": model.calibration_,
                "metrics": evaluator.aggregate(
                    {MODEL_NAME: seed_scores}
                )[MODEL_NAME],
                "training_history": history,
                "checkpoint": str(checkpoint),
            }
        )

    all_scores = {
        **baseline_scores,
        MODEL_NAME: [
            score
            for seed_scores in diffusion_seed_scores
            for score in seed_scores
        ],
    }
    metrics = evaluator.aggregate(all_scores)
    diffusion_by_origin = _aggregate_seed_scores(diffusion_seed_scores)
    paired_tests = {
        baseline: {
            metric: _bootstrap_difference(
                diffusion_by_origin[metric],
                [row[metric] for row in baseline_scores[baseline]],
                seed=20260730 + index,
            )
            for index, metric in enumerate(diffusion_by_origin)
        }
        for baseline in baselines.METHODS
    }
    ratios = {
        metric: (
            metrics[MODEL_NAME][metric]["mean"]
            / metrics["VAR-GARCH"][metric]["mean"]
        )
        for metric in PRIMARY_SELECTION_METRICS
    }
    selection_score = float(
        np.exp(np.mean(np.log(list(ratios.values()))))
    )

    manifest = Path(config.data.universe_manifest)
    report = {
        "timestamp": datetime.now().isoformat(),
        "protocol": {
            "purpose": "pre-2024 market selection; not confirmation testing",
            "data_source": config.data.source,
            "universe": config.data.universe,
            "universe_manifest": str(manifest),
            "universe_manifest_sha256": hashlib.sha256(
                manifest.read_bytes()
            ).hexdigest(),
            "universe_snapshot_date": config.data.universe_snapshot_date,
            "data_start": str(returns.index.min().date()),
            "data_end": str(returns.index.max().date()),
            "train_end": args.train_end,
            "checkpoint_validation_end": args.validation_end,
            "market_selection_start": str(
                splits.test_returns.index.min().date()
            ),
            "market_selection_end": str(
                splits.test_returns.index.max().date()
            ),
            "n_assets": len(baselines.stocks),
            "n_origins": n_origins,
            "n_paths_per_seed": n_paths,
            "seeds": seeds,
            "training_steps_requested": (
                args.steps or config.temporal.training_steps
            ),
            "eligibility_end_date": config.data.eligibility_end_date,
            "preprocess_fit_end_date": (
                config.data.preprocess_fit_end_date
            ),
            "confirmation_data_loaded": False,
            "checkpoints_reused_for_calibration": args.reuse_checkpoints,
            "selection_rule": (
                "lowest geometric mean of diffusion/VAR-GARCH loss ratios "
                "for state_energy_score and state_rmse"
            ),
            "primary_selection_metrics": list(PRIMARY_SELECTION_METRICS),
        },
        "selection_score": selection_score,
        "primary_ratios_vs_var_garch": ratios,
        "metrics": metrics,
        "paired_origin_bootstrap": paired_tests,
        "seed_reports": seed_reports,
        "origins": origins,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Selection score={selection_score:.6f} "
        f"(lower is better; <1 beats VAR-GARCH)"
    )
    print(f"Saved {output}")
    return report


def pd_timestamp(value):
    # Local import keeps CLI startup errors focused on configuration first.
    import pandas as pd

    return pd.to_datetime(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--validation-end", default="2021-12-31")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--paths", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--max-origins", type=int, default=None)
    parser.add_argument("--max-stocks", type=int, default=None)
    parser.add_argument("--reuse-checkpoints", action="store_true")
    parser.add_argument(
        "--residual-scale-grid", type=float, nargs="+", default=None
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
