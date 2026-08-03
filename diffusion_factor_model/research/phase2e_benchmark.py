"""Phase 2E: diffuse one-step VAR innovations on the S&P development sample."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

from config.config import load_config
from data.yf_loader import YFinanceDataPipeline
from diffusion.conditional_path import VARInnovationPathDiffusion
from phase2c_benchmark import _evaluate_diffusion, _score_baselines
from research.market_selection import (
    PRIMARY_SELECTION_METRICS,
    _aggregate_seed_scores,
    _bootstrap_difference,
)
from sequences.baselines import Phase2ABaselines
from sequences.dataset import FactorStateSequenceBuilder
from sequences.evaluation import PathForecastEvaluator


MODEL_NAME = "Phase2E-VAR-Innovation-Diffusion"


def _checkpoint_payload(model, splits, seed):
    return {
        "model": MODEL_NAME,
        "state_dict": model.network.state_dict(),
        "state_dim": splits.train_states.shape[1],
        "n_mean_factors": splits.latent_metadata["n_mean_factors"],
        "context_length": model.config.temporal.context_length,
        "horizon": model.horizon,
        "seed": seed,
        "best_step": model.best_step_,
        "best_validation_loss": model.best_validation_loss_,
        "best_validation_components": model.best_validation_components_,
        "selection_metric": model.selection_metric_,
        "best_selection_score": model.best_selection_score_,
        "var_intercept": model.var.intercept_,
        "var_transition": model.var.transition_,
        "var_innovation_cov": model.var.innovation_cov_,
        "var_student_df": model.var.student_df_,
        "innovation_location": model.innovation_location_,
        "innovation_scale": model.innovation_scale_,
    }


def run(args):
    config = load_config(args.config)
    if config.data.source != "yfinance":
        raise ValueError("Phase 2E selected-market run requires yfinance.")
    if pd.to_datetime(config.data.end_date) >= pd.Timestamp("2024-01-02"):
        raise ValueError("Phase 2E development data must end before 2024.")

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
    output = Path(args.output)
    checkpoint_dir = output.parent / "checkpoints"
    diffusion_seed_scores = []
    seed_reports = []
    for seed in seeds:
        print(f"Training Phase 2E seed {seed}")
        model = VARInnovationPathDiffusion(
            config,
            state_dim=splits.train_states.shape[1],
            horizon=config.temporal.horizon,
            n_mean_factors=splits.latent_metadata["n_mean_factors"],
        )
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
        scores = _evaluate_diffusion(
            model,
            seed,
            splits,
            baselines,
            evaluator,
            evaluation_rows,
            n_paths,
        )
        diffusion_seed_scores.append(scores)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_dir / f"phase2e_seed{seed}.pt"
        torch.save(_checkpoint_payload(model, splits, seed), checkpoint)
        seed_reports.append(
            {
                "seed": seed,
                "best_step": model.best_step_,
                "best_validation_loss": model.best_validation_loss_,
                "best_selection_score": model.best_selection_score_,
                "metrics": evaluator.aggregate({MODEL_NAME: scores})[
                    MODEL_NAME
                ],
                "training_history": history,
                "checkpoint": str(checkpoint),
            }
        )

    all_scores = {
        **baseline_scores,
        MODEL_NAME: [
            row for seed_rows in diffusion_seed_scores for row in seed_rows
        ],
    }
    metrics = evaluator.aggregate(all_scores)
    diffusion_by_origin = _aggregate_seed_scores(diffusion_seed_scores)
    paired = {
        baseline: {
            metric: _bootstrap_difference(
                diffusion_by_origin[metric],
                [row[metric] for row in baseline_scores[baseline]],
                seed=20260731 + index,
            )
            for index, metric in enumerate(diffusion_by_origin)
        }
        for baseline in baselines.METHODS
    }
    primary_ratios = {
        metric: (
            metrics[MODEL_NAME][metric]["mean"]
            / metrics["VAR-GARCH"][metric]["mean"]
        )
        for metric in PRIMARY_SELECTION_METRICS
    }
    selection_score = float(
        np.exp(np.mean(np.log(list(primary_ratios.values()))))
    )
    gates = {}
    for metric, diffusion_values in metrics[MODEL_NAME].items():
        baseline, value = min(
            (
                (name, metrics[name][metric]["mean"])
                for name in baselines.METHODS
            ),
            key=lambda item: item[1],
        )
        gates[metric] = {
            "diffusion": diffusion_values["mean"],
            "best_baseline": value,
            "best_baseline_method": baseline,
            "ratio": float(diffusion_values["mean"] / value),
            "passed": bool(diffusion_values["mean"] < value),
        }

    manifest = Path(config.data.universe_manifest)
    report = {
        "timestamp": datetime.now().isoformat(),
        "protocol": {
            "phase": "2E",
            "purpose": "pre-2024 model development; not confirmation",
            "model_target": "normalized one-step VAR innovation path",
            "path_reconstruction": "recursive fitted VAR dynamics",
            "checkpoint_selection": (
                "validation energy after recursive state reconstruction"
            ),
            "train_end": args.train_end,
            "checkpoint_validation_end": args.validation_end,
            "development_start": str(splits.test_returns.index.min().date()),
            "development_end": str(splits.test_returns.index.max().date()),
            "n_assets": len(baselines.stocks),
            "n_origins": n_origins,
            "n_paths_per_seed": n_paths,
            "seeds": seeds,
            "training_steps_requested": (
                args.steps or config.temporal.training_steps
            ),
            "universe_manifest": str(manifest),
            "universe_manifest_sha256": hashlib.sha256(
                manifest.read_bytes()
            ).hexdigest(),
            "confirmation_data_loaded": False,
            "baselines": list(baselines.METHODS),
        },
        "selection_score": selection_score,
        "primary_ratios_vs_var_garch": primary_ratios,
        "metrics": metrics,
        "diffusion_gates": gates,
        "gates_passed": sum(row["passed"] for row in gates.values()),
        "gates_total": len(gates),
        "paired_origin_bootstrap": paired,
        "seed_reports": seed_reports,
        "origins": origins,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Phase 2E selection score={selection_score:.6f}; "
        f"gates={report['gates_passed']}/{report['gates_total']}"
    )
    print(f"Saved {output}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="research_sp500_selection.yaml"
    )
    parser.add_argument(
        "--output",
        default="research_output/sp500/phase2e_development.json",
    )
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--validation-end", default="2021-12-31")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--paths", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--max-origins", type=int, default=None)
    parser.add_argument("--max-stocks", type=int, default=100)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
