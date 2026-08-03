"""Matched Phase 2B benchmark for conditional temporal path diffusion."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

from config.config import load_config
from data.yf_loader import YFinanceDataPipeline
from diffusion.conditional_path import VARResidualPathDiffusion
from sequences.baselines import Phase2ABaselines
from sequences.dataset import FactorStateSequenceBuilder
from sequences.evaluation import PathForecastEvaluator


def reconstruct_paths(splits, states, innovations, stocks):
    paths = []
    horizon = states.shape[1]
    for index in range(len(states)):
        residuals = splits.innovations_train.iloc[:horizon].copy()
        residuals.loc[:, :] = innovations[index]
        paths.append(
            splits.reconstructor.reconstruct(
                states[index], residuals, verbose=False
            )[stocks].values
        )
    return np.stack(paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--paths", type=int, default=None)
    parser.add_argument("--max-origins", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", default="../prod_output/phase2b_benchmark.json"
    )
    parser.add_argument(
        "--checkpoint",
        default="../prod_output/conditional_path_diffusion.pt",
    )
    args = parser.parse_args()

    config = load_config("../us_config.yaml")
    n_paths = args.paths or config.temporal.ensemble_paths
    returns, market = YFinanceDataPipeline(config).load_all_data()
    builder = FactorStateSequenceBuilder(
        config,
        context_length=config.temporal.context_length,
        horizon=config.temporal.horizon,
        evaluation_stride=config.temporal.horizon,
    )
    splits = builder.build(returns, market)
    baselines = Phase2ABaselines(config, splits)
    evaluator = PathForecastEvaluator(
        splits.latent_metadata["n_mean_factors"]
    )

    diffusion = VARResidualPathDiffusion(
        config,
        state_dim=splits.train_states.shape[1],
        horizon=config.temporal.horizon,
    )
    history = diffusion.fit(
        splits.train_states.values,
        splits.train.context,
        splits.train.target,
        splits.validation.context,
        splits.validation.target,
        seed=args.seed,
        training_steps=args.steps,
    )
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": diffusion.network.state_dict(),
            "state_dim": splits.train_states.shape[1],
            "context_length": config.temporal.context_length,
            "horizon": config.temporal.horizon,
            "best_step": diffusion.best_step_,
            "best_validation_loss": diffusion.best_validation_loss_,
            "var_intercept": diffusion.var.intercept_,
            "var_transition": diffusion.var.transition_,
            "var_innovation_cov": diffusion.var.innovation_cov_,
            "residual_location": diffusion.residual_location_,
            "residual_scale": diffusion.residual_scale_,
            "latent_metadata": {
                key: value
                for key, value in splits.latent_metadata.items()
                if key != "dates"
            },
        },
        checkpoint,
    )

    diffusion_name = "VAR-Residual-Diffusion"
    methods = list(baselines.METHODS) + [diffusion_name]
    scores = {method: [] for method in methods}
    origins = []
    n_origins = len(splits.test)
    if args.max_origins is not None:
        n_origins = min(n_origins, args.max_origins)

    for origin in range(n_origins):
        context = splits.test.context[origin]
        forecast = baselines.forecast(
            context,
            config.temporal.horizon,
            n_paths=n_paths,
            seed=1000 + origin * n_paths,
        )
        diffusion_states = diffusion.sample(
            context,
            n_paths=n_paths,
            seed=2000 + origin * n_paths,
        )
        diffusion_returns = reconstruct_paths(
            splits,
            diffusion_states,
            forecast.innovations["VAR-GARCH"],
            baselines.stocks,
        )
        forecast.states[diffusion_name] = diffusion_states
        forecast.returns[diffusion_name] = diffusion_returns

        target_dates = splits.test.target_dates[origin]
        target_returns = (
            splits.test_returns.reindex(target_dates)[baselines.stocks].values
        )
        target_states = splits.test.target[origin]
        origins.append(
            {
                "context_end": str(
                    np.datetime_as_string(
                        splits.test.context_dates[origin, -1], unit="D"
                    )
                ),
                "target_start": str(
                    np.datetime_as_string(target_dates[0], unit="D")
                ),
                "target_end": str(
                    np.datetime_as_string(target_dates[-1], unit="D")
                ),
            }
        )
        for method in methods:
            scores[method].append(
                evaluator.score(
                    forecast.states[method],
                    forecast.returns[method],
                    target_states,
                    target_returns,
                )
            )
        print(
            f"origin {origin + 1}/{n_origins}: "
            f"{origins[-1]['target_start']} -> {origins[-1]['target_end']}"
        )

    metrics = evaluator.aggregate(scores)
    gates = {}
    diffusion_metrics = metrics[diffusion_name]
    for metric, values in diffusion_metrics.items():
        best_method, best_value = min(
            (
                (method, metrics[method][metric]["mean"])
                for method in baselines.METHODS
            ),
            key=lambda item: item[1],
        )
        gates[metric] = {
            "diffusion": values["mean"],
            "best_baseline": best_value,
            "best_baseline_method": best_method,
            "ratio": float(values["mean"] / max(best_value, 1e-12)),
            "passed": bool(values["mean"] < best_value),
        }

    report = {
        "timestamp": datetime.now().isoformat(),
        "protocol": {
            "context_length": config.temporal.context_length,
            "horizon": config.temporal.horizon,
            "ensemble_paths": n_paths,
            "n_origins": n_origins,
            "n_assets": len(baselines.stocks),
            "state_dim": splits.train_states.shape[1],
            "training_windows": len(splits.train),
            "validation_windows": len(splits.validation),
            "training_steps_requested": (
                args.steps or config.temporal.training_steps
            ),
            "best_training_step": diffusion.best_step_,
            "best_validation_loss": diffusion.best_validation_loss_,
            "train_only_transformations": True,
            "test_used_for_model_selection": False,
            "shared_garch_innovations_with_var_baseline": True,
            "diffusion_target": "normalized_residual_around_var_mean",
            "seed": args.seed,
        },
        "training_history": history,
        "metrics": metrics,
        "diffusion_gates": gates,
        "gates_passed": sum(gate["passed"] for gate in gates.values()),
        "gates_total": len(gates),
        "origins": origins,
        "checkpoint": str(checkpoint),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nPHASE 2B CONDITIONAL DIFFUSION BENCHMARK")
    print(
        f"{'method':<23}{'energy':>10}{'state RMSE':>13}"
        f"{'vol MAE':>11}{'drawdown':>11}"
    )
    for method, values in metrics.items():
        print(
            f"{method:<23}"
            f"{values['state_energy_score']['mean']:>10.3f}"
            f"{values['state_rmse']['mean']:>13.3f}"
            f"{values['daily_volatility_mae']['mean']:>11.5f}"
            f"{values['max_drawdown_error']['mean']:>11.4f}"
        )
    print(
        f"\nDiffusion passed {report['gates_passed']}/"
        f"{report['gates_total']} best-baseline metric gates."
    )
    print(f"Saved {output}")
    print(f"Saved {checkpoint}")


if __name__ == "__main__":
    main()
