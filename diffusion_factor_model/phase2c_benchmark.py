"""Phase 2C benchmark for split-head, volatility-aware path diffusion."""

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
from phase2b_benchmark import reconstruct_paths
from sequences.baselines import Phase2ABaselines
from sequences.dataset import FactorStateSequenceBuilder
from sequences.evaluation import PathForecastEvaluator


DIFFUSION_NAME = "Phase2C-SplitHead-Diffusion"


def _checkpoint_payload(model, splits, seed):
    return {
        "state_dict": model.network.state_dict(),
        "state_dim": splits.train_states.shape[1],
        "n_mean_factors": splits.latent_metadata["n_mean_factors"],
        "context_length": model.config.temporal.context_length,
        "horizon": model.config.temporal.horizon,
        "split_output_heads": model.network.split_output_heads,
        "mean_loss_weight": model.config.temporal.mean_loss_weight,
        "volatility_loss_weight": (
            model.config.temporal.volatility_loss_weight
        ),
        "seed": seed,
        "best_step": model.best_step_,
        "best_validation_loss": model.best_validation_loss_,
        "best_validation_components": model.best_validation_components_,
        "selection_metric": model.selection_metric_,
        "best_selection_score": model.best_selection_score_,
        "var_intercept": model.var.intercept_,
        "var_transition": model.var.transition_,
        "var_innovation_cov": model.var.innovation_cov_,
        "residual_location": model.residual_location_,
        "residual_scale": model.residual_scale_,
        "residual_multiplier": model.residual_multiplier_,
        "residual_calibration": model.calibration_,
        "latent_metadata": {
            key: value
            for key, value in splits.latent_metadata.items()
            if key != "dates"
        },
    }


def _restore_checkpoint(model, splits, checkpoint):
    payload = torch.load(
        checkpoint, map_location=model.diffusion.device, weights_only=False
    )
    expected = {
        "state_dim": splits.train_states.shape[1],
        "n_mean_factors": splits.latent_metadata["n_mean_factors"],
        "context_length": model.config.temporal.context_length,
        "horizon": model.config.temporal.horizon,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"Checkpoint {key}={payload.get(key)!r} does not match "
                f"current protocol value {value!r}."
            )
    model.network.load_state_dict(payload["state_dict"])
    model.var.intercept_ = payload["var_intercept"]
    model.var.transition_ = payload["var_transition"]
    model.var.innovation_cov_ = payload["var_innovation_cov"]
    model.var.last_state_ = splits.train_states.values[-1].copy()
    model.residual_location_ = payload["residual_location"]
    model.residual_scale_ = payload["residual_scale"]
    model.residual_multiplier_ = payload.get(
        "residual_multiplier",
        np.ones_like(payload["residual_scale"][0]),
    )
    model.calibration_ = payload.get("residual_calibration")
    model.diffusion.best_step_ = payload["best_step"]
    model.diffusion.best_validation_loss_ = payload[
        "best_validation_loss"
    ]
    model.diffusion.best_validation_components_ = payload.get(
        "best_validation_components"
    )
    model.diffusion.selection_metric_ = payload.get(
        "selection_metric", "denoising_loss"
    )
    model.diffusion.best_selection_score_ = payload.get(
        "best_selection_score", payload["best_validation_loss"]
    )
    return payload


def _score_baselines(config, splits, baselines, evaluator, n_paths, n_origins):
    scores = {method: [] for method in baselines.METHODS}
    evaluation_rows = []
    origins = []
    for origin in range(n_origins):
        context = splits.test.context[origin]
        forecast = baselines.forecast(
            context,
            config.temporal.horizon,
            n_paths=n_paths,
            seed=1000 + origin * n_paths,
        )
        target_dates = splits.test.target_dates[origin]
        target_returns = (
            splits.test_returns.reindex(target_dates)[baselines.stocks].values
        )
        target_states = splits.test.target[origin]
        for method in baselines.METHODS:
            scores[method].append(
                evaluator.score(
                    forecast.states[method],
                    forecast.returns[method],
                    target_states,
                    target_returns,
                )
            )
        evaluation_rows.append(
            {
                "context": context,
                "innovations": forecast.innovations["VAR-GARCH"],
                "baseline_state_paths": forecast.states,
                "baseline_return_paths": forecast.returns,
                "target_states": target_states,
                "target_returns": target_returns,
            }
        )
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
    return scores, evaluation_rows, origins


def _evaluate_diffusion(
    model,
    seed,
    splits,
    baselines,
    evaluator,
    evaluation_rows,
    n_paths,
):
    scores = []
    for origin, row in enumerate(evaluation_rows):
        state_paths = model.sample(
            row["context"],
            n_paths=n_paths,
            seed=seed * 10000 + 2000 + origin * n_paths,
        )
        return_paths = reconstruct_paths(
            splits,
            state_paths,
            row["innovations"],
            baselines.stocks,
        )
        scores.append(
            evaluator.score(
                state_paths,
                return_paths,
                row["target_states"],
                row["target_returns"],
            )
        )
    return scores


def _prior_comparison(
    output_path,
    diffusion_metrics,
    previous_filename,
    previous_model_name,
    previous_label,
    current_label,
):
    previous_path = output_path.parent / previous_filename
    if not previous_path.exists():
        return None
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    previous_metrics = previous["metrics"][previous_model_name]
    return {
        metric: {
            previous_label: previous_metrics[metric]["mean"],
            current_label: values["mean"],
            "ratio": float(
                values["mean"]
                / max(previous_metrics[metric]["mean"], 1e-12)
            ),
            "improved": bool(
                values["mean"] < previous_metrics[metric]["mean"]
            ),
        }
        for metric, values in diffusion_metrics.items()
    }


def run_benchmark(
    phase="2C",
    diffusion_name=DIFFUSION_NAME,
    selection_metric="denoising_loss",
    output_default="../prod_output/phase2c_benchmark.json",
    checkpoint_default="../prod_output/phase2c_diffusion_seed{seed}.pt",
    previous_filename="phase2b_benchmark.json",
    previous_model_name="VAR-Residual-Diffusion",
    previous_label="phase2b",
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--paths", type=int, default=None)
    parser.add_argument("--max-origins", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--reuse-checkpoints",
        action="store_true",
        help="Reload existing selected checkpoints and recompute evaluation.",
    )
    parser.add_argument(
        "--output", default=output_default
    )
    parser.add_argument(
        "--checkpoint-pattern",
        default=checkpoint_default,
    )
    args = parser.parse_args()

    config = load_config("../us_config.yaml")
    seeds = args.seeds or config.temporal.validation_seeds
    if not seeds:
        raise ValueError("At least one validation seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Validation seeds must be unique.")
    n_paths = args.paths or config.temporal.ensemble_paths
    output = Path(args.output)
    existing_report = None
    if args.reuse_checkpoints:
        if not output.exists():
            raise FileNotFoundError(
                "Checkpoint reuse requires the existing benchmark report "
                f"at {output}."
            )
        existing_report = json.loads(output.read_text(encoding="utf-8"))
        if existing_report.get("protocol", {}).get("seeds") != seeds:
            raise ValueError(
                "Existing report seeds do not match requested seeds."
            )

    returns, market = YFinanceDataPipeline(config).load_all_data()
    builder = FactorStateSequenceBuilder(
        config,
        context_length=config.temporal.context_length,
        horizon=config.temporal.horizon,
        evaluation_stride=config.temporal.horizon,
    )
    splits = builder.build(returns, market)
    baselines = Phase2ABaselines(config, splits)
    n_mean_factors = splits.latent_metadata["n_mean_factors"]
    evaluator = PathForecastEvaluator(n_mean_factors)
    n_origins = len(splits.test)
    if args.max_origins is not None:
        n_origins = min(n_origins, args.max_origins)

    scores, evaluation_rows, origins = _score_baselines(
        config, splits, baselines, evaluator, n_paths, n_origins
    )
    scores[diffusion_name] = []
    seed_reports = []
    checkpoint_paths = []

    for run_index, seed in enumerate(seeds, start=1):
        print(f"training seed {seed} ({run_index}/{len(seeds)})")
        model = VARResidualPathDiffusion(
            config,
            state_dim=splits.train_states.shape[1],
            horizon=config.temporal.horizon,
            n_mean_factors=n_mean_factors,
        )
        checkpoint = Path(
            args.checkpoint_pattern.format(seed=seed)
        )
        if args.reuse_checkpoints:
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            _restore_checkpoint(model, splits, checkpoint)
            previous_seed_report = next(
                row
                for row in existing_report["seed_reports"]
                if row["seed"] == seed
            )
            history = previous_seed_report["training_history"]
        else:
            history = model.fit(
                splits.train_states.values,
                splits.train.context,
                splits.train.target,
                splits.validation.context,
                splits.validation.target,
                seed=seed,
                training_steps=args.steps,
                selection_metric=selection_metric,
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                _checkpoint_payload(model, splits, seed),
                checkpoint,
            )
        checkpoint_paths.append(str(checkpoint))

        seed_scores = _evaluate_diffusion(
            model,
            seed,
            splits,
            baselines,
            evaluator,
            evaluation_rows,
            n_paths,
        )
        scores[diffusion_name].extend(seed_scores)
        seed_metrics = evaluator.aggregate(
            {diffusion_name: seed_scores}
        )[diffusion_name]
        seed_reports.append(
            {
                "seed": seed,
                "best_training_step": model.best_step_,
                "best_validation_loss": model.best_validation_loss_,
                "best_validation_components": (
                    model.best_validation_components_
                ),
                "selection_metric": model.selection_metric_,
                "best_selection_score": model.best_selection_score_,
                "training_history": history,
                "metrics": seed_metrics,
                "checkpoint": str(checkpoint),
            }
        )

    metrics = evaluator.aggregate(scores)
    diffusion_metrics = metrics[diffusion_name]
    gates = {}
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

    seed_robustness = {
        metric: {
            "mean_across_seeds": float(
                np.mean(
                    [
                        report["metrics"][metric]["mean"]
                        for report in seed_reports
                    ]
                )
            ),
            "std_across_seeds": float(
                np.std(
                    [
                        report["metrics"][metric]["mean"]
                        for report in seed_reports
                    ]
                )
            ),
            "min": float(
                min(
                    report["metrics"][metric]["mean"]
                    for report in seed_reports
                )
            ),
            "max": float(
                max(
                    report["metrics"][metric]["mean"]
                    for report in seed_reports
                )
            ),
        }
        for metric in diffusion_metrics
    }

    report = {
        "timestamp": datetime.now().isoformat(),
        "protocol": {
            "phase": phase,
            "context_length": config.temporal.context_length,
            "horizon": config.temporal.horizon,
            "ensemble_paths_per_seed": n_paths,
            "n_origins": n_origins,
            "n_assets": len(baselines.stocks),
            "state_dim": splits.train_states.shape[1],
            "n_mean_factors": n_mean_factors,
            "training_windows": len(splits.train),
            "validation_windows": len(splits.validation),
            "training_steps_requested_per_seed": (
                args.steps or config.temporal.training_steps
            ),
            "seeds": seeds,
            "split_output_heads": config.temporal.split_output_heads,
            "mean_loss_weight": config.temporal.mean_loss_weight,
            "volatility_loss_weight": (
                config.temporal.volatility_loss_weight
            ),
            "checkpoint_selection_metric": selection_metric,
            "path_validation_paths": (
                config.temporal.path_validation_paths
            ),
            "path_validation_steps": (
                config.temporal.path_validation_steps
            ),
            "train_only_transformations": True,
            "test_used_for_model_or_seed_selection": False,
            "seed_results_aggregated_without_selection": True,
            "shared_garch_innovations_with_var_baseline": True,
            "diffusion_target": "normalized_residual_around_var_mean",
            "deterministic_garch_simulation": True,
            "selected_checkpoints_reused": args.reuse_checkpoints,
        },
        "seed_reports": seed_reports,
        "seed_robustness": seed_robustness,
        "metrics": metrics,
        "diffusion_gates": gates,
        "gates_passed": sum(gate["passed"] for gate in gates.values()),
        "gates_total": len(gates),
        f"{previous_label}_comparison": _prior_comparison(
            output,
            diffusion_metrics,
            previous_filename,
            previous_model_name,
            previous_label,
            f"phase{phase.lower()}",
        ),
        "origins": origins,
        "checkpoints": checkpoint_paths,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nPHASE {phase} SPLIT-HEAD DIFFUSION BENCHMARK")
    print(
        f"{'method':<31}{'energy':>10}{'state RMSE':>13}"
        f"{'logvol':>10}{'vol MAE':>11}{'drawdown':>11}"
    )
    for method, values in metrics.items():
        print(
            f"{method:<31}"
            f"{values['state_energy_score']['mean']:>10.3f}"
            f"{values['state_rmse']['mean']:>13.3f}"
            f"{values['logvol_factor_rmse']['mean']:>10.3f}"
            f"{values['daily_volatility_mae']['mean']:>11.5f}"
            f"{values['max_drawdown_error']['mean']:>11.4f}"
        )
    print(
        f"\nDiffusion passed {report['gates_passed']}/"
        f"{report['gates_total']} best-baseline metric gates."
    )
    print(f"Saved {output}")
    for checkpoint in checkpoint_paths:
        print(f"Saved {checkpoint}")


def main():
    run_benchmark()


if __name__ == "__main__":
    main()
