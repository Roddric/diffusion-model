"""Phase 2F: validation-selected linear pool of VAR and Phase 2D forecasts."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from config.config import load_config
from data.yf_loader import YFinanceDataPipeline
from diffusion.conditional_path import VARResidualPathDiffusion
from phase2b_benchmark import reconstruct_paths
from phase2c_benchmark import _restore_checkpoint, _score_baselines
from research.market_selection import (
    PRIMARY_SELECTION_METRICS,
    _aggregate_seed_scores,
    _bootstrap_difference,
)
from sequences.baselines import Phase2ABaselines
from sequences.dataset import FactorStateSequenceBuilder
from sequences.evaluation import PathForecastEvaluator, _energy_score


MODEL_NAME = "Phase2F-Validation-Pooled-Diffusion"


def _linear_pool(diffusion_paths, var_paths, weight):
    if diffusion_paths.shape != var_paths.shape:
        raise ValueError("Pooled forecast arrays must have identical shapes.")
    n_diffusion = int(round(weight * len(diffusion_paths)))
    return np.concatenate(
        [
            diffusion_paths[:n_diffusion],
            var_paths[n_diffusion:],
        ],
        axis=0,
    )


def _select_weight(model, seed, config, splits, baselines, grid):
    n_paths = config.temporal.path_validation_paths
    rows = {weight: [] for weight in grid}
    for origin in range(len(splits.validation)):
        context = splits.validation.context[origin]
        baseline = baselines.forecast(
            context,
            config.temporal.horizon,
            n_paths=n_paths,
            seed=50000 + origin * n_paths,
        ).states["VAR-GARCH"]
        diffusion = model.sample(
            context,
            n_paths=n_paths,
            seed=seed * 10000 + 60000 + origin * n_paths,
            sampling_steps=config.temporal.path_validation_steps,
        )
        target = splits.validation.target[origin]
        for weight in grid:
            rows[weight].append(
                _energy_score(
                    _linear_pool(diffusion, baseline, weight), target
                )
            )
    scores = {
        weight: float(np.mean(values)) for weight, values in rows.items()
    }
    selected = min(grid, key=lambda weight: (scores[weight], weight))
    return selected, scores


def run(args):
    config = load_config(args.config)
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

    grid = sorted(set(args.pool_weights))
    if not grid or grid[0] < 0 or grid[-1] > 1:
        raise ValueError("Pool weights must lie in [0, 1].")
    seeds = args.seeds or config.temporal.validation_seeds
    seed_scores = []
    seed_reports = []
    for seed in seeds:
        print(f"Pooling saved Phase 2D seed {seed}")
        model = VARResidualPathDiffusion(
            config,
            state_dim=splits.train_states.shape[1],
            horizon=config.temporal.horizon,
            n_mean_factors=splits.latent_metadata["n_mean_factors"],
        )
        checkpoint = Path(args.checkpoint_pattern.format(seed=seed))
        _restore_checkpoint(model, splits, checkpoint)
        weight, validation_scores = _select_weight(
            model, seed, config, splits, baselines, grid
        )

        scores = []
        for origin, row in enumerate(evaluation_rows):
            diffusion_paths = model.sample(
                row["context"],
                n_paths=n_paths,
                seed=seed * 10000 + 70000 + origin * n_paths,
            )
            pooled_states = _linear_pool(
                diffusion_paths,
                row["baseline_state_paths"]["VAR-GARCH"],
                weight,
            )
            pooled_returns = reconstruct_paths(
                splits,
                pooled_states,
                row["innovations"],
                baselines.stocks,
            )
            scores.append(
                evaluator.score(
                    pooled_states,
                    pooled_returns,
                    row["target_states"],
                    row["target_returns"],
                )
            )
        seed_scores.append(scores)
        seed_reports.append(
            {
                "seed": seed,
                "selected_diffusion_weight": weight,
                "validation_energy_by_weight": {
                    str(key): value
                    for key, value in validation_scores.items()
                },
                "metrics": evaluator.aggregate({MODEL_NAME: scores})[
                    MODEL_NAME
                ],
                "checkpoint": str(checkpoint),
            }
        )

    all_scores = {
        **baseline_scores,
        MODEL_NAME: [row for rows in seed_scores for row in rows],
    }
    metrics = evaluator.aggregate(all_scores)
    pooled_by_origin = _aggregate_seed_scores(seed_scores)
    paired = {
        baseline: {
            metric: _bootstrap_difference(
                pooled_by_origin[metric],
                [row[metric] for row in baseline_scores[baseline]],
                seed=20260801 + index,
            )
            for index, metric in enumerate(pooled_by_origin)
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
    selection_score = float(np.exp(np.mean(np.log(list(ratios.values())))))
    gates = {}
    for metric, values in metrics[MODEL_NAME].items():
        baseline, best = min(
            (
                (name, metrics[name][metric]["mean"])
                for name in baselines.METHODS
            ),
            key=lambda item: item[1],
        )
        gates[metric] = {
            "pooled_diffusion": values["mean"],
            "best_baseline": best,
            "best_baseline_method": baseline,
            "ratio": float(values["mean"] / best),
            "passed": bool(values["mean"] < best),
        }

    report = {
        "timestamp": datetime.now().isoformat(),
        "protocol": {
            "phase": "2F",
            "purpose": "pre-2024 model development; not confirmation",
            "pool": "finite linear pool of Phase 2D and Gaussian VAR paths",
            "pool_weight_selected_on": "2021 validation state energy",
            "pool_weight_grid": grid,
            "weight_zero_included": 0.0 in grid,
            "development_period": "2022-01-03 through 2023-12-29",
            "n_assets": len(baselines.stocks),
            "n_origins": n_origins,
            "n_paths_per_seed": n_paths,
            "seeds": seeds,
            "confirmation_data_loaded": False,
            "baselines": list(baselines.METHODS),
        },
        "selection_score": selection_score,
        "primary_ratios_vs_var_garch": ratios,
        "metrics": metrics,
        "diffusion_gates": gates,
        "gates_passed": sum(row["passed"] for row in gates.values()),
        "gates_total": len(gates),
        "paired_origin_bootstrap": paired,
        "seed_reports": seed_reports,
        "origins": origins,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Phase 2F selection score={selection_score:.6f}; "
        f"weights={[row['selected_diffusion_weight'] for row in seed_reports]}; "
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
        default="research_output/sp500/phase2f_development.json",
    )
    parser.add_argument(
        "--checkpoint-pattern",
        default=(
            "research_output/sp500/checkpoints/selection_seed{seed}.pt"
        ),
    )
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--validation-end", default="2021-12-31")
    parser.add_argument("--paths", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--max-origins", type=int, default=None)
    parser.add_argument("--max-stocks", type=int, default=100)
    parser.add_argument(
        "--pool-weights",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()

