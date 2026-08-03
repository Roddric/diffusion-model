"""Phase 4B: shared cross-market residual diffusion, pre-2024 only."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

from config.config import load_config
from data.loader import DataPipeline
from data.yf_loader import YFinanceDataPipeline
from diffusion.conditional_path import VARResidualPathDiffusion
from diffusion.shared_market import SharedMarketResidualDiffusion
from phase2b_benchmark import reconstruct_paths
from phase2c_benchmark import _restore_checkpoint
from research.market_selection import (
    _aggregate_seed_scores,
    _bootstrap_difference,
)
from research.phase2f_pooling import _linear_pool
from research.phase3a_reconstruction import (
    RETURN_SELECTION_METRICS,
    _geometric_ratio,
    _score,
)
from sequences.baselines import Phase2ABaselines
from sequences.dataset import FactorStateSequenceBuilder
from sequences.evaluation import PathForecastEvaluator


SHARED = "Phase4B-Shared-Market-Diffusion-Pool"
SEPARATE = "Separate-Market-Diffusion-Pool"
STUDENT = "Student-t-VAR"
STATE_METRICS = ("state_energy_score", "state_rmse")


def _pipeline(config):
    if config.data.source == "yfinance":
        return YFinanceDataPipeline(config)
    if config.data.source == "akshare":
        return DataPipeline(config)
    raise ValueError("Phase 4B requires yfinance or AKShare data.")


def _build_market(config_path, args):
    config = load_config(config_path)
    if pd.to_datetime(config.data.end_date) >= pd.Timestamp("2024-01-02"):
        raise ValueError("Phase 4B configs must end before 2024.")
    returns, market = _pipeline(config).load_all_data(
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
    return {
        "config": config,
        "returns": returns,
        "market": market,
        "splits": splits,
        "baselines": baselines,
        "evaluator": PathForecastEvaluator(
            splits.latent_metadata["n_mean_factors"]
        ),
        "return_scale": (
            splits.train_returns[baselines.stocks]
            .std(ddof=0)
            .clip(lower=1e-6)
            .values
        ),
    }


def _shared_training_data(markets):
    return {
        name: {
            "train_states": values["splits"].train_states.values,
            "train_context": values["splits"].train.context,
            "train_target": values["splits"].train.target,
            "validation_context": values["splits"].validation.context,
            "validation_target": values["splits"].validation.target,
        }
        for name, values in markets.items()
    }


def _evaluation_rows(values, n_paths, seed_offset):
    config = values["config"]
    splits = values["splits"]
    baselines = values["baselines"]
    rows = []
    origins = []
    for origin in range(len(splits.test)):
        forecast = baselines.forecast(
            splits.test.context[origin],
            config.temporal.horizon,
            n_paths=n_paths,
            seed=seed_offset + origin * n_paths,
        )
        target_dates = splits.test.target_dates[origin]
        rows.append(
            {
                "context": splits.test.context[origin],
                "student_states": forecast.states[STUDENT],
                "student_returns": forecast.returns[STUDENT],
                "innovations": forecast.innovations["VAR-GARCH"],
                "target_states": splits.test.target[origin],
                "target_returns": splits.test_returns.reindex(target_dates)[
                    baselines.stocks
                ].values,
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
    return rows, origins


def _checkpoint_payload(model, seed):
    return {
        "state_dict": model.network.state_dict(),
        "seed": seed,
        "market_names": model.market_names,
        "state_dim": model.state_dim,
        "horizon": model.horizon,
        "context_dim": model.network.context_dim,
        "best_step": model.best_step_,
        "best_validation_loss": model.best_validation_loss_,
        "best_validation_state_energy": model.best_selection_score_,
        "market_window_counts": model.market_window_counts_,
        "market_parameters": {
            market: {
                "var_intercept": model.vars[market].intercept_,
                "var_transition": model.vars[market].transition_,
                "var_innovation_cov": model.vars[market].innovation_cov_,
                "residual_location": model.residual_locations[market],
                "residual_scale": model.residual_scales[market],
            }
            for market in model.market_names
        },
    }


def _load_separate_models(markets, report_paths, seeds):
    reports = {
        name: json.loads(Path(path).read_text(encoding="utf-8"))
        for name, path in report_paths.items()
    }
    output = {name: {} for name in markets}
    for name, values in markets.items():
        seed_rows = {
            int(row["seed"]): row
            for row in reports[name]["seed_reports"]
        }
        for seed in seeds:
            if seed not in seed_rows:
                raise ValueError(f"{name} report lacks seed {seed}.")
            model = VARResidualPathDiffusion(
                values["config"],
                state_dim=values["splits"].train_states.shape[1],
                horizon=values["config"].temporal.horizon,
                n_mean_factors=values["splits"].latent_metadata[
                    "n_mean_factors"
                ],
            )
            _restore_checkpoint(
                model,
                values["splits"],
                Path(seed_rows[seed]["checkpoint"]),
            )
            output[name][seed] = model
    return output


def _score_market(
    name,
    values,
    rows,
    shared_models,
    separate_models,
    weight,
    n_paths,
):
    evaluator = values["evaluator"]
    splits = values["splits"]
    stocks = values["baselines"].stocks
    return_scale = values["return_scale"]
    student_scores = []
    for row in rows:
        student_scores.append(
            _score(
                evaluator,
                row["student_states"],
                row["student_returns"],
                row["target_states"],
                row["target_returns"],
                return_scale,
            )
        )
    shared_seed_scores = []
    separate_seed_scores = []
    for seed, shared in shared_models.items():
        shared_rows = []
        separate_rows = []
        separate = separate_models[seed]
        for origin, row in enumerate(rows):
            shared_diffusion = shared.sample(
                name,
                row["context"],
                n_paths=n_paths,
                seed=seed * 10000 + 1600000 + origin * n_paths,
            )
            separate_diffusion = separate.sample(
                row["context"],
                n_paths=n_paths,
                seed=seed * 10000 + 1700000 + origin * n_paths,
            )
            shared_states = _linear_pool(
                shared_diffusion, row["student_states"], weight
            )
            separate_states = _linear_pool(
                separate_diffusion, row["student_states"], weight
            )
            shared_returns = reconstruct_paths(
                splits, shared_states, row["innovations"], stocks
            )
            separate_returns = reconstruct_paths(
                splits, separate_states, row["innovations"], stocks
            )
            shared_rows.append(
                _score(
                    evaluator,
                    shared_states,
                    shared_returns,
                    row["target_states"],
                    row["target_returns"],
                    return_scale,
                )
            )
            separate_rows.append(
                _score(
                    evaluator,
                    separate_states,
                    separate_returns,
                    row["target_states"],
                    row["target_returns"],
                    return_scale,
                )
            )
        shared_seed_scores.append(shared_rows)
        separate_seed_scores.append(separate_rows)

    all_scores = {
        SHARED: [
            row for seed_rows in shared_seed_scores for row in seed_rows
        ],
        SEPARATE: [
            row for seed_rows in separate_seed_scores for row in seed_rows
        ],
        STUDENT: student_scores,
    }
    metrics = evaluator.aggregate(all_scores)
    comparisons = {}
    for comparison, comparison_scores in (
        (SEPARATE, separate_seed_scores),
        (STUDENT, student_scores),
    ):
        state_composite, state_ratios = _geometric_ratio(
            metrics[SHARED], metrics[comparison], STATE_METRICS
        )
        return_composite, return_ratios = _geometric_ratio(
            metrics[SHARED],
            metrics[comparison],
            RETURN_SELECTION_METRICS,
        )
        shared_by_origin = _aggregate_seed_scores(shared_seed_scores)
        if comparison == SEPARATE:
            other_by_origin = _aggregate_seed_scores(comparison_scores)
        else:
            other_by_origin = {
                metric: [row[metric] for row in comparison_scores]
                for metric in comparison_scores[0]
            }
        paired = {
            metric: _bootstrap_difference(
                shared_by_origin[metric],
                other_by_origin[metric],
                seed=20261110 + index,
            )
            for index, metric in enumerate(shared_by_origin)
        }
        comparisons[comparison] = {
            "state_composite": state_composite,
            "state_ratios": state_ratios,
            "return_composite": return_composite,
            "return_ratios": return_ratios,
            "paired": paired,
        }
    return {
        "metrics": metrics,
        "comparisons": comparisons,
        "n_origins": len(rows),
    }


def _integration_decision(market_reports):
    names = list(market_reports)
    separate = {
        name: market_reports[name]["comparisons"][SEPARATE]
        for name in names
    }
    pooled_ratios = {
        metric: float(
            np.exp(
                np.mean(
                    [
                        np.log(separate[name]["state_ratios"][metric])
                        for name in names
                    ]
                )
            )
        )
        for metric in STATE_METRICS
    }
    pooled_composite = float(
        np.exp(np.mean(np.log(list(pooled_ratios.values()))))
    )
    checks = {
        "pooled_state_improvement_at_least_one_percent": (
            pooled_composite < 0.99
        ),
        "both_pooled_state_primaries_improve": all(
            value < 1.0 for value in pooled_ratios.values()
        ),
        "state_energy_improves_in_every_market": all(
            separate[name]["state_ratios"]["state_energy_score"] < 1.0
            for name in names
        ),
        "at_least_one_market_energy_p_below_0_10": any(
            separate[name]["paired"]["state_energy_score"][
                "one_sided_p_diffusion_not_better"
            ]
            < 0.10
            for name in names
        ),
        "no_market_state_composite_worse_by_one_percent": all(
            separate[name]["state_composite"] <= 1.01 for name in names
        ),
        "no_market_return_composite_worse_by_two_percent": all(
            separate[name]["return_composite"] <= 1.02 for name in names
        ),
        "shared_pool_beats_student_t_state_in_every_market": all(
            market_reports[name]["comparisons"][STUDENT][
                "state_composite"
            ]
            < 1.0
            for name in names
        ),
    }
    return {
        "accepted_for_third_market_freeze": bool(all(checks.values())),
        "checks": checks,
        "pooled_state_composite_vs_separate": pooled_composite,
        "pooled_state_ratios_vs_separate": pooled_ratios,
    }


def run(args):
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Phase 4B output already exists: {output}.")
    markets = {
        "sp500": _build_market(args.sp500_config, args),
        "csi300": _build_market(args.csi300_config, args),
    }
    state_dims = {
        values["splits"].train_states.shape[1] for values in markets.values()
    }
    mean_dims = {
        values["splits"].latent_metadata["n_mean_factors"]
        for values in markets.values()
    }
    if len(state_dims) != 1 or len(mean_dims) != 1:
        raise ValueError("Markets must share the same latent schema dimensions.")
    seeds = args.seeds or markets["sp500"]["config"].temporal.validation_seeds
    shared_models = {}
    seed_reports = []
    checkpoint_dir = output.parent / "phase4b_checkpoints"
    training_data = _shared_training_data(markets)
    for seed in seeds:
        print(f"Training shared Phase 4B seed {seed}")
        model = SharedMarketResidualDiffusion(
            markets["sp500"]["config"],
            market_names=list(markets),
            state_dim=next(iter(state_dims)),
            horizon=markets["sp500"]["config"].temporal.horizon,
            n_mean_factors=next(iter(mean_dims)),
        )
        history = model.fit(
            training_data,
            seed=seed,
            training_steps=args.steps,
            selection_metric="sampled_path_energy",
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_dir / f"shared_seed{seed}.pt"
        torch.save(_checkpoint_payload(model, seed), checkpoint)
        shared_models[seed] = model
        seed_reports.append(
            {
                "seed": seed,
                "best_step": model.best_step_,
                "best_validation_loss": model.best_validation_loss_,
                "best_validation_state_energy": (
                    model.best_selection_score_
                ),
                "checkpoint": str(checkpoint),
                "training_history": history,
                "market_window_counts": model.market_window_counts_,
            }
        )

    separate_models = _load_separate_models(
        markets,
        {
            "sp500": args.sp500_separate_report,
            "csi300": args.csi300_separate_report,
        },
        seeds,
    )
    market_reports = {}
    origins = {}
    for index, (name, values) in enumerate(markets.items()):
        rows, market_origins = _evaluation_rows(
            values,
            args.paths or values["config"].temporal.ensemble_paths,
            seed_offset=1800000 + index * 100000,
        )
        market_reports[name] = _score_market(
            name,
            values,
            rows,
            shared_models,
            separate_models[name],
            args.diffusion_weight,
            args.paths or values["config"].temporal.ensemble_paths,
        )
        market_reports[name].update(
            {
                "n_assets": len(values["baselines"].stocks),
                "data_start": str(values["returns"].index.min().date()),
                "data_end": str(values["returns"].index.max().date()),
            }
        )
        origins[name] = market_origins

    decision = _integration_decision(market_reports)
    report = {
        "created_at": datetime.now().isoformat(),
        "status": (
            "shared_model_accepted_for_third_market_freeze"
            if decision["accepted_for_third_market_freeze"]
            else "shared_model_rejected"
        ),
        "protocol": {
            "phase": "4B",
            "purpose": (
                "pre-2024 shared-market diffusion development; "
                "not confirmation"
            ),
            "confirmation_data_loaded": False,
            "markets": list(markets),
            "market_token": "constant one-hot appended to each context step",
            "market_specific_components": [
                "factor transforms",
                "Student-t VAR",
                "residual location and scale",
                "return reconstruction",
            ],
            "shared_component": "conditional residual-path denoiser",
            "train_end": args.train_end,
            "validation_end": args.validation_end,
            "development_period": "2022 through 2023",
            "training_steps_requested": (
                args.steps
                or markets["sp500"]["config"].temporal.training_steps
            ),
            "seeds": seeds,
            "diffusion_pool_weight": args.diffusion_weight,
            "weight_reselected": False,
        },
        "integration_decision": decision,
        "markets": market_reports,
        "seed_reports": seed_reports,
        "origins": origins,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Phase 4B pooled state composite="
        f"{decision['pooled_state_composite_vs_separate']:.6f}; "
        f"accepted={decision['accepted_for_third_market_freeze']}"
    )
    print(f"Saved {output}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sp500-config", default="research_sp500_selection.yaml"
    )
    parser.add_argument(
        "--csi300-config", default="research_csi300_selection.yaml"
    )
    parser.add_argument(
        "--sp500-separate-report",
        default="research_output/sp500/phase2f_development.json",
    )
    parser.add_argument(
        "--csi300-separate-report",
        default="research_output/csi300/market_selection_top100.json",
    )
    parser.add_argument(
        "--output",
        default="research_output/phase4b_shared_diffusion.json",
    )
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--validation-end", default="2021-12-31")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--paths", type=int, default=None)
    parser.add_argument("--max-stocks", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--diffusion-weight", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

