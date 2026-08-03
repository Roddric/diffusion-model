"""Phase 3A: gated dependent-innovation return reconstruction.

This is a pre-2024 development experiment. It must never load or score the
locked 2024-current S&P 500 confirmation panel.
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import load_config
from data.yf_loader import YFinanceDataPipeline
from diffusion.conditional_path import VARResidualPathDiffusion
from phase2b_benchmark import reconstruct_paths
from phase2c_benchmark import _restore_checkpoint
from reconstruction.dependent_innovations import DependentInnovationModel
from research.market_selection import (
    _aggregate_seed_scores,
    _bootstrap_difference,
)
from research.phase2f_pooling import MODEL_NAME, _linear_pool
from sequences.baselines import Phase2ABaselines
from sequences.dataset import FactorStateSequenceBuilder
from sequences.evaluation import (
    PathForecastEvaluator,
    _energy_score,
    _variogram_score,
)


INCUMBENT = "Independent-GARCH"
JOINT_STUDENT = "Joint-Student-t"
RETURN_SELECTION_METRICS = (
    "return_energy_score",
    "return_variogram_score",
    "daily_volatility_mae",
    "tail_quantile_error",
    "max_drawdown_error",
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _geometric_ratio(candidate, incumbent, metrics):
    ratios = {
        metric: float(
            candidate[metric]["mean"]
            / max(incumbent[metric]["mean"], 1e-12)
        )
        for metric in metrics
    }
    composite = float(np.exp(np.mean(np.log(list(ratios.values())))))
    return composite, ratios


def _accept_candidate(composite, ratios, energy_p_value):
    """Conservative development gate; this is not a confirmation rule."""
    improvements = sum(value < 1.0 for value in ratios.values())
    accepted = bool(
        composite < 1.0
        and ratios["return_energy_score"] < 1.0
        and improvements >= 3
        and max(ratios.values()) <= 1.05
        and energy_p_value < 0.10
    )
    return {
        "accepted": accepted,
        "requirements": {
            "return_composite_below_one": composite < 1.0,
            "return_energy_below_one": ratios["return_energy_score"] < 1.0,
            "at_least_three_of_five_improve": improvements >= 3,
            "no_metric_worse_by_more_than_five_percent": (
                max(ratios.values()) <= 1.05
            ),
            "paired_energy_one_sided_p_below_0_10": energy_p_value < 0.10,
        },
        "metrics_improved": improvements,
    }


def _score(
    evaluator,
    states,
    returns,
    target_states,
    target_returns,
    return_scale,
):
    scores = evaluator.score(
        states, returns, target_states, target_returns
    )
    standardized_paths = returns / return_scale[None, None, :]
    standardized_target = target_returns / return_scale[None, :]
    scores["return_energy_score"] = _energy_score(
        standardized_paths, standardized_target
    )
    scores["return_variogram_score"] = _variogram_score(
        standardized_paths, standardized_target
    )
    return scores


def _candidate_innovations(
    model, incumbent, n_paths, horizon, block_lengths, seed
):
    candidates = {
        INCUMBENT: incumbent,
        JOINT_STUDENT: model.sample_student_t(
            n_paths, horizon, seed=seed
        ),
    }
    for block_length in block_lengths:
        candidates[f"Joint-Block-{block_length}"] = model.sample_blocks(
            n_paths,
            horizon,
            block_length=block_length,
            seed=seed + block_length * 1000,
        )
    return candidates


def _make_rows(
    config,
    splits,
    baselines,
    windows,
    returns,
    dependent,
    n_paths,
    block_lengths,
    seed_offset,
):
    rows = []
    origins = []
    for origin in range(len(windows)):
        context = windows.context[origin]
        forecast = baselines.forecast(
            context,
            config.temporal.horizon,
            n_paths=n_paths,
            seed=seed_offset + origin * n_paths,
        )
        target_dates = windows.target_dates[origin]
        target_returns = returns.reindex(target_dates)[
            baselines.stocks
        ].values
        innovations = _candidate_innovations(
            dependent,
            forecast.innovations["VAR-GARCH"],
            n_paths,
            config.temporal.horizon,
            block_lengths,
            seed=seed_offset + 500000 + origin * n_paths,
        )
        rows.append(
            {
                "context": context,
                "gaussian_var_states": forecast.states["Gaussian-VAR"],
                "candidate_innovations": innovations,
                "target_states": windows.target[origin],
                "target_returns": target_returns,
            }
        )
        origins.append(
            {
                "context_end": str(
                    np.datetime_as_string(
                        windows.context_dates[origin, -1], unit="D"
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


def _score_rows(
    config,
    splits,
    evaluator,
    models,
    rows,
    return_scale,
    n_paths,
    sample_offset,
):
    candidate_names = list(rows[0]["candidate_innovations"])
    seed_scores = {name: [] for name in candidate_names}
    oracle_scores = []
    for seed, weight, model in models:
        per_candidate = {name: [] for name in candidate_names}
        for origin, row in enumerate(rows):
            diffusion = model.sample(
                row["context"],
                n_paths=n_paths,
                seed=seed * 10000 + sample_offset + origin * n_paths,
            )
            states = _linear_pool(
                diffusion, row["gaussian_var_states"], weight
            )
            for name, innovations in row["candidate_innovations"].items():
                reconstructed = reconstruct_paths(
                    splits, states, innovations, splits.reconstructor.stocks
                )
                per_candidate[name].append(
                    _score(
                        evaluator,
                        states,
                        reconstructed,
                        row["target_states"],
                        row["target_returns"],
                        return_scale,
                    )
                )

            if seed == models[0][0]:
                oracle_states = np.repeat(
                    row["target_states"][None, :, :], n_paths, axis=0
                )
                oracle_returns = reconstruct_paths(
                    splits,
                    oracle_states,
                    row["candidate_innovations"][INCUMBENT],
                    splits.reconstructor.stocks,
                )
                oracle_scores.append(
                    _score(
                        evaluator,
                        oracle_states,
                        oracle_returns,
                        row["target_states"],
                        row["target_returns"],
                        return_scale,
                    )
                )
        for name in candidate_names:
            seed_scores[name].append(per_candidate[name])
    return seed_scores, oracle_scores


def _flatten_seed_scores(seed_scores):
    return {
        name: [row for seed_rows in all_seed_rows for row in seed_rows]
        for name, all_seed_rows in seed_scores.items()
    }


def _paired_selected_vs_incumbent(seed_scores, selected):
    selected_by_origin = _aggregate_seed_scores(seed_scores[selected])
    incumbent_by_origin = _aggregate_seed_scores(seed_scores[INCUMBENT])
    return {
        metric: _bootstrap_difference(
            selected_by_origin[metric],
            incumbent_by_origin[metric],
            seed=20260810 + index,
        )
        for index, metric in enumerate(selected_by_origin)
    }


def run(args):
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Phase 3A output already exists: {output}. "
            "Pass --overwrite only for development reruns."
        )

    config = load_config(args.config)
    if pd.to_datetime(config.data.end_date) >= pd.Timestamp("2024-01-02"):
        raise ValueError(
            "Phase 3A config must end before 2024; confirmation data "
            "cannot be used for reconstruction selection."
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
    evaluator = PathForecastEvaluator(
        splits.latent_metadata["n_mean_factors"]
    )
    dependent = DependentInnovationModel().fit(
        splits.innovations_train.reindex(columns=baselines.stocks)
    )
    n_paths = args.paths or config.temporal.ensemble_paths
    return_scale = (
        splits.train_returns[baselines.stocks]
        .std(ddof=0)
        .clip(lower=1e-6)
        .values
    )

    phase2f_path = Path(args.phase2f_report)
    phase2f = json.loads(phase2f_path.read_text(encoding="utf-8"))
    models = []
    for seed_report in phase2f["seed_reports"]:
        seed = seed_report["seed"]
        model = VARResidualPathDiffusion(
            config,
            state_dim=splits.train_states.shape[1],
            horizon=config.temporal.horizon,
            n_mean_factors=splits.latent_metadata["n_mean_factors"],
        )
        _restore_checkpoint(
            model, splits, Path(seed_report["checkpoint"])
        )
        models.append(
            (
                seed,
                seed_report["selected_diffusion_weight"],
                model,
            )
        )

    block_lengths = sorted(set(args.block_lengths))
    validation_rows, validation_origins = _make_rows(
        config,
        splits,
        baselines,
        splits.validation,
        splits.validation_returns,
        dependent,
        n_paths,
        block_lengths,
        seed_offset=300000,
    )
    validation_seed_scores, _ = _score_rows(
        config,
        splits,
        evaluator,
        models,
        validation_rows,
        return_scale,
        n_paths,
        sample_offset=400000,
    )
    validation_metrics = evaluator.aggregate(
        _flatten_seed_scores(validation_seed_scores)
    )
    validation_composites = {}
    for name, metrics in validation_metrics.items():
        composite, ratios = _geometric_ratio(
            metrics,
            validation_metrics[INCUMBENT],
            RETURN_SELECTION_METRICS,
        )
        validation_composites[name] = {
            "composite_ratio_vs_incumbent": composite,
            "ratios_vs_incumbent": ratios,
        }
    selected = min(
        validation_composites,
        key=lambda name: (
            validation_composites[name][
                "composite_ratio_vs_incumbent"
            ],
            name,
        ),
    )

    development_rows, development_origins = _make_rows(
        config,
        splits,
        baselines,
        splits.test,
        splits.test_returns,
        dependent,
        n_paths,
        block_lengths,
        seed_offset=600000,
    )
    development_seed_scores, oracle_scores = _score_rows(
        config,
        splits,
        evaluator,
        models,
        development_rows,
        return_scale,
        n_paths,
        sample_offset=700000,
    )
    development_metrics = evaluator.aggregate(
        _flatten_seed_scores(development_seed_scores)
    )
    oracle_metrics = evaluator.aggregate(
        {"Oracle-State-GARCH": oracle_scores}
    )["Oracle-State-GARCH"]
    composite, ratios = _geometric_ratio(
        development_metrics[selected],
        development_metrics[INCUMBENT],
        RETURN_SELECTION_METRICS,
    )
    paired = _paired_selected_vs_incumbent(
        development_seed_scores, selected
    )
    acceptance = _accept_candidate(
        composite,
        ratios,
        paired["return_energy_score"][
            "one_sided_p_diffusion_not_better"
        ],
    )
    oracle_composite, oracle_ratios = _geometric_ratio(
        oracle_metrics,
        development_metrics[INCUMBENT],
        RETURN_SELECTION_METRICS,
    )

    report = {
        "created_at": datetime.now().isoformat(),
        "status": (
            "phase3_candidate_accepted"
            if acceptance["accepted"]
            else "phase3_candidate_rejected"
        ),
        "protocol": {
            "phase": "3A",
            "purpose": (
                "pre-2024 return-reconstruction development; "
                "not confirmation"
            ),
            "train_end": args.train_end,
            "candidate_selection_period": (
                f"{validation_origins[0]['target_start']} through "
                f"{validation_origins[-1]['target_end']}"
            ),
            "development_evaluation_period": (
                f"{development_origins[0]['target_start']} through "
                f"{development_origins[-1]['target_end']}"
            ),
            "confirmation_data_loaded": False,
            "n_assets": len(baselines.stocks),
            "n_validation_origins": len(validation_rows),
            "n_development_origins": len(development_rows),
            "n_paths_per_seed": n_paths,
            "seeds": [seed for seed, _, _ in models],
            "candidate_set": list(validation_metrics),
            "selection_metrics": list(RETURN_SELECTION_METRICS),
            "phase2f_report": str(phase2f_path),
            "phase2f_report_sha256": _sha256(phase2f_path),
        },
        "innovation_model": {
            "estimated_student_df": dependent.student_df_,
            "block_lengths": block_lengths,
            "training_rows": len(dependent.values_),
        },
        "validation": {
            "selected_candidate": selected,
            "metrics": validation_metrics,
            "selection_composites": validation_composites,
            "origins": validation_origins,
        },
        "development": {
            "selected_candidate": selected,
            "return_composite_ratio_vs_incumbent": composite,
            "return_ratios_vs_incumbent": ratios,
            "acceptance_gate": acceptance,
            "metrics": development_metrics,
            "paired_origin_bootstrap_selected_vs_incumbent": paired,
            "origins": development_origins,
        },
        "oracle_state_diagnostic": {
            "description": (
                "True future factor states with incumbent GARCH innovations; "
                "diagnostic only and never a feasible forecast."
            ),
            "metrics": oracle_metrics,
            "return_composite_ratio_vs_phase2f_incumbent": oracle_composite,
            "return_ratios_vs_phase2f_incumbent": oracle_ratios,
        },
        "application_decision": {
            "apply_as_phase3_candidate": acceptance["accepted"],
            "selected_reconstruction": (
                selected if acceptance["accepted"] else INCUMBENT
            ),
            "phase2f_frozen_model_modified": False,
            "requires_new_external_confirmation": acceptance["accepted"],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Phase 3A selected={selected}; development composite={composite:.6f}; "
        f"accepted={acceptance['accepted']}"
    )
    print(f"Saved {output}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="research_sp500_selection.yaml"
    )
    parser.add_argument(
        "--phase2f-report",
        default="research_output/sp500/phase2f_development.json",
    )
    parser.add_argument(
        "--output",
        default="research_output/sp500/phase3a_reconstruction.json",
    )
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--validation-end", default="2021-12-31")
    parser.add_argument("--paths", type=int, default=None)
    parser.add_argument("--max-stocks", type=int, default=100)
    parser.add_argument(
        "--block-lengths", type=int, nargs="+", default=[1, 5, 20]
    )
    parser.add_argument("--overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

