"""Phase 3B: gate stronger classical components inside the diffusion pool."""

import argparse
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
from research.market_selection import (
    PRIMARY_SELECTION_METRICS,
    _aggregate_seed_scores,
    _bootstrap_difference,
)
from research.phase2f_pooling import MODEL_NAME, _linear_pool
from research.phase3a_reconstruction import (
    RETURN_SELECTION_METRICS,
    _geometric_ratio,
    _score,
)
from sequences.baselines import Phase2ABaselines
from sequences.dataset import FactorStateSequenceBuilder
from sequences.evaluation import PathForecastEvaluator, _energy_score


BASE_METHODS = (
    "Gaussian-VAR",
    "Student-t-VAR",
    "Residual-Bootstrap-VAR",
)
INCUMBENT = MODEL_NAME
CANDIDATE = "Phase3B-Stronger-Base-Diffusion-Pool"


def _make_rows(
    config, splits, baselines, windows, returns, n_paths, seed_offset
):
    rows = []
    origins = []
    for origin in range(len(windows)):
        forecast = baselines.forecast(
            windows.context[origin],
            config.temporal.horizon,
            n_paths=n_paths,
            seed=seed_offset + origin * n_paths,
        )
        target_dates = windows.target_dates[origin]
        rows.append(
            {
                "context": windows.context[origin],
                "base_states": {
                    name: forecast.states[name] for name in BASE_METHODS
                },
                "innovations": forecast.innovations["VAR-GARCH"],
                "target_states": windows.target[origin],
                "target_returns": returns.reindex(target_dates)[
                    baselines.stocks
                ].values,
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


def _validation_selection(
    config, models, rows, grid, n_paths, sample_offset
):
    reports = {}
    for base in BASE_METHODS:
        seed_reports = []
        for seed, _, model in models:
            scores = {weight: [] for weight in grid}
            for origin, row in enumerate(rows):
                diffusion = model.sample(
                    row["context"],
                    n_paths=n_paths,
                    seed=seed * 10000 + sample_offset + origin * n_paths,
                    sampling_steps=config.temporal.path_validation_steps,
                )
                for weight in grid:
                    states = _linear_pool(
                        diffusion, row["base_states"][base], weight
                    )
                    scores[weight].append(
                        _energy_score(states, row["target_states"])
                    )
            means = {
                weight: float(np.mean(values))
                for weight, values in scores.items()
            }
            selected_weight = min(
                grid, key=lambda weight: (means[weight], weight)
            )
            seed_reports.append(
                {
                    "seed": seed,
                    "selected_diffusion_weight": selected_weight,
                    "energy_by_weight": {
                        str(weight): value
                        for weight, value in means.items()
                    },
                }
            )
        reports[base] = {
            "mean_selected_validation_energy": float(
                np.mean(
                    [
                        row["energy_by_weight"][
                            str(row["selected_diffusion_weight"])
                        ]
                        for row in seed_reports
                    ]
                )
            ),
            "seed_reports": seed_reports,
        }
    selected_base = min(
        reports,
        key=lambda name: (
            reports[name]["mean_selected_validation_energy"],
            name,
        ),
    )
    return selected_base, reports


def _score_development(
    splits,
    evaluator,
    models,
    rows,
    selected_base,
    selected_weights,
    return_scale,
    n_paths,
    sample_offset,
):
    incumbent_seed_scores = []
    candidate_seed_scores = []
    for seed, incumbent_weight, model in models:
        incumbent_rows = []
        candidate_rows = []
        candidate_weight = selected_weights[seed]
        for origin, row in enumerate(rows):
            diffusion = model.sample(
                row["context"],
                n_paths=n_paths,
                seed=seed * 10000 + sample_offset + origin * n_paths,
            )
            incumbent_states = _linear_pool(
                diffusion,
                row["base_states"]["Gaussian-VAR"],
                incumbent_weight,
            )
            candidate_states = _linear_pool(
                diffusion,
                row["base_states"][selected_base],
                candidate_weight,
            )
            incumbent_returns = reconstruct_paths(
                splits,
                incumbent_states,
                row["innovations"],
                splits.reconstructor.stocks,
            )
            candidate_returns = reconstruct_paths(
                splits,
                candidate_states,
                row["innovations"],
                splits.reconstructor.stocks,
            )
            incumbent_rows.append(
                _score(
                    evaluator,
                    incumbent_states,
                    incumbent_returns,
                    row["target_states"],
                    row["target_returns"],
                    return_scale,
                )
            )
            candidate_rows.append(
                _score(
                    evaluator,
                    candidate_states,
                    candidate_returns,
                    row["target_states"],
                    row["target_returns"],
                    return_scale,
                )
            )
        incumbent_seed_scores.append(incumbent_rows)
        candidate_seed_scores.append(candidate_rows)
    return incumbent_seed_scores, candidate_seed_scores


def _paired(candidate_seed_scores, incumbent_seed_scores):
    candidate = _aggregate_seed_scores(candidate_seed_scores)
    incumbent = _aggregate_seed_scores(incumbent_seed_scores)
    return {
        metric: _bootstrap_difference(
            candidate[metric],
            incumbent[metric],
            seed=20260820 + index,
        )
        for index, metric in enumerate(candidate)
    }


def _accept(state_composite, state_ratios, paired, return_composite):
    energy_p = paired["state_energy_score"][
        "one_sided_p_diffusion_not_better"
    ]
    accepted = bool(
        state_composite < 1.0
        and all(value <= 1.0 for value in state_ratios.values())
        and energy_p < 0.10
        and return_composite <= 1.02
    )
    return {
        "accepted": accepted,
        "requirements": {
            "state_composite_below_one": state_composite < 1.0,
            "neither_state_primary_worse": all(
                value <= 1.0 for value in state_ratios.values()
            ),
            "paired_energy_one_sided_p_below_0_10": energy_p < 0.10,
            "return_composite_not_worse_by_more_than_two_percent": (
                return_composite <= 1.02
            ),
        },
    }


def run(args):
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Phase 3B output already exists: {output}. "
            "Pass --overwrite only for development reruns."
        )
    config = load_config(args.config)
    if pd.to_datetime(config.data.end_date) >= pd.Timestamp("2024-01-02"):
        raise ValueError("Phase 3B must not load the confirmation period.")
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
    return_scale = (
        splits.train_returns[baselines.stocks]
        .std(ddof=0)
        .clip(lower=1e-6)
        .values
    )

    phase2f = json.loads(Path(args.phase2f_report).read_text(encoding="utf-8"))
    models = []
    for seed_report in phase2f["seed_reports"]:
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
                seed_report["seed"],
                seed_report["selected_diffusion_weight"],
                model,
            )
        )

    validation_rows, validation_origins = _make_rows(
        config,
        splits,
        baselines,
        splits.validation,
        splits.validation_returns,
        n_paths,
        seed_offset=800000,
    )
    grid = sorted(set(args.pool_weights))
    selected_base, validation_reports = _validation_selection(
        config,
        models,
        validation_rows,
        grid,
        n_paths,
        sample_offset=900000,
    )
    selected_weights = {
        row["seed"]: row["selected_diffusion_weight"]
        for row in validation_reports[selected_base]["seed_reports"]
    }

    development_rows, development_origins = _make_rows(
        config,
        splits,
        baselines,
        splits.test,
        splits.test_returns,
        n_paths,
        seed_offset=1000000,
    )
    incumbent_scores, candidate_scores = _score_development(
        splits,
        evaluator,
        models,
        development_rows,
        selected_base,
        selected_weights,
        return_scale,
        n_paths,
        sample_offset=1100000,
    )
    metrics = evaluator.aggregate(
        {
            INCUMBENT: [
                row for seed_rows in incumbent_scores for row in seed_rows
            ],
            CANDIDATE: [
                row for seed_rows in candidate_scores for row in seed_rows
            ],
        }
    )
    state_composite, state_ratios = _geometric_ratio(
        metrics[CANDIDATE], metrics[INCUMBENT], PRIMARY_SELECTION_METRICS
    )
    return_composite, return_ratios = _geometric_ratio(
        metrics[CANDIDATE], metrics[INCUMBENT], RETURN_SELECTION_METRICS
    )
    paired = _paired(candidate_scores, incumbent_scores)
    acceptance = _accept(
        state_composite, state_ratios, paired, return_composite
    )
    report = {
        "created_at": datetime.now().isoformat(),
        "status": (
            "phase3_candidate_accepted"
            if acceptance["accepted"]
            else "phase3_candidate_rejected"
        ),
        "protocol": {
            "phase": "3B",
            "purpose": "pre-2024 stronger state-pool development",
            "confirmation_data_loaded": False,
            "train_end": args.train_end,
            "validation_end": args.validation_end,
            "n_assets": len(baselines.stocks),
            "n_validation_origins": len(validation_rows),
            "n_development_origins": len(development_rows),
            "n_paths_per_seed": n_paths,
            "base_candidates": list(BASE_METHODS),
            "diffusion_weight_grid": grid,
        },
        "validation": {
            "selected_base": selected_base,
            "selected_diffusion_weights": selected_weights,
            "base_reports": validation_reports,
            "origins": validation_origins,
        },
        "development": {
            "state_composite_ratio_vs_phase2f": state_composite,
            "state_ratios_vs_phase2f": state_ratios,
            "return_composite_ratio_vs_phase2f": return_composite,
            "return_ratios_vs_phase2f": return_ratios,
            "acceptance_gate": acceptance,
            "metrics": metrics,
            "paired_origin_bootstrap_candidate_vs_phase2f": paired,
            "origins": development_origins,
        },
        "application_decision": {
            "apply_as_phase3_candidate": acceptance["accepted"],
            "selected_classical_base": (
                selected_base if acceptance["accepted"] else "Gaussian-VAR"
            ),
            "selected_diffusion_weights": (
                selected_weights
                if acceptance["accepted"]
                else {
                    seed: weight for seed, weight, _ in models
                }
            ),
            "phase2f_frozen_model_modified": False,
            "requires_new_external_confirmation": acceptance["accepted"],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Phase 3B base={selected_base}; state composite={state_composite:.6f}; "
        f"return composite={return_composite:.6f}; "
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
        default="research_output/sp500/phase3b_state_pool.json",
    )
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--validation-end", default="2021-12-31")
    parser.add_argument("--paths", type=int, default=None)
    parser.add_argument("--max-stocks", type=int, default=100)
    parser.add_argument(
        "--pool-weights",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument("--overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

