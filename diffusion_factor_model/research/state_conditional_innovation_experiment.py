"""Gated pre-2024 state-conditional innovation reconstruction experiment."""

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
from reconstruction.state_conditional_innovations import (
    StateConditionalInnovationModel,
)
from research.market_selection import _aggregate_seed_scores, _bootstrap_difference
from research.phase2f_pooling import _linear_pool
from research.phase3a_reconstruction import (
    INCUMBENT,
    RETURN_SELECTION_METRICS,
    _accept_candidate,
    _flatten_seed_scores,
    _geometric_ratio,
    _score,
)
from sequences.baselines import Phase2ABaselines
from sequences.dataset import FactorStateSequenceBuilder
from sequences.evaluation import PathForecastEvaluator


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _candidate_name(neighbors):
    return f"State-Conditional-k{neighbors}"


def _make_rows(config, baselines, windows, returns, n_paths, seed_offset):
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
        rows.append(
            {
                "context": context,
                "gaussian_var_states": forecast.states["Gaussian-VAR"],
                "incumbent_innovations": forecast.innovations["VAR-GARCH"],
                "target_states": windows.target[origin],
                "target_returns": returns.reindex(target_dates)[
                    baselines.stocks
                ].to_numpy(),
            }
        )
        origins.append(
            {
                "context_end": str(
                    np.datetime_as_string(windows.context_dates[origin, -1], unit="D")
                ),
                "target_start": str(np.datetime_as_string(target_dates[0], unit="D")),
                "target_end": str(np.datetime_as_string(target_dates[-1], unit="D")),
            }
        )
    return rows, origins


def _score_rows(
    splits,
    evaluator,
    models,
    rows,
    return_scale,
    conditional,
    neighbor_grid,
    n_paths,
    sample_offset,
):
    names = [INCUMBENT] + [_candidate_name(k) for k in neighbor_grid]
    seed_scores = {name: [] for name in names}
    stocks = splits.reconstructor.stocks
    for seed, weight, model in models:
        per_candidate = {name: [] for name in names}
        for origin, row in enumerate(rows):
            diffusion = model.sample(
                row["context"],
                n_paths=n_paths,
                seed=seed * 10000 + sample_offset + origin * n_paths,
            )
            states = _linear_pool(diffusion, row["gaussian_var_states"], weight)
            candidates = {INCUMBENT: row["incumbent_innovations"]}
            for index, neighbors in enumerate(neighbor_grid):
                candidates[_candidate_name(neighbors)] = conditional.sample(
                    states,
                    neighbors=neighbors,
                    seed=(
                        seed * 100000
                        + sample_offset
                        + origin * n_paths
                        + index * 10000000
                    ),
                )
            for name, innovations in candidates.items():
                reconstructed = reconstruct_paths(
                    splits, states, innovations, stocks
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
        for name in names:
            seed_scores[name].append(per_candidate[name])
    return seed_scores


def _paired_candidate_vs_incumbent(seed_scores, selected):
    candidate = _aggregate_seed_scores(seed_scores[selected])
    incumbent = _aggregate_seed_scores(seed_scores[INCUMBENT])
    return {
        metric: _bootstrap_difference(
            candidate[metric], incumbent[metric], seed=20260815 + index
        )
        for index, metric in enumerate(candidate)
    }


def run(args):
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"State-conditional innovation output exists: {output}. "
            "Pass --overwrite only for deterministic development reruns."
        )
    config = load_config(args.config)
    if pd.to_datetime(config.data.end_date) >= pd.Timestamp("2024-01-02"):
        raise ValueError(
            "State-conditional innovation config must end before 2024."
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
    conditional = StateConditionalInnovationModel().fit(
        splits.train_states,
        splits.innovations_train.reindex(columns=baselines.stocks),
        splits.latent_metadata["n_mean_factors"],
    )
    neighbor_grid = sorted(set(args.neighbors))
    if max(neighbor_grid) > len(conditional.innovations_):
        raise ValueError("Neighbor grid exceeds the aligned training row count.")

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

    validation_rows, validation_origins = _make_rows(
        config,
        baselines,
        splits.validation,
        splits.validation_returns,
        n_paths,
        seed_offset=300000,
    )
    validation_scores = _score_rows(
        splits,
        evaluator,
        models,
        validation_rows,
        return_scale,
        conditional,
        neighbor_grid,
        n_paths,
        sample_offset=400000,
    )
    validation_metrics = evaluator.aggregate(_flatten_seed_scores(validation_scores))
    validation_composites = {}
    for name, metrics in validation_metrics.items():
        composite, ratios = _geometric_ratio(
            metrics, validation_metrics[INCUMBENT], RETURN_SELECTION_METRICS
        )
        validation_composites[name] = {
            "composite_ratio_vs_incumbent": composite,
            "ratios_vs_incumbent": ratios,
        }
    eligible = [_candidate_name(k) for k in neighbor_grid]
    selected = min(
        eligible,
        key=lambda name: (
            validation_composites[name]["composite_ratio_vs_incumbent"], name
        ),
    )

    development_rows, development_origins = _make_rows(
        config,
        baselines,
        splits.test,
        splits.test_returns,
        n_paths,
        seed_offset=600000,
    )
    development_scores = _score_rows(
        splits,
        evaluator,
        models,
        development_rows,
        return_scale,
        conditional,
        neighbor_grid,
        n_paths,
        sample_offset=700000,
    )
    development_metrics = evaluator.aggregate(
        _flatten_seed_scores(development_scores)
    )
    composite, ratios = _geometric_ratio(
        development_metrics[selected],
        development_metrics[INCUMBENT],
        RETURN_SELECTION_METRICS,
    )
    paired = _paired_candidate_vs_incumbent(development_scores, selected)
    acceptance = _accept_candidate(
        composite,
        ratios,
        paired["return_energy_score"]["one_sided_p_diffusion_not_better"],
    )

    report = {
        "created_at": datetime.now().isoformat(),
        "status": (
            "state_conditional_candidate_accepted"
            if acceptance["accepted"]
            else "state_conditional_candidate_rejected"
        ),
        "protocol": {
            "phase": "state_conditional_innovation_reconstruction",
            "purpose": "pre-2024 development; not confirmation",
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
            "consumed_hsi_data_loaded": False,
            "n_assets": len(baselines.stocks),
            "n_validation_origins": len(validation_rows),
            "n_development_origins": len(development_rows),
            "n_paths_per_seed": n_paths,
            "seeds": [seed for seed, _, _ in models],
            "neighbor_grid": neighbor_grid,
            "selection_metrics": list(RETURN_SELECTION_METRICS),
            "selection_rule": "minimum 2021 validation five-metric geometric ratio",
            "development_gate": (
                "composite < 1; return energy < 1; at least 3/5 improve; "
                "no metric > 1.05; paired energy p < 0.10"
            ),
            "phase2f_report": str(phase2f_path),
            "phase2f_report_sha256": _sha256(phase2f_path),
        },
        "innovation_model": {
            "conditioning_variables": "training-standardized log-volatility factors",
            "sampler": "uniform k-nearest-neighbor full cross-sectional rows",
            "local_centering": True,
            "aligned_training_rows": len(conditional.innovations_),
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
        "application_decision": {
            "apply_as_reconstruction_candidate": acceptance["accepted"],
            "selected_reconstruction": selected if acceptance["accepted"] else INCUMBENT,
            "phase2f_frozen_model_modified": False,
            "external_sample_reused": False,
            "requires_new_external_confirmation": acceptance["accepted"],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"STATE-CONDITIONAL INNOVATIONS selected={selected}; "
        f"development composite={composite:.6f}; accepted={acceptance['accepted']}"
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
        "--output",
        default="research_output/sp500/state_conditional_innovations.json",
    )
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--validation-end", default="2021-12-31")
    parser.add_argument("--paths", type=int, default=None)
    parser.add_argument("--max-stocks", type=int, default=100)
    parser.add_argument("--neighbors", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
