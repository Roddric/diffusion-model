"""Factorwise Phase2F/Student-t routing for mean-state calibration.

This post-diagnostic experiment uses only pre-2024 data.  A mean factor routes
to Student-t VAR only when it improves both validation RMSE and CRPS by at least
the frozen margin.  Volatility factors always retain Phase 2F paths.
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
from research.market_selection import _aggregate_seed_scores, _bootstrap_difference
from research.phase2f_pooling import MODEL_NAME, _linear_pool
from research.phase3a_reconstruction import (
    RETURN_SELECTION_METRICS,
    _geometric_ratio,
    _score,
)
from sequences.baselines import Phase2ABaselines
from sequences.dataset import FactorStateSequenceBuilder
from sequences.evaluation import PathForecastEvaluator


CANDIDATE = "Factorwise-Mean-Routed-Pool"
VALIDATION_MARGIN = 0.02
STATE_METRICS = ("state_energy_score", "state_rmse")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _select_student_factors(diagnostic, margin=VALIDATION_MARGIN):
    ratios = diagnostic["validation"]["pooled_ratios"]["Student-t-VAR"]
    threshold = 1.0 + margin
    return sorted(
        name
        for name, values in ratios.items()
        if values["rmse_ratio"] >= threshold
        and values["crps_ratio"] >= threshold
    )


def _route_dimensions(pooled, student, indices):
    if pooled.shape != student.shape:
        raise ValueError("Pooled and Student-t state paths must have equal shapes.")
    routed = pooled.copy()
    routed[:, :, indices] = student[:, :, indices]
    return routed


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
                "gaussian_states": forecast.states["Gaussian-VAR"],
                "student_states": forecast.states["Student-t-VAR"],
                "innovations": forecast.innovations["VAR-GARCH"],
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
    routed_indices,
    n_paths,
    sample_offset,
):
    incumbent_seed_scores = []
    candidate_seed_scores = []
    stocks = splits.reconstructor.stocks
    for seed, weight, model in models:
        incumbent_rows = []
        candidate_rows = []
        for origin, row in enumerate(rows):
            diffusion = model.sample(
                row["context"],
                n_paths=n_paths,
                seed=seed * 10000 + sample_offset + origin * n_paths,
            )
            incumbent_states = _linear_pool(
                diffusion, row["gaussian_states"], weight
            )
            candidate_states = _route_dimensions(
                incumbent_states, row["student_states"], routed_indices
            )
            incumbent_returns = reconstruct_paths(
                splits, incumbent_states, row["innovations"], stocks
            )
            candidate_returns = reconstruct_paths(
                splits, candidate_states, row["innovations"], stocks
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


def _flatten(seed_scores):
    return [row for rows in seed_scores for row in rows]


def _paired(candidate_seed_scores, incumbent_seed_scores):
    candidate = _aggregate_seed_scores(candidate_seed_scores)
    incumbent = _aggregate_seed_scores(incumbent_seed_scores)
    return {
        metric: _bootstrap_difference(
            candidate[metric], incumbent[metric], seed=20260816 + index
        )
        for index, metric in enumerate(candidate)
    }


def _gate(state_composite, state_ratios, mean_rmse_ratio, return_composite, paired):
    requirements = {
        "mean_factor_rmse_below_one": mean_rmse_ratio < 1.0,
        "state_composite_below_one": state_composite < 1.0,
        "no_state_metric_worse_by_over_half_percent": (
            max(state_ratios.values()) <= 1.005
        ),
        "paired_state_energy_p_below_0_10": (
            paired["state_energy_score"]["one_sided_p_diffusion_not_better"]
            < 0.10
        ),
        "return_composite_not_worse_by_over_two_percent": return_composite <= 1.02,
    }
    return {"accepted": bool(all(requirements.values())), "requirements": requirements}


def run(args):
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Factorwise routing output exists: {output}")
    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["selection_rule"]["minimum_joint_improvement"] != VALIDATION_MARGIN:
        raise ValueError("Protocol margin differs from the code-level frozen margin.")
    diagnostic_path = Path(args.diagnostic)
    if _sha256(diagnostic_path) != protocol["diagnostic_sha256"]:
        raise ValueError("Mean-state diagnostic hash differs from the frozen protocol.")
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))

    config = load_config(args.config)
    if pd.to_datetime(config.data.end_date) >= pd.Timestamp("2024-01-02"):
        raise ValueError("Factorwise routing must end before 2024.")
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
    evaluator = PathForecastEvaluator(splits.latent_metadata["n_mean_factors"])
    selected_factors = _select_student_factors(diagnostic)
    expected = protocol["selection_rule"]["expected_student_t_factors"]
    if selected_factors != expected:
        raise ValueError(
            f"Frozen expected routing differs: {selected_factors} != {expected}"
        )
    names = splits.latent_metadata["mean_names"]
    routed_indices = [names.index(name) for name in selected_factors]

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
    return_scale = (
        splits.train_returns[baselines.stocks]
        .std(ddof=0)
        .clip(lower=1e-6)
        .to_numpy()
    )
    rows, origins = _make_rows(
        config, baselines, splits.test, splits.test_returns, n_paths, 600000
    )
    incumbent_seeds, candidate_seeds = _score_rows(
        splits,
        evaluator,
        models,
        rows,
        return_scale,
        routed_indices,
        n_paths,
        700000,
    )
    metrics = evaluator.aggregate(
        {MODEL_NAME: _flatten(incumbent_seeds), CANDIDATE: _flatten(candidate_seeds)}
    )
    state_composite, state_ratios = _geometric_ratio(
        metrics[CANDIDATE], metrics[MODEL_NAME], STATE_METRICS
    )
    return_composite, return_ratios = _geometric_ratio(
        metrics[CANDIDATE], metrics[MODEL_NAME], RETURN_SELECTION_METRICS
    )
    mean_rmse_ratio = float(
        metrics[CANDIDATE]["mean_factor_rmse"]["mean"]
        / metrics[MODEL_NAME]["mean_factor_rmse"]["mean"]
    )
    paired = _paired(candidate_seeds, incumbent_seeds)
    gate = _gate(
        state_composite, state_ratios, mean_rmse_ratio, return_composite, paired
    )
    report = {
        "created_at": datetime.now().isoformat(),
        "status": "descriptive_gate_passed" if gate["accepted"] else "descriptive_gate_failed",
        "protocol": {
            "path": str(protocol_path),
            "sha256": _sha256(protocol_path),
            "purpose": "post-diagnostic pre-2024 development; not confirmation",
            "confirmation_data_loaded": False,
            "consumed_external_samples_loaded": False,
            "n_origins": len(rows),
            "n_paths_per_seed": n_paths,
            "seeds": [seed for seed, _, _ in models],
        },
        "routing": {
            "student_t_factors": selected_factors,
            "phase2f_factors": [name for name in names if name not in selected_factors],
            "volatility_source": MODEL_NAME,
            "minimum_joint_validation_improvement": VALIDATION_MARGIN,
        },
        "development": {
            "metrics": metrics,
            "mean_factor_rmse_ratio": mean_rmse_ratio,
            "state_composite_ratio": state_composite,
            "state_metric_ratios": state_ratios,
            "return_composite_ratio": return_composite,
            "return_metric_ratios": return_ratios,
            "paired_origin_bootstrap": paired,
            "gate": gate,
            "origins": origins,
        },
        "decision": {
            "candidate_promoted": False,
            "reason": (
                "The 2022-2023 panel informed the preceding diagnostic; this "
                "gate is descriptive even if its numerical requirements pass."
            ),
            "requires_cross_market_pre2024_replication_before_external_freeze": gate[
                "accepted"
            ],
            "external_sample_reused": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"FACTORWISE ROUTING factors={selected_factors}; "
        f"mean RMSE={mean_rmse_ratio:.6f}; state={state_composite:.6f}; "
        f"gate={gate['accepted']}"
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
        "--diagnostic", default="research_output/sp500/mean_state_diagnostic.json"
    )
    parser.add_argument(
        "--protocol", default="research_output/sp500/factorwise_mean_routing.protocol.json"
    )
    parser.add_argument(
        "--output", default="research_output/sp500/factorwise_mean_routing.json"
    )
    parser.add_argument("--paths", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
