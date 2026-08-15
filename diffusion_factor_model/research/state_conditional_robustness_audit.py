"""Locked robustness audit for the selected k=32 innovation reconstruction.

The protocol for this script must be committed and publicly pushed before the
audit is executed.  It never searches over neighbor counts or loads post-2023
external-market observations.
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
from reconstruction.state_conditional_innovations import (
    StateConditionalInnovationModel,
)
from research.market_selection import _aggregate_seed_scores, _bootstrap_difference
from research.phase2f_pooling import _linear_pool
from research.phase3a_reconstruction import (
    INCUMBENT,
    RETURN_SELECTION_METRICS,
    _flatten_seed_scores,
    _geometric_ratio,
    _score,
)
from research.state_conditional_innovation_experiment import _make_rows
from sequences.baselines import Phase2ABaselines
from sequences.dataset import FactorStateSequenceBuilder
from sequences.evaluation import PathForecastEvaluator


CANDIDATE = "State-Conditional-k32"
UNCONDITIONAL = "Unconditional-Full-Vector"
METHODS = (INCUMBENT, CANDIDATE, UNCONDITIONAL)
FIXED_NEIGHBORS = 32
DIRECT_RISK_METRICS = (
    "direct_covariance_frobenius_scaled_error",
    "direct_correlation_rmse",
    "random_portfolio_variance_scaled_mae",
    "portfolio_var_05_pinball",
    "portfolio_var_05_coverage_error",
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_frozen_hashes(protocol, args):
    paths = {
        "audit_script_sha256": Path(__file__),
        "innovation_model_sha256": Path(
            "diffusion_factor_model/reconstruction/state_conditional_innovations.py"
        ),
        "sp500_config_sha256": Path(args.config),
        "phase2f_report_sha256": Path(args.phase2f_report),
    }
    checks = {
        name: _sha256(path) == protocol["frozen_hashes"][name]
        for name, path in paths.items()
    }
    checks["selection_artifact_sha256"] = (
        _sha256(args.candidate_report)
        == protocol["candidate"]["selection_artifact_sha256"]
    )
    if not all(checks.values()):
        raise ValueError(f"Frozen robustness hash mismatch: {checks}")
    return checks


def _covariance_matrix(values):
    centered = values - values.mean(axis=0, keepdims=True)
    return centered.T @ centered / max(len(values) - 1, 1)


def _correlation_matrix(covariance):
    scale = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
    return covariance / np.maximum(scale[:, None] * scale[None, :], 1e-12)


def _direct_covariance_scores(return_paths, target_returns, portfolio_weights):
    """Score an ensemble forecast of the horizon realized covariance matrix."""
    predicted = np.mean(
        [_covariance_matrix(path) for path in return_paths], axis=0
    )
    realized = _covariance_matrix(target_returns)
    frobenius_scale = max(np.linalg.norm(realized, ord="fro"), 1e-12)
    predicted_correlation = _correlation_matrix(predicted)
    realized_correlation = _correlation_matrix(realized)
    upper = np.triu_indices_from(realized_correlation, k=1)
    predicted_variances = np.einsum(
        "wi,ij,wj->w", portfolio_weights, predicted, portfolio_weights
    )
    realized_variances = np.einsum(
        "wi,ij,wj->w", portfolio_weights, realized, portfolio_weights
    )
    portfolio_scale = max(np.mean(np.abs(realized_variances)), 1e-12)
    return {
        "direct_covariance_frobenius_scaled_error": float(
            np.linalg.norm(predicted - realized, ord="fro") / frobenius_scale
        ),
        "direct_correlation_rmse": float(
            np.sqrt(
                np.mean(
                    (
                        predicted_correlation[upper]
                        - realized_correlation[upper]
                    )
                    ** 2
                )
            )
        ),
        "random_portfolio_variance_scaled_mae": float(
            np.mean(np.abs(predicted_variances - realized_variances))
            / portfolio_scale
        ),
    }


def _score_paths(
    evaluator,
    states,
    returns,
    target_states,
    target_returns,
    return_scale,
    portfolio_weights,
):
    scores = _score(
        evaluator,
        states,
        returns,
        target_states,
        target_returns,
        return_scale,
    )
    scores.update(
        _direct_covariance_scores(returns, target_returns, portfolio_weights)
    )
    return scores


def _state_regime_signal(splits, states):
    _, volatility = splits.parametrizer.inverse_transform(
        np.asarray(states), splits.latent_metadata
    )
    loadings = splits.reconstructor.vol_loadings.loc[
        splits.reconstructor.stocks
    ].to_numpy()
    center = splits.reconstructor.log_variance_mean.reindex(
        splits.reconstructor.stocks
    ).to_numpy()
    log_variance = volatility @ loadings.T + center
    bounds = splits.reconstructor.log_variance_bounds
    if bounds is not None:
        lower = bounds.loc[splits.reconstructor.stocks, "lower"].to_numpy()
        upper = bounds.loc[splits.reconstructor.stocks, "upper"].to_numpy()
        log_variance = np.clip(log_variance, lower, upper)
    return np.mean(log_variance, axis=1)


def _score_configuration(
    splits,
    baselines,
    evaluator,
    models,
    conditional,
    rows,
    return_scale,
    portfolio_weights,
    n_paths,
    sample_offset,
):
    seed_scores = {method: [] for method in METHODS}
    stocks = splits.reconstructor.stocks
    for seed, weight, model in models:
        per_method = {method: [] for method in METHODS}
        for origin, row in enumerate(rows):
            diffusion = model.sample(
                row["context"],
                n_paths=n_paths,
                seed=seed * 10000 + sample_offset + origin * n_paths,
            )
            states = _linear_pool(diffusion, row["gaussian_var_states"], weight)
            innovation_paths = {
                INCUMBENT: row["incumbent_innovations"],
                CANDIDATE: conditional.sample(
                    states,
                    neighbors=FIXED_NEIGHBORS,
                    seed=seed * 100000 + sample_offset + origin * n_paths,
                ),
                UNCONDITIONAL: conditional.sample_unconditional(
                    n_paths,
                    states.shape[1],
                    seed=(
                        seed * 100000
                        + sample_offset
                        + origin * n_paths
                        + 50000000
                    ),
                ),
            }
            for method, innovations in innovation_paths.items():
                reconstructed = reconstruct_paths(
                    splits, states, innovations, stocks
                )
                per_method[method].append(
                    _score_paths(
                        evaluator,
                        states,
                        reconstructed,
                        row["target_states"],
                        row["target_returns"],
                        return_scale,
                        portfolio_weights,
                    )
                )
        for method in METHODS:
            seed_scores[method].append(per_method[method])
    return seed_scores


def _ratios(candidate, comparator, metrics):
    return {
        metric: float(
            candidate[metric]["mean"]
            / max(comparator[metric]["mean"], 1e-12)
        )
        for metric in metrics
    }


def _paired(seed_scores, candidate, comparator, seed):
    candidate_scores = _aggregate_seed_scores(seed_scores[candidate])
    comparator_scores = _aggregate_seed_scores(seed_scores[comparator])
    return {
        metric: _bootstrap_difference(
            candidate_scores[metric], comparator_scores[metric], seed=seed + index
        )
        for index, metric in enumerate(candidate_scores)
    }


def _regime_composites(seed_scores, regimes):
    candidate = _aggregate_seed_scores(seed_scores[CANDIDATE])
    incumbent = _aggregate_seed_scores(seed_scores[INCUMBENT])
    output = {}
    for regime in ("low", "high"):
        indices = np.flatnonzero(np.asarray(regimes) == regime)
        candidate_means = {
            metric: {"mean": float(np.mean(np.asarray(candidate[metric])[indices]))}
            for metric in RETURN_SELECTION_METRICS
        }
        incumbent_means = {
            metric: {"mean": float(np.mean(np.asarray(incumbent[metric])[indices]))}
            for metric in RETURN_SELECTION_METRICS
        }
        composite, ratios = _geometric_ratio(
            candidate_means, incumbent_means, RETURN_SELECTION_METRICS
        )
        output[regime] = {
            "n_origins": int(len(indices)),
            "composite_ratio": composite,
            "metric_ratios": ratios,
        }
    return output


def _summarize_configuration(seed_scores, regimes, paired_seed):
    metrics = PathForecastEvaluator.aggregate(_flatten_seed_scores(seed_scores))
    composite_incumbent, return_ratios_incumbent = _geometric_ratio(
        metrics[CANDIDATE], metrics[INCUMBENT], RETURN_SELECTION_METRICS
    )
    composite_unconditional, return_ratios_unconditional = _geometric_ratio(
        metrics[CANDIDATE], metrics[UNCONDITIONAL], RETURN_SELECTION_METRICS
    )
    return {
        "metrics": metrics,
        "candidate_vs_incumbent": {
            "return_composite_ratio": composite_incumbent,
            "return_metric_ratios": return_ratios_incumbent,
            "direct_risk_ratios": _ratios(
                metrics[CANDIDATE], metrics[INCUMBENT], DIRECT_RISK_METRICS
            ),
            "paired_origin_bootstrap": _paired(
                seed_scores, CANDIDATE, INCUMBENT, paired_seed
            ),
        },
        "candidate_vs_unconditional": {
            "return_composite_ratio": composite_unconditional,
            "return_metric_ratios": return_ratios_unconditional,
            "direct_risk_ratios": _ratios(
                metrics[CANDIDATE], metrics[UNCONDITIONAL], DIRECT_RISK_METRICS
            ),
        },
        "regime_composites_vs_incumbent": _regime_composites(
            seed_scores, regimes
        ),
    }


def _survival_gate(configurations):
    path50 = configurations["paths_50_repeat_0"]
    path100 = configurations["paths_100_repeat_0"]
    repeat_composites = [
        configurations[f"paths_20_repeat_{repeat}"]["candidate_vs_incumbent"]
        ["return_composite_ratio"]
        for repeat in range(5)
    ]
    risk = path100["candidate_vs_incumbent"]["direct_risk_ratios"]
    regimes = path100["regime_composites_vs_incumbent"]
    requirements = {
        "composite_below_one_at_50_paths": (
            path50["candidate_vs_incumbent"]["return_composite_ratio"] < 1.0
        ),
        "composite_below_one_at_100_paths": (
            path100["candidate_vs_incumbent"]["return_composite_ratio"] < 1.0
        ),
        "energy_below_one_at_50_paths": (
            path50["candidate_vs_incumbent"]["return_metric_ratios"]
            ["return_energy_score"]
            < 1.0
        ),
        "energy_below_one_at_100_paths": (
            path100["candidate_vs_incumbent"]["return_metric_ratios"]
            ["return_energy_score"]
            < 1.0
        ),
        "paired_energy_p_below_0_10_at_100_paths": (
            path100["candidate_vs_incumbent"]["paired_origin_bootstrap"]
            ["return_energy_score"]["one_sided_p_diffusion_not_better"]
            < 0.10
        ),
        "at_least_four_of_five_20_path_repeats_improve": (
            sum(value < 1.0 for value in repeat_composites) >= 4
        ),
        "median_20_path_repeat_composite_below_one": (
            float(np.median(repeat_composites)) < 1.0
        ),
        "beats_unconditional_composite_at_100_paths": (
            path100["candidate_vs_unconditional"]["return_composite_ratio"] < 1.0
        ),
        "direct_covariance_not_worse_by_over_five_percent": (
            risk["direct_covariance_frobenius_scaled_error"] <= 1.05
        ),
        "random_portfolio_variance_not_worse_by_over_five_percent": (
            risk["random_portfolio_variance_scaled_mae"] <= 1.05
        ),
        "var_pinball_not_worse_by_over_five_percent": (
            risk["portfolio_var_05_pinball"] <= 1.05
        ),
        "no_regime_composite_worse_by_over_two_percent": all(
            values["composite_ratio"] <= 1.02 for values in regimes.values()
        ),
    }
    return {
        "survives": bool(all(requirements.values())),
        "requirements": requirements,
        "twenty_path_repeat_composites": repeat_composites,
        "twenty_path_median_composite": float(np.median(repeat_composites)),
    }


def run(args):
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Robustness output already exists: {output}")
    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["candidate"]["neighbors"] != FIXED_NEIGHBORS:
        raise ValueError("Protocol does not freeze the code-level k=32 candidate.")
    frozen_hash_checks = _verify_frozen_hashes(protocol, args)
    config = load_config(args.config)
    if pd.to_datetime(config.data.end_date) >= pd.Timestamp("2024-01-02"):
        raise ValueError("Robustness audit must end before 2024.")

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
    conditional = StateConditionalInnovationModel().fit(
        splits.train_states,
        splits.innovations_train.reindex(columns=baselines.stocks),
        splits.latent_metadata["n_mean_factors"],
    )
    return_scale = (
        splits.train_returns[baselines.stocks]
        .std(ddof=0)
        .clip(lower=1e-6)
        .to_numpy()
    )
    rng = np.random.default_rng(20260815)
    portfolio_weights = rng.dirichlet(
        np.ones(len(baselines.stocks)), size=128
    )

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

    train_signal = _state_regime_signal(splits, splits.train_states.to_numpy())
    regime_threshold = float(np.median(train_signal))
    origin_signal = _state_regime_signal(
        splits, np.stack([window[-1] for window in splits.test.context])
    )
    regimes = ["high" if value >= regime_threshold else "low" for value in origin_signal]

    configurations = {}
    schedule = [(20, repeat) for repeat in range(5)] + [(50, 0), (100, 0)]
    for n_paths, repeat in schedule:
        repeat_offset = repeat * 100000000
        rows, _ = _make_rows(
            config,
            baselines,
            splits.test,
            splits.test_returns,
            n_paths,
            seed_offset=600000 + repeat_offset,
        )
        seed_scores = _score_configuration(
            splits,
            baselines,
            evaluator,
            models,
            conditional,
            rows,
            return_scale,
            portfolio_weights,
            n_paths,
            sample_offset=700000 + repeat_offset,
        )
        key = f"paths_{n_paths}_repeat_{repeat}"
        configurations[key] = _summarize_configuration(
            seed_scores, regimes, paired_seed=20260815 + n_paths + repeat * 1000
        )
        print(
            f"{key}: composite="
            f"{configurations[key]['candidate_vs_incumbent']['return_composite_ratio']:.6f}; "
            "energy="
            f"{configurations[key]['candidate_vs_incumbent']['return_metric_ratios']['return_energy_score']:.6f}"
        )

    original_path = Path(args.candidate_report)
    original = json.loads(original_path.read_text(encoding="utf-8"))
    original_metrics = original["development"]["metrics"][CANDIDATE]
    repeat0_metrics = configurations["paths_20_repeat_0"]["metrics"][CANDIDATE]
    reproduction = {
        "candidate_report": str(original_path),
        "candidate_report_sha256": _sha256(original_path),
        "floating_tolerance": 1e-12,
        "selected_k_is_32": original["validation"]["selected_candidate"] == CANDIDATE,
        "repeat0_candidate_return_metrics_match": all(
            abs(repeat0_metrics[metric]["mean"] - original_metrics[metric]["mean"])
            <= 1e-12
            for metric in RETURN_SELECTION_METRICS
        ),
    }
    gate = _survival_gate(configurations)
    report = {
        "created_at": datetime.now().isoformat(),
        "status": "robustness_survived" if gate["survives"] else "robustness_failed",
        "protocol": {
            "path": str(protocol_path),
            "sha256": _sha256(protocol_path),
            "public_freeze_required_before_execution": True,
            "confirmation_data_loaded": False,
            "consumed_external_samples_loaded": False,
            "candidate": CANDIDATE,
            "neighbors": FIXED_NEIGHBORS,
            "path_counts": [20, 50, 100],
            "twenty_path_repeats": 5,
            "comparators": [INCUMBENT, UNCONDITIONAL],
            "direct_covariance_estimator": (
                "mean across scenario-specific 20-day sample covariance matrices"
            ),
            "random_long_only_portfolios": 128,
            "regime_threshold": "training-period median reconstructed log variance",
            "frozen_hash_checks": frozen_hash_checks,
        },
        "reproduction_checks": reproduction,
        "regimes": {
            "threshold": regime_threshold,
            "origin_labels": regimes,
            "origin_signal": origin_signal.tolist(),
        },
        "configurations": configurations,
        "survival_gate": gate,
        "decision": {
            "candidate_retained_for_external_preregistration": gate["survives"],
            "candidate_or_k_retuned": False,
            "external_sample_reused": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"ROBUSTNESS SURVIVES={gate['survives']}")
    print(f"Saved {output}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_sp500_selection.yaml")
    parser.add_argument(
        "--phase2f-report", default="research_output/sp500/phase2f_development.json"
    )
    parser.add_argument(
        "--candidate-report",
        default="research_output/sp500/state_conditional_innovations.json",
    )
    parser.add_argument(
        "--protocol",
        default="research_output/sp500/state_conditional_robustness.protocol.json",
    )
    parser.add_argument(
        "--output",
        default="research_output/sp500/state_conditional_robustness.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
