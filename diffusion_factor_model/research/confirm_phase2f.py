"""One-time 2024-current confirmation of the frozen Phase 2F protocol."""

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
from phase2c_benchmark import _restore_checkpoint, _score_baselines
from research.freeze_phase2f import _panel_fingerprint
from research.market_selection import (
    PRIMARY_SELECTION_METRICS,
    _aggregate_seed_scores,
    _bootstrap_difference,
)
from research.phase2f_pooling import MODEL_NAME, _linear_pool
from research.robustness import dependence_robust_comparison
from sequences.baselines import Phase2ABaselines
from sequences.dataset import FactorStateSequenceBuilder
from sequences.evaluation import PathForecastEvaluator


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bh_adjust(p_values):
    names = list(p_values)
    values = np.asarray([p_values[name] for name in names], dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    running = 1.0
    for reverse_rank in range(len(values) - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        candidate = values[index] * len(values) / rank
        running = min(running, candidate)
        adjusted[index] = min(running, 1.0)
    return {
        name: float(adjusted[index]) for index, name in enumerate(names)
    }


OBSERVABLE_RISK_METRICS = (
    "portfolio_path_energy_score",
    "portfolio_volatility_error",
    "portfolio_var_05_pinball",
    "portfolio_var_05_coverage_error",
    "covariance_frobenius_scaled_error",
)


def _load_frozen_history_with_current_extension(
    frozen_config, current_config, max_stocks
):
    """Use immutable history and apply its preprocessing to new observations."""
    frozen_pipeline = YFinanceDataPipeline(frozen_config)
    current_pipeline = YFinanceDataPipeline(current_config)
    frozen_raw, frozen_market = frozen_pipeline.prepare_raw_returns(
        frozen_pipeline._download(), max_stocks=max_stocks
    )
    current_raw, current_market = current_pipeline.prepare_raw_returns(
        current_pipeline._download(), max_stocks=max_stocks
    )
    if list(frozen_raw.columns) != list(current_raw.columns):
        missing = sorted(set(frozen_raw.columns) - set(current_raw.columns))
        extra = sorted(set(current_raw.columns) - set(frozen_raw.columns))
        raise ValueError(
            "Current panel cannot reproduce frozen asset columns; "
            f"missing={missing}, extra={extra}."
        )
    cutoff = pd.Timestamp("2024-01-01")
    fit_end = pd.to_datetime(frozen_config.data.preprocess_fit_end_date)
    reference = frozen_raw.loc[frozen_raw.index <= fit_end]
    frozen_returns = frozen_pipeline.winsorize(frozen_raw, reference)
    current_post = current_raw.loc[current_raw.index >= cutoff]
    current_post = frozen_pipeline.winsorize(current_post, reference)
    returns = pd.concat([frozen_returns, current_post]).sort_index()
    returns = returns.loc[~returns.index.duplicated(keep="first")]
    market = pd.concat(
        [
            frozen_market,
            current_market.loc[current_market.index >= cutoff],
        ]
    ).sort_index()
    market = market.loc[~market.index.duplicated(keep="first")]
    index = returns.index.intersection(market.index)
    return returns.loc[index], market.loc[index]


def run(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(
            f"Confirmation output already exists: {output}. "
            "The one-time protocol cannot be overwritten."
        )
    frozen_path = Path(args.frozen_protocol)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen["status"] != "frozen_before_confirmation":
        raise ValueError("Protocol is not in the frozen pre-confirmation state.")

    config = load_config(args.config)
    frozen_config = load_config(args.freeze_config)
    returns, market = _load_frozen_history_with_current_extension(
        frozen_config, config, args.max_stocks
    )
    pre2024_returns = returns.loc[returns.index < "2024-01-01"]
    pre2024_market = market.reindex(pre2024_returns.index)
    actual_fingerprint = _panel_fingerprint(
        pre2024_returns, pre2024_market
    )
    expected_fingerprint = frozen["protocol"][
        "pre2024_panel_fingerprint"
    ]
    if actual_fingerprint != expected_fingerprint:
        raise ValueError(
            "Pre-2024 panel fingerprint changed. Refusing confirmation: "
            f"expected {expected_fingerprint}, got {actual_fingerprint}."
        )
    manifest = Path(config.data.universe_manifest)
    if _sha256(manifest) != frozen["protocol"]["universe_manifest_sha256"]:
        raise ValueError("Universe manifest hash differs from frozen protocol.")

    splits = FactorStateSequenceBuilder(
        config,
        context_length=config.temporal.context_length,
        horizon=config.temporal.horizon,
        evaluation_stride=config.temporal.horizon,
        train_end_date=frozen["protocol"]["train_end"],
        validation_end_date=frozen["protocol"][
            "pool_and_checkpoint_validation_end"
        ],
    ).build(returns, market)
    baselines = Phase2ABaselines(config, splits)
    evaluator = PathForecastEvaluator(
        splits.latent_metadata["n_mean_factors"]
    )
    n_paths = frozen["protocol"]["ensemble_paths_per_seed"]
    baseline_scores, evaluation_rows, origins = _score_baselines(
        config,
        splits,
        baselines,
        evaluator,
        n_paths,
        len(splits.test),
    )

    horizon_values = [5, 10, config.temporal.horizon]
    baseline_horizon_scores = {
        horizon: {method: [] for method in baselines.METHODS}
        for horizon in horizon_values
    }
    for row in evaluation_rows:
        for horizon in horizon_values:
            for method in baselines.METHODS:
                baseline_horizon_scores[horizon][method].append(
                    evaluator.score(
                        row["baseline_state_paths"][method][:, :horizon],
                        row["baseline_return_paths"][method][:, :horizon],
                        row["target_states"][:horizon],
                        row["target_returns"][:horizon],
                    )
                )

    seeds = [row["seed"] for row in frozen["seed_reports"]]
    seed_scores = []
    pooled_horizon_seed_scores = {
        horizon: [] for horizon in horizon_values
    }
    confirmation_seed_reports = []
    for frozen_seed in frozen["seed_reports"]:
        seed = frozen_seed["seed"]
        checkpoint = Path(frozen_seed["checkpoint"])
        if _sha256(checkpoint) != frozen_seed["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch for seed {seed}.")
        model = VARResidualPathDiffusion(
            config,
            state_dim=splits.train_states.shape[1],
            horizon=config.temporal.horizon,
            n_mean_factors=splits.latent_metadata["n_mean_factors"],
        )
        _restore_checkpoint(model, splits, checkpoint)
        weight = frozen_seed["selected_diffusion_weight"]
        scores = []
        horizon_scores = {
            horizon: [] for horizon in horizon_values
        }
        for origin, row in enumerate(evaluation_rows):
            diffusion_paths = model.sample(
                row["context"],
                n_paths=n_paths,
                seed=seed * 10000 + 80000 + origin * n_paths,
            )
            states = _linear_pool(
                diffusion_paths,
                row["baseline_state_paths"]["VAR-GARCH"],
                weight,
            )
            generated_returns = reconstruct_paths(
                splits,
                states,
                row["innovations"],
                baselines.stocks,
            )
            score = evaluator.score(
                states,
                generated_returns,
                row["target_states"],
                row["target_returns"],
            )
            scores.append(score)
            for horizon in horizon_values:
                horizon_scores[horizon].append(
                    evaluator.score(
                        states[:, :horizon],
                        generated_returns[:, :horizon],
                        row["target_states"][:horizon],
                        row["target_returns"][:horizon],
                    )
                )
        seed_scores.append(scores)
        for horizon in horizon_values:
            pooled_horizon_seed_scores[horizon].append(
                horizon_scores[horizon]
            )
        confirmation_seed_reports.append(
            {
                "seed": seed,
                "frozen_diffusion_weight": weight,
                "metrics": evaluator.aggregate({MODEL_NAME: scores})[
                    MODEL_NAME
                ],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": frozen_seed["checkpoint_sha256"],
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
                seed=20260802 + index,
            )
            for index, metric in enumerate(pooled_by_origin)
        }
        for baseline in baselines.METHODS
    }
    dependence_robust = {
        baseline: {
            metric: dependence_robust_comparison(
                pooled_by_origin[metric],
                [row[metric] for row in baseline_scores[baseline]],
                seed=20260803 + baseline_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(PRIMARY_SELECTION_METRICS)
        }
        for baseline_index, baseline in enumerate(
            ("Gaussian-VAR", "Student-t-VAR")
        )
    }
    dependence_robust_observable = {
        baseline: {
            metric: dependence_robust_comparison(
                pooled_by_origin[metric],
                [row[metric] for row in baseline_scores[baseline]],
                seed=20260804 + baseline_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(OBSERVABLE_RISK_METRICS)
        }
        for baseline_index, baseline in enumerate(
            ("Gaussian-VAR", "Student-t-VAR", "Block-Bootstrap")
        )
    }
    ratios = {
        metric: (
            metrics[MODEL_NAME][metric]["mean"]
            / metrics["VAR-GARCH"][metric]["mean"]
        )
        for metric in PRIMARY_SELECTION_METRICS
    }
    composite = float(np.exp(np.mean(np.log(list(ratios.values())))))
    success = bool(
        composite < 1.0 and all(value <= 1.0 for value in ratios.values())
    )

    secondary = frozen["confirmatory_endpoints"]["secondary"]
    secondary_raw_p = {
        metric: paired["VAR-GARCH"][metric][
            "one_sided_p_diffusion_not_better"
        ]
        for metric in secondary
    }
    secondary_adjusted_p = _bh_adjust(secondary_raw_p)

    horizon_robustness = {}
    for horizon in horizon_values:
        pooled_rows = [
            row
            for seed_rows in pooled_horizon_seed_scores[horizon]
            for row in seed_rows
        ]
        combined = {
            **baseline_horizon_scores[horizon],
            MODEL_NAME: pooled_rows,
        }
        horizon_metrics = evaluator.aggregate(combined)
        horizon_robustness[str(horizon)] = {
            method: {
                metric: values[metric]["mean"]
                for metric in PRIMARY_SELECTION_METRICS
            }
            for method, values in horizon_metrics.items()
        }

    report_role = getattr(args, "report_role", "locked_confirmation")
    timestamp_key = (
        "confirmed_at"
        if report_role == "locked_confirmation"
        else "analyzed_at"
    )
    report_status = (
        "confirmatory_success" if success else "confirmatory_failure"
    )
    if report_role == "posthoc_robustness_audit":
        report_status = "posthoc_robustness_audit"
    report = {
        timestamp_key: datetime.now().isoformat(),
        "report_role": report_role,
        "status": report_status,
        "frozen_protocol": str(frozen_path),
        "frozen_protocol_sha256": _sha256(frozen_path),
        "protocol_checks": {
            "pre2024_panel_fingerprint_match": True,
            "universe_manifest_hash_match": True,
            "checkpoint_hashes_match": True,
            "output_was_absent_before_run": True,
            "model_or_weight_retuning_after_2024": False,
        },
        "confirmation_period": {
            "start": str(splits.test_returns.index.min().date()),
            "end": str(splits.test_returns.index.max().date()),
            "n_days": len(splits.test_returns),
            "n_nonoverlapping_origins": len(splits.test),
            "n_assets": len(baselines.stocks),
        },
        "confirmatory_decision": {
            "success": success,
            "rule": frozen["confirmatory_endpoints"]["success_rule"],
            "composite_ratio": composite,
            "co_primary_ratios_vs_var_garch": ratios,
            "co_primary_paired_bootstrap_vs_var_garch": {
                metric: paired["VAR-GARCH"][metric]
                for metric in PRIMARY_SELECTION_METRICS
            },
        },
        "metrics": metrics,
        "paired_origin_bootstrap": paired,
        "dependence_robust_primary_comparisons": dependence_robust,
        "dependence_robust_observable_risk_comparisons": (
            dependence_robust_observable
        ),
        "observable_risk_metric_definitions": {
            "portfolio_path_energy_score": (
                "Energy score of the equal-weight portfolio return path."
            ),
            "portfolio_volatility_error": (
                "Absolute error between mean forecast-path volatility and "
                "realized equal-weight portfolio volatility."
            ),
            "portfolio_var_05_pinball": (
                "Mean 5% quantile pinball loss for daily equal-weight "
                "portfolio returns."
            ),
            "portfolio_var_05_coverage_error": (
                "Absolute difference between empirical 5% VaR breach rate "
                "and 5%."
            ),
            "covariance_frobenius_scaled_error": (
                "Frobenius error of predicted versus realized return "
                "covariance, divided by realized covariance Frobenius norm; "
                "exploratory because each target window has only 20 days."
            ),
        },
        "origin_level_scores": {
            MODEL_NAME: pooled_by_origin,
            **{
                baseline: {
                    metric: [row[metric] for row in rows]
                    for metric in rows[0]
                }
                for baseline, rows in baseline_scores.items()
            },
        },
        "secondary_one_sided_p_raw": secondary_raw_p,
        "secondary_one_sided_p_bh_adjusted": secondary_adjusted_p,
        "horizon_robustness": horizon_robustness,
        "seed_reports": confirmation_seed_reports,
        "origins": origins,
        "seeds": seeds,
    }
    if report_role == "posthoc_robustness_audit":
        locked_result_path = Path(
            getattr(
                args,
                "locked_result_reference",
                "research_output/sp500_confirmation/confirmation.json",
            )
        )
        locked_result = json.loads(
            locked_result_path.read_text(encoding="utf-8")
        )
        locked_decision = locked_result["confirmatory_decision"]
        report["locked_result_reference"] = str(locked_result_path)
        report["reproduction_checks"] = {
            "composite_ratio_exact_match": bool(
                composite == locked_decision["composite_ratio"]
            ),
            "co_primary_ratios_exact_match": bool(
                ratios
                == locked_decision["co_primary_ratios_vs_var_garch"]
            ),
            "locked_decision_unchanged": True,
            "interpretation": (
                "Post-hoc inference only; this audit does not create a new "
                "confirmation sample or alter the locked decision."
            ),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Confirmation status={report['status']}; "
        f"composite={composite:.6f}; origins={len(splits.test)}"
    )
    print(f"Saved immutable confirmation report to {output}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="research_sp500_confirmation.yaml"
    )
    parser.add_argument(
        "--locked-result-reference",
        default="research_output/sp500_confirmation/confirmation.json",
        help="Original immutable result used to verify a post-hoc audit rerun.",
    )
    parser.add_argument(
        "--frozen-protocol",
        default="research_output/sp500_frozen/frozen_protocol.json",
    )
    parser.add_argument(
        "--freeze-config", default="research_sp500_freeze.yaml"
    )
    parser.add_argument(
        "--output",
        default="research_output/sp500_confirmation/confirmation.json",
    )
    parser.add_argument("--max-stocks", type=int, default=100)
    parser.add_argument(
        "--report-role",
        choices=("locked_confirmation", "posthoc_robustness_audit"),
        default="locked_confirmation",
        help=(
            "Label a non-overwriting rerun as a post-hoc robustness audit; "
            "this does not create a new confirmation sample."
        ),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
