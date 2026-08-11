"""One-time 2024-current external evaluation of the frozen FTSE dual pools."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import load_config
from diffusion.conditional_path import VARResidualPathDiffusion
from phase2b_benchmark import reconstruct_paths
from phase2c_benchmark import _restore_checkpoint, _score_baselines
from research.confirm_phase2f import (
    OBSERVABLE_RISK_METRICS,
    _bh_adjust,
    _load_frozen_history_with_current_extension,
    _sha256,
)
from research.freeze_phase2f import _panel_fingerprint
from research.freeze_ftse100_pools import (
    POOL_BASES,
    PRIMARY_BASE,
    SECONDARY_BASE,
)
from research.market_selection import (
    PRIMARY_SELECTION_METRICS,
    _aggregate_seed_scores,
    _bootstrap_difference,
)
from research.phase2f_pooling import _linear_pool
from research.robustness import dependence_robust_comparison
from sequences.baselines import Phase2ABaselines
from sequences.dataset import FactorStateSequenceBuilder
from sequences.evaluation import PathForecastEvaluator


POOL_NAMES = {
    "gaussian_base_pool": "FTSE100-Gaussian-Base-Pool",
    "student_t_base_pool": "FTSE100-Student-t-Base-Pool",
}
PRIMARY_POOL = POOL_NAMES["gaussian_base_pool"]
SECONDARY_POOL = POOL_NAMES["student_t_base_pool"]


def _composite_ratio(metrics, pool_name, baseline_name):
    ratios = {
        metric: (
            metrics[pool_name][metric]["mean"]
            / metrics[baseline_name][metric]["mean"]
        )
        for metric in PRIMARY_SELECTION_METRICS
    }
    composite = float(np.exp(np.mean(np.log(list(ratios.values())))))
    success = bool(
        composite < 1.0 and all(value <= 1.0 for value in ratios.values())
    )
    return ratios, composite, success


def run(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(
            f"FTSE evaluation output already exists: {output}. "
            "The one-time protocol cannot be overwritten."
        )
    frozen_path = Path(args.frozen_protocol)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen["status"] != "frozen_before_external_evaluation":
        raise ValueError(
            "Protocol is not in the frozen pre-evaluation state."
        )

    config = load_config(args.config)
    frozen_config = load_config(args.freeze_config)
    returns, market = _load_frozen_history_with_current_extension(
        frozen_config, config, args.max_stocks
    )
    pre2024_returns = returns.loc[returns.index < "2024-01-01"]
    pre2024_market = market.reindex(pre2024_returns.index)
    actual_fingerprint = _panel_fingerprint(pre2024_returns, pre2024_market)
    expected_fingerprint = frozen["protocol"]["pre2024_panel_fingerprint"]
    if actual_fingerprint != expected_fingerprint:
        raise ValueError(
            "Pre-2024 panel fingerprint changed. Refusing evaluation: "
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

    pool_weight_lookup = {
        int(row["seed"]): row["pool_weights"]
        for row in frozen["seed_reports"]
    }
    seeds = [row["seed"] for row in frozen["seed_reports"]]
    pool_seed_scores = {pool_name: [] for pool_name in POOL_NAMES.values()}
    pooled_horizon_seed_scores = {
        pool_name: {horizon: [] for horizon in horizon_values}
        for pool_name in POOL_NAMES.values()
    }
    evaluation_seed_reports = []
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
        frozen_weights = pool_weight_lookup[seed]
        scores = {pool_name: [] for pool_name in POOL_NAMES.values()}
        horizon_scores = {
            pool_name: {horizon: [] for horizon in horizon_values}
            for pool_name in POOL_NAMES.values()
        }
        for origin, row in enumerate(evaluation_rows):
            diffusion_paths = model.sample(
                row["context"],
                n_paths=n_paths,
                seed=seed * 10000 + 80000 + origin * n_paths,
            )
            for pool_key, display_name in POOL_NAMES.items():
                base = POOL_BASES[pool_key]
                weight = float(frozen_weights[pool_key])
                states = _linear_pool(
                    diffusion_paths,
                    row["baseline_state_paths"][base],
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
                scores[display_name].append(score)
                for horizon in horizon_values:
                    horizon_scores[display_name][horizon].append(
                        evaluator.score(
                            states[:, :horizon],
                            generated_returns[:, :horizon],
                            row["target_states"][:horizon],
                            row["target_returns"][:horizon],
                        )
                    )
        for display_name in POOL_NAMES.values():
            pool_seed_scores[display_name].append(scores[display_name])
        for display_name in POOL_NAMES.values():
            for horizon in horizon_values:
                pooled_horizon_seed_scores[display_name][horizon].append(
                    horizon_scores[display_name][horizon]
                )
        evaluation_seed_reports.append(
            {
                "seed": seed,
                "frozen_pool_weights": frozen_weights,
                "metrics": evaluator.aggregate(scores),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": frozen_seed["checkpoint_sha256"],
            }
        )

    all_scores = {
        **baseline_scores,
        **{
            display_name: [
                row for seed_rows in pool_seed_scores[display_name]
                for row in seed_rows
            ]
            for display_name in POOL_NAMES.values()
        },
    }
    metrics = evaluator.aggregate(all_scores)
    pooled_by_origin = {
        display_name: _aggregate_seed_scores(pool_seed_scores[display_name])
        for display_name in POOL_NAMES.values()
    }
    paired = {
        display_name: {
            baseline: {
                metric: _bootstrap_difference(
                    pooled_by_origin[display_name][metric],
                    [row[metric] for row in baseline_scores[baseline]],
                    seed=20260811 + pool_index * 1000 + index,
                )
                for index, metric in enumerate(pooled_by_origin[display_name])
            }
            for baseline in baselines.METHODS
        }
        for pool_index, display_name in enumerate(POOL_NAMES.values())
    }
    dependence_robust = {
        display_name: {
            baseline: {
                metric: dependence_robust_comparison(
                    pooled_by_origin[display_name][metric],
                    [row[metric] for row in baseline_scores[baseline]],
                    seed=(
                        20260812
                        + pool_index * 1000
                        + baseline_index * 100
                        + metric_index
                    ),
                )
                for metric_index, metric in enumerate(
                    PRIMARY_SELECTION_METRICS
                )
            }
            for baseline_index, baseline in enumerate(
                ("Gaussian-VAR", "Student-t-VAR")
            )
        }
        for pool_index, display_name in enumerate(POOL_NAMES.values())
    }
    dependence_robust_observable = {
        display_name: {
            baseline: {
                metric: dependence_robust_comparison(
                    pooled_by_origin[display_name][metric],
                    [row[metric] for row in baseline_scores[baseline]],
                    seed=(
                        20260813
                        + pool_index * 1000
                        + baseline_index * 100
                        + metric_index
                    ),
                )
                for metric_index, metric in enumerate(OBSERVABLE_RISK_METRICS)
            }
            for baseline_index, baseline in enumerate(
                ("Gaussian-VAR", "Student-t-VAR", "Block-Bootstrap")
            )
        }
        for pool_index, display_name in enumerate(POOL_NAMES.values())
    }

    primary_ratios, primary_composite, primary_success = _composite_ratio(
        metrics, PRIMARY_POOL, PRIMARY_BASE
    )
    secondary_ratios, secondary_composite, secondary_success = (
        _composite_ratio(metrics, SECONDARY_POOL, SECONDARY_BASE)
    )

    secondary_endpoints = frozen["confirmatory_endpoints"]["secondary"]
    secondary_raw_p = {
        metric: paired[PRIMARY_POOL][PRIMARY_BASE][metric][
            "one_sided_p_diffusion_not_better"
        ]
        for metric in secondary_endpoints
    }
    secondary_adjusted_p = _bh_adjust(secondary_raw_p)

    horizon_robustness = {}
    for horizon in horizon_values:
        horizon_metrics = {}
        for method, rows in baseline_horizon_scores[horizon].items():
            horizon_metrics[method] = rows
        for display_name in POOL_NAMES.values():
            horizon_metrics[display_name] = [
                row
                for seed_rows in pooled_horizon_seed_scores[display_name][
                    horizon
                ]
                for row in seed_rows
            ]
        aggregated = evaluator.aggregate(horizon_metrics)
        horizon_robustness[str(horizon)] = {
            method: {
                metric: values[metric]["mean"]
                for metric in PRIMARY_SELECTION_METRICS
            }
            for method, values in aggregated.items()
        }

    report_role = getattr(args, "report_role", "locked_external_evaluation")
    timestamp_key = (
        "evaluated_at"
        if report_role == "locked_external_evaluation"
        else "analyzed_at"
    )
    report = {
        timestamp_key: datetime.now().isoformat(),
        "report_role": report_role,
        "status": (
            "locked_external_evaluation"
            if report_role == "locked_external_evaluation"
            else "posthoc_robustness_audit"
        ),
        "frozen_protocol": str(frozen_path),
        "frozen_protocol_sha256": _sha256(frozen_path),
        "protocol_checks": {
            "pre2024_panel_fingerprint_match": True,
            "universe_manifest_hash_match": True,
            "checkpoint_hashes_match": True,
            "output_was_absent_before_run": True,
            "model_or_weight_retuning_after_2024": False,
        },
        "evaluation_period": {
            "start": str(splits.test_returns.index.min().date()),
            "end": str(splits.test_returns.index.max().date()),
            "n_days": len(splits.test_returns),
            "n_nonoverlapping_origins": len(splits.test),
            "n_assets": len(baselines.stocks),
        },
        "primary_decision": {
            "pool": PRIMARY_POOL,
            "baseline": PRIMARY_BASE,
            "success": primary_success,
            "rule": frozen["confirmatory_endpoints"]["success_rule"],
            "composite_ratio": primary_composite,
            "co_primary_ratios": primary_ratios,
            "co_primary_paired_bootstrap": paired[PRIMARY_POOL][
                PRIMARY_BASE
            ],
        },
        "secondary_decision": {
            "pool": SECONDARY_POOL,
            "baseline": SECONDARY_BASE,
            "success": secondary_success,
            "rule": frozen["confirmatory_endpoints"]["success_rule"],
            "composite_ratio": secondary_composite,
            "co_primary_ratios": secondary_ratios,
            "co_primary_paired_bootstrap": paired[SECONDARY_POOL][
                SECONDARY_BASE
            ],
        },
        "metrics": metrics,
        "paired_origin_bootstrap": paired,
        "dependence_robust_primary_comparisons": dependence_robust,
        "dependence_robust_observable_risk_comparisons": (
            dependence_robust_observable
        ),
        "origin_level_scores": {
            **pooled_by_origin,
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
        "seed_reports": evaluation_seed_reports,
        "origins": origins,
        "seeds": seeds,
    }
    if report_role == "posthoc_robustness_audit":
        locked_result_path = Path(args.locked_result_reference)
        locked_result = json.loads(
            locked_result_path.read_text(encoding="utf-8")
        )
        locked_primary = locked_result["primary_decision"]
        locked_secondary = locked_result["secondary_decision"]
        report["locked_result_reference"] = str(locked_result_path)
        report["reproduction_checks"] = {
            "primary_composite_exact_match": bool(
                primary_composite == locked_primary["composite_ratio"]
            ),
            "secondary_composite_exact_match": bool(
                secondary_composite == locked_secondary["composite_ratio"]
            ),
            "locked_decisions_unchanged": True,
            "interpretation": (
                "Post-hoc inference only; this audit does not create a new "
                "evaluation sample or alter the locked decisions."
            ),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"FTSE evaluation saved; primary composite={primary_composite:.6f} "
        f"(success={primary_success}); secondary composite="
        f"{secondary_composite:.6f} (success={secondary_success}); "
        f"origins={len(splits.test)}"
    )
    print(f"Saved immutable FTSE evaluation report to {output}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="research_ftse100_confirmation.yaml"
    )
    parser.add_argument(
        "--locked-result-reference",
        default="research_output/ftse100_confirmation/confirmation.json",
        help="Original immutable result used to verify a post-hoc audit rerun.",
    )
    parser.add_argument(
        "--frozen-protocol",
        default="research_output/ftse100_frozen/frozen_protocol.json",
    )
    parser.add_argument(
        "--freeze-config", default="research_ftse100_freeze.yaml"
    )
    parser.add_argument(
        "--output",
        default="research_output/ftse100_confirmation/confirmation.json",
    )
    parser.add_argument("--max-stocks", type=int, default=100)
    parser.add_argument(
        "--report-role",
        choices=("locked_external_evaluation", "posthoc_robustness_audit"),
        default="locked_external_evaluation",
        help=(
            "Label a non-overwriting rerun as a post-hoc robustness audit; "
            "this does not create a new evaluation sample."
        ),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
