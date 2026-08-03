"""One-time CSI 300 external replication of the fixed Phase 3B pool."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import load_config
from data.loader import DataPipeline
from diffusion.conditional_path import VARResidualPathDiffusion
from phase2b_benchmark import reconstruct_paths
from phase2c_benchmark import _restore_checkpoint
from research.freeze_phase2f import (
    _file_sha256,
    _panel_fingerprint,
)
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
from research.phase3b_state_pool import CANDIDATE
from sequences.baselines import Phase2ABaselines
from sequences.dataset import FactorStateSequenceBuilder
from sequences.evaluation import PathForecastEvaluator


def _load_frozen_history_with_current_extension(
    frozen_config, current_config, max_stocks
):
    """Splice only new CSI observations onto immutable pre-2024 history."""
    frozen_pipeline = DataPipeline(frozen_config)
    current_pipeline = DataPipeline(current_config)
    frozen_raw, frozen_market = frozen_pipeline.prepare_raw_returns(
        max_stocks=max_stocks
    )
    current_raw, current_market = current_pipeline.prepare_raw_returns(
        max_stocks=max_stocks
    )
    if list(frozen_raw.columns) != list(current_raw.columns):
        missing = sorted(set(frozen_raw.columns) - set(current_raw.columns))
        extra = sorted(set(current_raw.columns) - set(frozen_raw.columns))
        raise ValueError(
            "Current CSI panel cannot reproduce frozen asset columns; "
            f"missing={missing}, extra={extra}."
        )

    cutoff = pd.Timestamp("2024-01-01")
    fit_end = pd.to_datetime(frozen_config.data.preprocess_fit_end_date)
    reference = frozen_raw.loc[frozen_raw.index <= fit_end]
    frozen_returns = frozen_pipeline.winsorize(frozen_raw, reference)
    current_post = frozen_pipeline.winsorize(
        current_raw.loc[current_raw.index >= cutoff], reference
    )
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


def _external_success(state_composite, state_ratios, energy_p, return_composite):
    checks = {
        "state_composite_below_one": state_composite < 1.0,
        "neither_state_primary_worse": all(
            value <= 1.0 for value in state_ratios.values()
        ),
        "paired_energy_one_sided_p_below_0_10": energy_p < 0.10,
        "return_composite_not_worse_by_more_than_two_percent": (
            return_composite <= 1.02
        ),
    }
    return bool(all(checks.values())), checks


def _score_baselines_with_returns(
    config,
    splits,
    baselines,
    evaluator,
    return_scale,
    n_paths,
):
    scores = {method: [] for method in baselines.METHODS}
    rows = []
    origins = []
    for origin in range(len(splits.test)):
        forecast = baselines.forecast(
            splits.test.context[origin],
            config.temporal.horizon,
            n_paths=n_paths,
            seed=1200000 + origin * n_paths,
        )
        target_dates = splits.test.target_dates[origin]
        target_returns = splits.test_returns.reindex(target_dates)[
            baselines.stocks
        ].values
        target_states = splits.test.target[origin]
        for method in baselines.METHODS:
            scores[method].append(
                _score(
                    evaluator,
                    forecast.states[method],
                    forecast.returns[method],
                    target_states,
                    target_returns,
                    return_scale,
                )
            )
        rows.append(
            {
                "context": splits.test.context[origin],
                "base_states": forecast.states,
                "innovations": forecast.innovations["VAR-GARCH"],
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
    return scores, rows, origins


def _paired(candidate_seed_scores, comparison_scores, seed_offset):
    candidate = _aggregate_seed_scores(candidate_seed_scores)
    if isinstance(comparison_scores[0], list):
        comparison = _aggregate_seed_scores(comparison_scores)
    else:
        comparison = {
            key: [row[key] for row in comparison_scores]
            for key in comparison_scores[0]
        }
    return {
        metric: _bootstrap_difference(
            candidate[metric],
            comparison[metric],
            seed=seed_offset + index,
        )
        for index, metric in enumerate(candidate)
    }


def run(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(
            f"CSI external replication output already exists: {output}. "
            "The one-time result cannot be overwritten."
        )
    frozen_path = Path(args.frozen_protocol)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen["status"] != "frozen_before_external_replication":
        raise ValueError("CSI protocol is not in its frozen state.")

    frozen_config = load_config(args.freeze_config)
    current_config = load_config(args.config)
    returns, market = _load_frozen_history_with_current_extension(
        frozen_config, current_config, args.max_stocks
    )
    pre2024 = returns.loc[returns.index < "2024-01-01"]
    pre2024_market = market.reindex(pre2024.index)
    actual_fingerprint = _panel_fingerprint(pre2024, pre2024_market)
    expected_fingerprint = frozen["protocol"][
        "pre2024_panel_fingerprint"
    ]
    if actual_fingerprint != expected_fingerprint:
        raise ValueError(
            "Frozen CSI history changed. Refusing external replication: "
            f"expected {expected_fingerprint}, got {actual_fingerprint}."
        )
    manifest = Path(current_config.data.universe_manifest)
    if _file_sha256(manifest) != frozen["protocol"][
        "universe_manifest_sha256"
    ]:
        raise ValueError("CSI universe manifest hash differs from frozen protocol.")
    if list(pre2024.columns) != frozen["protocol"]["asset_columns"]:
        raise ValueError("CSI frozen asset ordering changed.")

    splits = FactorStateSequenceBuilder(
        current_config,
        context_length=current_config.temporal.context_length,
        horizon=current_config.temporal.horizon,
        evaluation_stride=current_config.temporal.horizon,
        train_end_date=frozen["protocol"]["train_end"],
        validation_end_date=frozen["protocol"][
            "checkpoint_validation_end"
        ],
    ).build(returns, market)
    baselines = Phase2ABaselines(current_config, splits)
    evaluator = PathForecastEvaluator(
        splits.latent_metadata["n_mean_factors"]
    )
    n_paths = frozen["protocol"]["ensemble_paths_per_seed"]
    return_scale = (
        splits.train_returns[baselines.stocks]
        .std(ddof=0)
        .clip(lower=1e-6)
        .values
    )
    baseline_scores, rows, origins = _score_baselines_with_returns(
        current_config,
        splits,
        baselines,
        evaluator,
        return_scale,
        n_paths,
    )

    candidate_seed_scores = []
    incumbent_seed_scores = []
    seed_reports = []
    for frozen_seed in frozen["seed_reports"]:
        seed = frozen_seed["seed"]
        checkpoint = Path(frozen_seed["checkpoint"])
        if _file_sha256(checkpoint) != frozen_seed["checkpoint_sha256"]:
            raise ValueError(f"CSI checkpoint hash mismatch for seed {seed}.")
        model = VARResidualPathDiffusion(
            current_config,
            state_dim=splits.train_states.shape[1],
            horizon=current_config.temporal.horizon,
            n_mean_factors=splits.latent_metadata["n_mean_factors"],
        )
        _restore_checkpoint(model, splits, checkpoint)
        candidate_weight = frozen_seed["candidate_diffusion_weight"]
        incumbent_weight = frozen_seed["incumbent_diffusion_weight"]
        candidate_rows = []
        incumbent_rows = []
        for origin, row in enumerate(rows):
            diffusion = model.sample(
                row["context"],
                n_paths=n_paths,
                seed=seed * 10000 + 1300000 + origin * n_paths,
            )
            candidate_states = _linear_pool(
                diffusion,
                row["base_states"]["Student-t-VAR"],
                candidate_weight,
            )
            incumbent_states = _linear_pool(
                diffusion,
                row["base_states"]["Gaussian-VAR"],
                incumbent_weight,
            )
            candidate_returns = reconstruct_paths(
                splits,
                candidate_states,
                row["innovations"],
                baselines.stocks,
            )
            incumbent_returns = reconstruct_paths(
                splits,
                incumbent_states,
                row["innovations"],
                baselines.stocks,
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
        candidate_seed_scores.append(candidate_rows)
        incumbent_seed_scores.append(incumbent_rows)
        seed_reports.append(
            {
                "seed": seed,
                "candidate_diffusion_weight": candidate_weight,
                "incumbent_diffusion_weight": incumbent_weight,
                "candidate_metrics": evaluator.aggregate(
                    {CANDIDATE: candidate_rows}
                )[CANDIDATE],
                "incumbent_metrics": evaluator.aggregate(
                    {MODEL_NAME: incumbent_rows}
                )[MODEL_NAME],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": frozen_seed["checkpoint_sha256"],
            }
        )

    all_scores = {
        **baseline_scores,
        CANDIDATE: [
            row for seed_rows in candidate_seed_scores for row in seed_rows
        ],
        MODEL_NAME: [
            row for seed_rows in incumbent_seed_scores for row in seed_rows
        ],
    }
    metrics = evaluator.aggregate(all_scores)
    state_composite, state_ratios = _geometric_ratio(
        metrics[CANDIDATE], metrics[MODEL_NAME], PRIMARY_SELECTION_METRICS
    )
    return_composite, return_ratios = _geometric_ratio(
        metrics[CANDIDATE], metrics[MODEL_NAME], RETURN_SELECTION_METRICS
    )
    paired_incumbent = _paired(
        candidate_seed_scores, incumbent_seed_scores, 20260830
    )
    energy_p = paired_incumbent["state_energy_score"][
        "one_sided_p_diffusion_not_better"
    ]
    success, checks = _external_success(
        state_composite, state_ratios, energy_p, return_composite
    )
    paired_student = _paired(
        candidate_seed_scores,
        baseline_scores["Student-t-VAR"],
        20260930,
    )
    student_state_composite, student_state_ratios = _geometric_ratio(
        metrics[CANDIDATE],
        metrics["Student-t-VAR"],
        PRIMARY_SELECTION_METRICS,
    )
    student_return_composite, student_return_ratios = _geometric_ratio(
        metrics[CANDIDATE],
        metrics["Student-t-VAR"],
        RETURN_SELECTION_METRICS,
    )

    report = {
        "replicated_at": datetime.now().isoformat(),
        "status": (
            "external_replication_success"
            if success
            else "external_replication_failure"
        ),
        "frozen_protocol": str(frozen_path),
        "frozen_protocol_sha256": _file_sha256(frozen_path),
        "protocol_checks": {
            "pre2024_panel_fingerprint_match": True,
            "universe_manifest_hash_match": True,
            "asset_columns_match": True,
            "checkpoint_hashes_match": True,
            "output_was_absent_before_run": True,
            "post2023_model_or_weight_tuning": False,
        },
        "replication_period": {
            "start": str(splits.test_returns.index.min().date()),
            "end": str(splits.test_returns.index.max().date()),
            "n_days": len(splits.test_returns),
            "n_nonoverlapping_origins": len(splits.test),
            "n_assets": len(baselines.stocks),
        },
        "external_decision": {
            "success": success,
            "rule": frozen["external_replication_endpoints"][
                "success_rule"
            ],
            "checks": checks,
            "state_composite_ratio_vs_phase2f": state_composite,
            "state_ratios_vs_phase2f": state_ratios,
            "return_composite_ratio_vs_phase2f": return_composite,
            "return_ratios_vs_phase2f": return_ratios,
            "paired_candidate_vs_phase2f": paired_incumbent,
        },
        "strong_student_t_baseline": {
            "state_composite_ratio": student_state_composite,
            "state_ratios": student_state_ratios,
            "return_composite_ratio": student_return_composite,
            "return_ratios": student_return_ratios,
            "paired_candidate_vs_student_t": paired_student,
        },
        "metrics": metrics,
        "seed_reports": seed_reports,
        "origins": origins,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"CSI external status={report['status']}; "
        f"state composite={state_composite:.6f}; "
        f"return composite={return_composite:.6f}; "
        f"origins={len(splits.test)}"
    )
    print(f"Saved immutable CSI replication report to {output}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="research_csi300_replication.yaml"
    )
    parser.add_argument(
        "--freeze-config", default="research_csi300_freeze.yaml"
    )
    parser.add_argument(
        "--frozen-protocol",
        default="research_output/csi300_external/frozen_protocol.json",
    )
    parser.add_argument(
        "--output",
        default="research_output/csi300_external/replication.json",
    )
    parser.add_argument("--max-stocks", type=int, default=100)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

