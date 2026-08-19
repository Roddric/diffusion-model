"""Cross-market oracle-ceiling attribution for the frozen prospective markets.

Phase B of the final research route. Pre-2024 data only: windows are train
through 2022-12-30, a 2023-01 validation buffer, and development/attribution
origins in February-December 2023. Checkpoints and pool weights are the frozen
ones. Diagnostic only; no candidate is selected and no post-2023 observation
is loaded.
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from config.config import load_config
from data.yf_loader import YFinanceDataPipeline
from diffusion.conditional_path import VARResidualPathDiffusion
from phase2c_benchmark import _restore_checkpoint
from research.freeze_phase2f import _panel_fingerprint
from research.oracle_ceiling_attribution import (
    _aggregate_by_origin,
    _coalition_metrics,
    _coalition_scores,
    _origin_rows,
    _representation_drift_diagnostic,
    _summarize_attribution,
)
from research.prospective_multimarket import (
    load_json,
    market_config,
    sha256,
    validate_pre2024_inputs,
)
from sequences.baselines import Phase2ABaselines
from sequences.dataset import FactorStateSequenceBuilder
from sequences.evaluation import PathForecastEvaluator

TRAIN_END = "2022-12-30"
VALIDATION_BUFFER_END = "2023-01-31"
PROCEED_POINT_THRESHOLD = 0.80
PROCEED_INTERVAL_LOWER_THRESHOLD = 0.60
PROCEED_MIN_MARKETS_ABOVE_FLOOR = 5
PER_MARKET_SHARE_FLOOR = 0.50
COMPONENT_NAMES = ("mean_state", "volatility_state", "innovation")


def _mean_t_interval(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(n))
    half = float(stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1)) * se
    return {"mean": mean, "ci_lower": mean - half, "ci_upper": mean + half, "n": n}


def _attribute_market(base_config, registered_market, frozen_market, smoke_market, args):
    code = frozen_market["code"]
    config = market_config(base_config, registered_market)
    if pd.to_datetime(config.data.end_date) >= pd.Timestamp("2024-01-02"):
        raise ValueError(f"{code}: attribution config crosses the 2024 boundary.")
    returns, market = YFinanceDataPipeline(config).load_all_data(
        max_stocks=registered_market["max_stocks"]
    )
    if str(returns.index.max().date()) > "2023-12-31":
        raise ValueError(f"{code}: post-2023 observation loaded in attribution.")
    if list(returns.columns) != smoke_market["asset_columns"]:
        raise ValueError(f"{code}: eligible asset columns changed.")
    if _panel_fingerprint(returns, market) != smoke_market["pre2024_panel_fingerprint"]:
        raise ValueError(f"{code}: pre-2024 panel fingerprint changed.")

    splits = FactorStateSequenceBuilder(
        config,
        context_length=config.temporal.context_length,
        horizon=config.temporal.horizon,
        evaluation_stride=config.temporal.horizon,
        train_end_date=TRAIN_END,
        validation_end_date=VALIDATION_BUFFER_END,
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

    models = []
    for seed_report in frozen_market["seed_reports"]:
        checkpoint_path = Path(seed_report["checkpoint"])
        if sha256(checkpoint_path) != seed_report["checkpoint_sha256"]:
            raise ValueError(
                f"{code}: checkpoint hash mismatch for seed {seed_report['seed']}."
            )
        model = VARResidualPathDiffusion(
            config,
            state_dim=splits.train_states.shape[1],
            horizon=config.temporal.horizon,
            n_mean_factors=splits.latent_metadata["n_mean_factors"],
        )
        _restore_checkpoint(model, splits, checkpoint_path)
        models.append(
            (
                seed_report["seed"],
                seed_report["pool_weights"]["student_t_base_pool"],
                model,
            )
        )

    rows, origins = _origin_rows(config, splits, baselines, returns, n_paths)
    if len(rows) == 0:
        raise ValueError(f"{code}: no development origins available.")
    scores = _coalition_scores(
        config, splits, evaluator, models, rows, return_scale, n_paths
    )
    per_origin = _aggregate_by_origin(scores)
    attribution = _summarize_attribution(
        per_origin,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    representation = _representation_drift_diagnostic(
        splits, rows, returns, market, n_folds=args.loading_folds
    )
    return {
        "code": code,
        "country": frozen_market["country"],
        "status": "diagnosed_pre2024",
        "protocol": {
            "phase": "final_route_phase_b_cross_market_oracle_attribution",
            "train_end": TRAIN_END,
            "validation_buffer_end": VALIDATION_BUFFER_END,
            "development_period": (
                f"{origins[0]['target_start']} through {origins[-1]['target_end']}"
            ),
            "n_assets": len(baselines.stocks),
            "n_origins": len(rows),
            "n_paths_per_seed": n_paths,
            "seeds": [seed for seed, _, _ in models],
            "pool_weights": [weight for _, weight, _ in models],
            "checkpoint_hashes_match": True,
            "post2023_data_loaded": False,
            "candidate_selected": False,
        },
        "coalition_metrics": _coalition_metrics(per_origin),
        "shapley_attribution": attribution,
        "loading_representation_diagnostic": representation,
        "origins": origins,
    }


def _population_summary(market_reports):
    shares = {name: [] for name in COMPONENT_NAMES}
    for report in market_reports:
        for name in COMPONENT_NAMES:
            shares[name].append(
                report["shapley_attribution"]["components"][name][
                    "share_of_full_oracle_reduction"
                ]
            )
    combined = [
        shares["mean_state"][i] + shares["innovation"][i]
        for i in range(len(market_reports))
    ]
    intervals = {name: _mean_t_interval(shares[name]) for name in COMPONENT_NAMES}
    combined_interval = _mean_t_interval(combined)
    markets_above_floor = int(
        sum(value > PER_MARKET_SHARE_FLOOR for value in combined)
    )
    gate_passed = bool(
        combined_interval["mean"] >= PROCEED_POINT_THRESHOLD
        and combined_interval["ci_lower"] > PROCEED_INTERVAL_LOWER_THRESHOLD
        and markets_above_floor >= PROCEED_MIN_MARKETS_ABOVE_FLOOR
    )
    return {
        "unit_of_replication": "market",
        "component_share_intervals": intervals,
        "mean_state_plus_innovation": {
            "per_market": combined,
            **combined_interval,
            "markets_above_floor": markets_above_floor,
        },
        "proceed_gate": {
            "point_threshold": PROCEED_POINT_THRESHOLD,
            "interval_lower_threshold": PROCEED_INTERVAL_LOWER_THRESHOLD,
            "min_markets_above_floor": PROCEED_MIN_MARKETS_ABOVE_FLOOR,
            "per_market_floor": PER_MARKET_SHARE_FLOOR,
            "passed": gate_passed,
            "consequence": (
                "proceed to Phase C candidate development under the frozen rules"
                if gate_passed
                else "final route ends here; reconstruction is not a universal "
                "bottleneck and no candidate work is justified"
            ),
        },
    }


def run(args):
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not args.allow_overwrite:
        raise FileExistsError(
            f"Attribution summary already exists: {summary_path}; refusing overwrite."
        )
    registry, smoke, registered, eligible, _, checks = validate_pre2024_inputs(
        Path(args.base_config), Path(args.registry), Path(args.smoke)
    )
    frozen = load_json(args.frozen_protocol)
    if frozen.get("status") != "frozen_before_post2023_evaluation":
        raise ValueError("Frozen protocol is missing or has the wrong status.")
    base = load_config(args.base_config)
    frozen_by_code = {row["code"]: row for row in frozen["markets"]}
    output_dir.mkdir(parents=True, exist_ok=True)

    market_reports = []
    for smoke_market in eligible:
        code = smoke_market["code"]
        market_path = output_dir / f"{code}.json"
        if market_path.exists():
            market_reports.append(load_json(market_path))
            print(f"{code}: loaded existing attribution artifact")
            continue
        print(f"{code}: scoring coalitions...", flush=True)
        report = _attribute_market(
            base, registered[code], frozen_by_code[code], smoke_market, args
        )
        temporary = market_path.with_suffix(market_path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(market_path)
        market_reports.append(report)
        shares = {
            name: report["shapley_attribution"]["components"][name][
                "share_of_full_oracle_reduction"
            ]
            for name in COMPONENT_NAMES
        }
        print(
            f"{code}: mean_state={shares['mean_state']:.3f} "
            f"volatility_state={shares['volatility_state']:.3f} "
            f"innovation={shares['innovation']:.3f}",
            flush=True,
        )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "final_route_phase_b_complete",
        "study_id": registry["study_id"],
        "role": (
            "Cross-market oracle-ceiling attribution on pre-2024 data with the "
            "frozen candidate. Diagnostic only; the proceed gate decides whether "
            "the final route continues to candidate development or ends."
        ),
        "input_checks": checks,
        "frozen_protocol_sha256": sha256(Path(args.frozen_protocol)),
        "markets": [
            {
                "code": row["code"],
                "n_origins": row["protocol"]["n_origins"],
                "component_shares": {
                    name: row["shapley_attribution"]["components"][name][
                        "share_of_full_oracle_reduction"
                    ]
                    for name in COMPONENT_NAMES
                },
                "representation_drift": {
                    "mean_ratio_crossfit_vs_fixed": row[
                        "loading_representation_diagnostic"
                    ]["mean_loading_scaled_rmse"]["ratio_crossfit_vs_fixed"],
                    "vol_ratio_crossfit_vs_fixed": row[
                        "loading_representation_diagnostic"
                    ][
                        "volatility_loading_logvariance_rmse"
                    ]["ratio_crossfit_vs_fixed"],
                },
            }
            for row in market_reports
        ],
        "population_summary": _population_summary(market_reports),
        "decision": {
            "model_modified": False,
            "external_sample_reused": False,
            "candidate_selected": False,
            "post2023_data_loaded": False,
        },
    }
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    gate = summary["population_summary"]["proceed_gate"]
    combined = summary["population_summary"]["mean_state_plus_innovation"]
    print(
        f"mean_state+innovation pooled share={combined['mean']:.3f} "
        f"CI [{combined['ci_lower']:.3f}, {combined['ci_upper']:.3f}]; "
        f"markets above floor: {combined['markets_above_floor']}/6; "
        f"gate passed: {gate['passed']}"
    )
    print(f"Saved {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="research_multimarket_base.yaml")
    parser.add_argument("--registry", default="research_multimarket_markets.json")
    parser.add_argument(
        "--smoke",
        default="research_output/prospective_multimarket/pre2024_smoke.json",
    )
    parser.add_argument(
        "--frozen-protocol",
        default="research_output/prospective_multimarket/frozen_protocol.json",
    )
    parser.add_argument(
        "--output-dir",
        default="research_output/prospective_multimarket/oracle_attribution",
    )
    parser.add_argument("--paths", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260815)
    parser.add_argument("--loading-folds", type=int, default=5)
    parser.add_argument("--allow-overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
