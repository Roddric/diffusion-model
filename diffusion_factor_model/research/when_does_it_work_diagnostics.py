"""Descriptive 'when does it work' diagnostics for the six eligible markets.

Hypothesis-generating only. All features are computed on the training window
(observations through 2022-12-30). Post-2023 outcomes from the completed
one-time evaluation are read purely as descriptive labels; with K = 6 no
formal inference is performed or implied.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from config.config import load_config
from data.yf_loader import YFinanceDataPipeline
from research.freeze_phase2f import _panel_fingerprint
from research.prospective_multimarket import (
    load_json,
    market_config,
    validate_pre2024_inputs,
)

TRAIN_WINDOW_END = pd.Timestamp("2022-12-30")


def _spearman(features, outcome):
    if len(set(outcome)) < 2 or len(set(features)) < 2:
        return float("nan")
    rho, pvalue = stats.spearmanr(features, outcome)
    return {"rho": float(rho), "raw_two_sided_p": float(pvalue)}


def _market_structure(returns, market, asset_columns):
    train = returns.loc[returns.index <= TRAIN_WINDOW_END, asset_columns]
    market_train = market.loc[market.index <= TRAIN_WINDOW_END]
    correlation = train.corr()
    upper_triangle = correlation.values[np.triu_indices_from(correlation.values, k=1)]
    return {
        "train_window_end": str(train.index.max().date()),
        "n_sessions": int(len(train)),
        "annualized_benchmark_volatility": float(
            market_train.std(ddof=1) * np.sqrt(252)
        ),
        "mean_pairwise_correlation": float(np.nanmean(upper_triangle)),
        "annualized_cross_section_dispersion": float(
            train.std(axis=1, ddof=1).mean() * np.sqrt(252)
        ),
    }


def run(args):
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "diagnostics.json"
    if summary_path.exists() and not args.allow_overwrite:
        raise FileExistsError(
            f"Diagnostics already exist: {summary_path}; refusing overwrite."
        )
    evaluation_path = Path(args.evaluation)
    evaluation = load_json(evaluation_path)
    if evaluation.get("status") not in (
        "one_time_evaluation_complete",
        "one_time_evaluation_terminal_attrition",
    ):
        raise ValueError("Evaluation report is not a complete one-time report.")
    frozen = load_json(args.frozen_protocol)
    registry, smoke, registered, eligible, _, checks = validate_pre2024_inputs(
        Path(args.base_config), Path(args.registry), Path(args.smoke)
    )
    base = load_config(args.base_config)
    factor_quality_dir = Path(args.factor_quality_dir)
    evaluated = {row["code"]: row for row in evaluation["markets"]}

    markets = []
    for smoke_market in eligible:
        code = smoke_market["code"]
        registered_market = registered[code]
        config = market_config(base, registered_market)
        if pd.to_datetime(config.data.end_date) >= pd.Timestamp("2024-01-02"):
            raise ValueError(f"{code}: diagnostics config crosses 2023.")
        returns, market = YFinanceDataPipeline(config).load_all_data(
            max_stocks=registered_market["max_stocks"]
        )
        if str(returns.index.max().date()) > "2023-12-31":
            raise ValueError(f"{code}: post-2023 observation loaded.")
        if list(returns.columns) != smoke_market["asset_columns"]:
            raise ValueError(f"{code}: eligible asset columns changed.")
        if _panel_fingerprint(returns, market) != smoke_market[
            "pre2024_panel_fingerprint"
        ]:
            raise ValueError(f"{code}: pre-2024 panel fingerprint changed.")

        frozen_market = [m for m in frozen["markets"] if m["code"] == code][0]
        weights = [
            seed_report["pool_weights"]["student_t_base_pool"]
            for seed_report in frozen_market["seed_reports"]
        ]
        factor_quality = load_json(factor_quality_dir / f"{code}.json")
        outcome = evaluated[code]
        state_decision = outcome["state_decision"]
        covariance_decision = outcome["covariance_decision"]
        markets.append(
            {
                "code": code,
                "country": registered_market["country"],
                "features": {
                    "n_assets": int(len(smoke_market["asset_columns"])),
                    "mean_factor_r2_mean": factor_quality["mean_factor_r2"]["mean"],
                    "mean_factor_r2_min": factor_quality["mean_factor_r2"]["min"],
                    "volatility_pca_cumulative_explained_variance": factor_quality[
                        "volatility_pca"
                    ]["cumulative_explained_variance"],
                    **_market_structure(
                        returns, market, smoke_market["asset_columns"]
                    ),
                },
                "validation_selected_pool_weight_mean": float(np.mean(weights)),
                "validation_selected_pool_weights": weights,
                "post2023_outcome_descriptive_only": {
                    "state_strong_baseline_ratio": state_decision[
                        "strong_baseline_ratio"
                    ],
                    "state_passed": state_decision["passed"],
                    "covariance_strong_baseline_ratio": covariance_decision[
                        "strong_baseline_ratio"
                    ],
                    "covariance_passed": covariance_decision["passed"],
                },
            }
        )

    feature_names = sorted(
        name
        for name, value in markets[0]["features"].items()
        if isinstance(value, (int, float))
    )
    weights = np.asarray(
        [row["validation_selected_pool_weight_mean"] for row in markets]
    )
    state_ratios = np.asarray(
        [
            row["post2023_outcome_descriptive_only"]["state_strong_baseline_ratio"]
            for row in markets
        ]
    )
    correlations = {}
    for name in feature_names:
        values = np.asarray([row["features"][name] for row in markets], dtype=float)
        correlations[name] = {
            "values": [float(v) for v in values],
            "spearman_vs_selected_weight": _spearman(values, weights),
            "spearman_vs_post2023_state_ratio": _spearman(values, state_ratios),
        }

    summary = {
        "diagnosed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "descriptive_hypothesis_generating_only",
        "study_id": registry["study_id"],
        "role": (
            "Phase A of the final research route. Features use the training "
            "window through 2022-12-30 only. Post-2023 outcomes are read as "
            "descriptive labels and influence no decision. K=6: raw values and "
            "Spearman rank correlations only; no formal inference."
        ),
        "post2023_outcomes_used_descriptively": True,
        "candidate_selected": False,
        "input_checks": checks,
        "n_markets": len(markets),
        "markets": markets,
        "feature_correlations": correlations,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for code_row in markets:
        outcome = code_row["post2023_outcome_descriptive_only"]
        print(
            f"{code_row['code']:8s} weight={code_row['validation_selected_pool_weight_mean']:.3f} "
            f"state_ratio={outcome['state_strong_baseline_ratio']:.4f} "
            f"r2_mean={code_row['features']['mean_factor_r2_mean']:.3f} "
            f"vol_pca={code_row['features']['volatility_pca_cumulative_explained_variance']:.3f}"
        )
    print(f"Diagnostics saved to {summary_path}")
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
        "--evaluation",
        default="research_output/prospective_multimarket/evaluation.json",
    )
    parser.add_argument(
        "--factor-quality-dir",
        default="research_output/prospective_multimarket/factor_quality",
    )
    parser.add_argument(
        "--output-dir",
        default="research_output/prospective_multimarket/when_does_it_work",
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
