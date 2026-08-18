"""Pre-2024 descriptive factor-quality diagnostics for the eligible markets.

Purely descriptive and computed only on observations ending in 2023. It
documents how much of each frozen cross-section the ten-dimensional state can
represent (mean-factor OLS fit and log-volatility PCA), so the one-time
evaluation's heterogeneity can be interpreted against representation quality.
It performs no model training, no weight selection, and cannot change any
frozen decision.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import load_config
from data.yf_loader import YFinanceDataPipeline
from research.freeze_phase2f import _panel_fingerprint
from research.prospective_multimarket import (
    load_json,
    market_config,
    validate_pre2024_inputs,
)
from sequences.dataset import FactorStateSequenceBuilder


def mean_factor_r2(extractor, train_returns, train_market):
    """Cross-sectional OLS fit quality of the frozen mean-factor model."""
    factors = extractor._mean_factor_frame(train_returns, train_market).dropna()
    common = train_returns.index.intersection(factors.index)
    design = factors.loc[common].values
    per_stock = {}
    for stock in train_returns.columns:
        if stock not in extractor.mean_loadings_.index:
            continue
        observed = train_returns.loc[common, stock].values
        fitted = (
            design @ extractor.mean_loadings_.loc[stock].values
            + extractor.mean_intercepts_.loc[stock]
        )
        residual = observed - fitted
        total_ss = float(np.sum((observed - observed.mean()) ** 2))
        if not np.isfinite(total_ss) or total_ss <= 0:
            continue
        per_stock[stock] = float(1.0 - np.sum(residual**2) / total_ss)
    values = np.asarray(list(per_stock.values()), dtype=float)
    if len(values) == 0:
        raise ValueError("No mean-factor fits available for R2 diagnostics.")
    return {
        "n_fitted": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "per_stock": per_stock,
    }


def _diagnose_market(base_config, registered_market, smoke_market):
    code = registered_market["code"]
    config = market_config(base_config, registered_market)
    if pd.to_datetime(config.data.end_date) >= pd.Timestamp("2024-01-02"):
        raise ValueError(f"{code}: diagnostic config crosses the outcome boundary.")
    returns, market = YFinanceDataPipeline(config).load_all_data(
        max_stocks=registered_market["max_stocks"]
    )
    if str(returns.index.max().date()) > "2023-12-31":
        raise ValueError(f"{code}: post-2023 observation loaded in diagnostics.")
    if list(returns.columns) != smoke_market["asset_columns"]:
        raise ValueError(f"{code}: eligible asset columns changed after screening.")
    fingerprint = _panel_fingerprint(returns, market)
    if fingerprint != smoke_market["pre2024_panel_fingerprint"]:
        raise ValueError(f"{code}: pre-2024 panel fingerprint changed.")

    splits = FactorStateSequenceBuilder(
        config,
        context_length=config.temporal.context_length,
        horizon=config.temporal.horizon,
        evaluation_stride=config.temporal.horizon,
        train_end_date="2022-12-30",
        validation_end_date="2023-12-31",
        allow_empty_test=True,
    ).build(returns, market)
    extractor = splits.extractor
    explained = np.asarray(extractor.vol_pca_.explained_variance_ratio_)
    return {
        "code": code,
        "name": registered_market["name"],
        "country": registered_market["country"],
        "status": "diagnosed_pre2024",
        "data_start": str(returns.index.min().date()),
        "data_end": str(returns.index.max().date()),
        "train_window": {
            "start": str(splits.train_returns.index.min().date()),
            "end": str(splits.train_returns.index.max().date()),
        },
        "n_assets": int(len(splits.train_returns.columns)),
        "n_vol_pca_assets": int(len(extractor.vol_assets_)),
        "pre2024_panel_fingerprint": fingerprint,
        "mean_factor_r2": mean_factor_r2(
            extractor, splits.train_returns, splits.train_market
        ),
        "volatility_pca": {
            "n_components": int(len(explained)),
            "explained_variance_ratio": [float(value) for value in explained],
            "cumulative_explained_variance": float(explained.sum()),
        },
        "post2023_data_loaded": False,
    }


def run(args):
    output_root = Path(args.output_dir)
    summary_path = output_root / "summary.json"
    if summary_path.exists() and not args.allow_overwrite:
        raise FileExistsError(
            f"Factor-quality summary already exists: {summary_path}; "
            "refusing overwrite."
        )
    registry, smoke, registered, eligible, _, checks = validate_pre2024_inputs(
        Path(args.base_config), Path(args.registry), Path(args.smoke)
    )
    base = load_config(args.base_config)
    output_root.mkdir(parents=True, exist_ok=True)
    markets = []
    for smoke_market in eligible:
        report = _diagnose_market(
            base, registered[smoke_market["code"]], smoke_market
        )
        market_path = output_root / f"{report['code']}.json"
        market_path.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        markets.append(report)
        print(
            f"{report['code']}: mean-factor R2 mean "
            f"{report['mean_factor_r2']['mean']:.3f}; volatility PCA "
            f"{report['volatility_pca']['cumulative_explained_variance']:.2%} "
            f"over {report['n_assets']} assets"
        )
    summary = {
        "diagnosed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pre2024_factor_quality_descriptive",
        "study_id": registry["study_id"],
        "role": (
            "descriptive pre-lock diagnostics; cannot change any frozen "
            "decision and uses no post-2023 observations"
        ),
        "input_checks": checks,
        "markets": [
            {
                "code": row["code"],
                "n_assets": row["n_assets"],
                "n_vol_pca_assets": row["n_vol_pca_assets"],
                "mean_factor_r2_mean": row["mean_factor_r2"]["mean"],
                "mean_factor_r2_min": row["mean_factor_r2"]["min"],
                "volatility_pca_cumulative_explained_variance": row[
                    "volatility_pca"
                ]["cumulative_explained_variance"],
            }
            for row in markets
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Factor-quality summary saved to {summary_path}")
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
        "--output-dir",
        default="research_output/prospective_multimarket/factor_quality",
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
