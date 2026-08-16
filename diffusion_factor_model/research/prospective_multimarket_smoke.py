"""Pre-2024-only feasibility screen for the prospective multi-market study."""

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config.config import load_config
from data.yf_loader import YFinanceDataPipeline
from research.freeze_phase2f import _panel_fingerprint
from sequences.dataset import FactorStateSequenceBuilder


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_registry(path):
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    codes = [row["code"] for row in registry["primary_markets"]]
    if len(codes) != len(set(codes)):
        raise ValueError("Market registry contains duplicate codes.")
    return registry


def market_config(base_config, market):
    config = copy.deepcopy(base_config)
    code = market["code"]
    config.data.universe = code
    config.data.universe_manifest = market["universe_manifest"]
    config.data.market_benchmark = market["market_benchmark"]
    config.data.ticker_convention = "verbatim"
    config.data_dir = f"./research_data/prospective_multimarket/{code}"
    config.model_dir = f"./research_models/prospective_multimarket/{code}"
    config.output_dir = f"./research_output/prospective_multimarket/{code}"
    return config


def _screen_market(base_config, market, minimum_assets):
    config = market_config(base_config, market)
    if pd.to_datetime(config.data.end_date) >= pd.Timestamp("2024-01-02"):
        raise ValueError("Smoke-test data must end before 2024 outcomes.")
    manifest = Path(market["universe_manifest"])
    provenance = Path(market["universe_provenance"])
    if not manifest.exists() or not provenance.exists():
        raise FileNotFoundError(f'Missing universe files for {market["code"]}.')
    returns, benchmark = YFinanceDataPipeline(config).load_all_data(
        max_stocks=market["max_stocks"]
    )
    splits = FactorStateSequenceBuilder(
        config,
        context_length=config.temporal.context_length,
        horizon=config.temporal.horizon,
        evaluation_stride=config.temporal.horizon,
        train_end_date="2022-12-30",
        validation_end_date="2023-12-31",
        allow_empty_test=True,
    ).build(returns, benchmark)
    reasons = []
    if returns.shape[1] < minimum_assets:
        reasons.append(f"eligible assets {returns.shape[1]} < {minimum_assets}")
    if len(splits.validation) < 1:
        reasons.append("no 2023 validation origin")
    return {
        "code": market["code"],
        "name": market["name"],
        "country": market["country"],
        "eligible": not reasons,
        "exclusion_reasons": reasons,
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "provenance": str(provenance),
        "provenance_sha256": _sha256(provenance),
        "benchmark": market["market_benchmark"],
        "n_manifest_assets": len(pd.read_csv(manifest)),
        "n_eligible_assets": returns.shape[1],
        "asset_columns": list(returns.columns),
        "data_start": str(returns.index.min().date()),
        "data_end": str(returns.index.max().date()),
        "n_validation_origins": len(splits.validation),
        "pre2024_panel_fingerprint": _panel_fingerprint(returns, benchmark),
        "post2023_data_loaded": False,
    }


def run(args):
    output = Path(args.output)
    if output.exists() and not args.allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite smoke report: {output}")
    base_path = Path(args.base_config)
    registry_path = Path(args.registry)
    base = load_config(base_path)
    registry = load_registry(registry_path)
    registered = {row["code"]: row for row in registry["primary_markets"]}
    requested = args.markets or list(registered)
    unknown = sorted(set(requested) - set(registered))
    if unknown:
        raise ValueError(f"Unknown requested markets: {unknown}")
    markets = [
        _screen_market(base, registered[code], args.minimum_assets)
        for code in requested
    ]
    report = {
        "status": "pre2024_feasibility_complete",
        "screened_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_id": registry["study_id"],
        "base_config": str(base_path),
        "base_config_sha256": _sha256(base_path),
        "registry": str(registry_path),
        "registry_sha256": _sha256(registry_path),
        "requested_market_codes": requested,
        "screen_rule": {
            "minimum_eligible_assets": args.minimum_assets,
            "minimum_validation_origins": 1,
            "maximum_assets": 100,
            "eligibility_end": "2022-12-30",
            "outcome_boundary": "all loaded observations end before 2024-01-02",
        },
        "markets": markets,
        "eligible_market_codes": [row["code"] for row in markets if row["eligible"]],
        "excluded_market_codes": [row["code"] for row in markets if not row["eligible"]],
        "post2023_data_loaded": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="research_multimarket_base.yaml")
    parser.add_argument("--registry", default="research_multimarket_markets.json")
    parser.add_argument(
        "--output",
        default="research_output/prospective_multimarket/pre2024_smoke.json",
    )
    parser.add_argument("--minimum-assets", type=int, default=15)
    parser.add_argument("--markets", nargs="+", default=None)
    parser.add_argument("--allow-overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
