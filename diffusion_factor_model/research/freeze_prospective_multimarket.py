"""Freeze all eligible markets for the prospective strong-baseline study.

This runner is deliberately limited to observations ending in 2023. It trains
one shared residual-diffusion model per seed and market, selects both pool
weights on 2023 validation origins, and emits the single study protocol that
must be publicly timestamped before the one-time evaluation runner is used.
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

from config.config import load_config
from data.yf_loader import YFinanceDataPipeline
from diffusion.conditional_path import VARResidualPathDiffusion
from phase2c_benchmark import _checkpoint_payload
from research.freeze_ftse100_pools import POOL_BASES, _select_weight_for_base
from research.freeze_phase2f import _panel_fingerprint
from research.prospective_multimarket import (
    market_config,
    sha256,
    validate_pre2024_inputs,
)
from sequences.baselines import Phase2ABaselines
from sequences.dataset import FactorStateSequenceBuilder


def _validation_volatility_threshold(market, validation_windows):
    trailing = market.rolling(20, min_periods=20).std(ddof=1)
    values = []
    for dates in validation_windows.context_dates:
        value = trailing.loc[pd.Timestamp(dates[-1])]
        if np.isfinite(value):
            values.append(float(value))
    if not values:
        raise ValueError("No finite 2023 validation volatility signals.")
    return {
        "definition": (
            "median trailing-20-session benchmark-return standard deviation "
            "at the final context date of each 2023 validation origin"
        ),
        "threshold": float(np.median(values)),
        "n_validation_signals": len(values),
        "annualization": "none (classification is scale-invariant)",
    }


def _freeze_market(
    base_config,
    registered_market,
    smoke_market,
    output_root,
    seeds,
    steps,
    pool_weights,
):
    code = registered_market["code"]
    config = market_config(base_config, registered_market)
    if pd.to_datetime(config.data.end_date) >= pd.Timestamp("2024-01-02"):
        raise ValueError(f"{code}: freeze config crosses the outcome boundary.")
    returns, market = YFinanceDataPipeline(config).load_all_data(
        max_stocks=registered_market["max_stocks"]
    )
    if str(returns.index.max().date()) > "2023-12-31":
        raise ValueError(f"{code}: post-2023 observation loaded during freeze.")
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
    baselines = Phase2ABaselines(config, splits)
    checkpoint_dir = output_root / code / "checkpoints"
    seed_reports = []
    for seed in seeds:
        print(f"Freezing {code} seed {seed}")
        model = VARResidualPathDiffusion(
            config,
            state_dim=splits.train_states.shape[1],
            horizon=config.temporal.horizon,
            n_mean_factors=splits.latent_metadata["n_mean_factors"],
        )
        history = model.fit(
            splits.train_states.values,
            splits.train.context,
            splits.train.target,
            splits.validation.context,
            splits.validation.target,
            seed=seed,
            training_steps=steps,
            selection_metric="sampled_path_energy",
        )
        selected = {}
        validation_scores = {}
        for pool_name, baseline in POOL_BASES.items():
            weight, scores = _select_weight_for_base(
                model,
                seed,
                config,
                splits,
                baselines,
                pool_weights,
                baseline,
            )
            selected[pool_name] = float(weight)
            validation_scores[pool_name] = {
                str(key): float(value) for key, value in scores.items()
            }
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_dir / f"{code}_frozen_seed{seed}.pt"
        torch.save(_checkpoint_payload(model, splits, seed), checkpoint)
        seed_reports.append(
            {
                "seed": int(seed),
                "best_step": int(model.best_step_),
                "best_validation_loss": float(model.best_validation_loss_),
                "best_validation_state_energy": float(model.best_selection_score_),
                "pool_weights": selected,
                "validation_pool_energy_by_weight": validation_scores,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "training_history": history,
            }
        )

    manifest = Path(registered_market["universe_manifest"])
    provenance = Path(registered_market["universe_provenance"])
    return {
        "code": code,
        "name": registered_market["name"],
        "country": registered_market["country"],
        "universe_manifest": str(manifest),
        "universe_manifest_sha256": sha256(manifest),
        "universe_provenance": str(provenance),
        "universe_provenance_sha256": sha256(provenance),
        "market_benchmark": registered_market["market_benchmark"],
        "n_assets": len(baselines.stocks),
        "asset_columns": list(baselines.stocks),
        "data_start": str(returns.index.min().date()),
        "data_end": str(returns.index.max().date()),
        "train_end": "2022-12-30",
        "validation_start": str(splits.validation_returns.index.min().date()),
        "validation_end": str(splits.validation_returns.index.max().date()),
        "pre2024_panel_fingerprint": fingerprint,
        "volatility_regime": _validation_volatility_threshold(
            market, splits.validation
        ),
        "seed_reports": seed_reports,
        "post2023_data_loaded": False,
    }


def run(args):
    output = Path(args.output)
    if output.exists() and not args.allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite frozen protocol: {output}")
    base_path = Path(args.base_config)
    registry_path = Path(args.registry)
    smoke_path = Path(args.smoke)
    registry, smoke, registered, eligible, required, checks = (
        validate_pre2024_inputs(base_path, registry_path, smoke_path)
    )
    base = load_config(base_path)
    seeds = list(args.seeds or base.temporal.validation_seeds)
    pool_weights = sorted(set(args.pool_weights))
    if pool_weights != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise ValueError("The prospective study requires the frozen five-weight grid.")
    if len(seeds) != 3:
        raise ValueError("The confirmatory freeze requires exactly three seeds.")
    output_root = output.parent
    market_reports = []
    for smoke_market in eligible:
        market_reports.append(
            _freeze_market(
                base,
                registered[smoke_market["code"]],
                smoke_market,
                output_root,
                seeds,
                args.steps,
                pool_weights,
            )
        )

    evaluation_runner = Path(
        "diffusion_factor_model/research/evaluate_prospective_multimarket.py"
    )
    helper = Path("diffusion_factor_model/research/prospective_multimarket.py")
    design = Path("PROSPECTIVE_MULTIMARKET_DESIGN.md")
    report = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_post2023_evaluation",
        "study_id": registry["study_id"],
        "protocol": {
            "role": "prospective multi-market strong-baseline evaluation",
            "eligible_market_codes": [row["code"] for row in eligible],
            "excluded_market_codes": smoke["excluded_market_codes"],
            "n_markets": len(eligible),
            "required_individual_market_passes": required,
            "train_end": "2022-12-30",
            "validation_end": "2023-12-31",
            "evaluation_start": "2024-01-01",
            "evaluation_end": "latest complete session at one-time run",
            "seeds": seeds,
            "training_steps_requested": int(
                args.steps or base.temporal.training_steps
            ),
            "pool_weight_grid": pool_weights,
            "pool_bases": POOL_BASES,
            "primary_candidate": "student_t_base_pool",
            "strong_baselines": ["Gaussian-VAR", "Student-t-VAR"],
            "state_metrics": ["state_energy_score", "state_rmse"],
            "state_market_rule": (
                "composite below 1 versus both strong baselines and none of "
                "the four component ratios above 1"
            ),
            "population_rule": (
                "upper endpoint of equal-market 95% Student-t interval for "
                "mean log worse-baseline ratio below 0, plus at least 5 of "
                "6 individual market passes"
            ),
            "key_secondary": (
                "mean-within-path covariance Frobenius error, scaled by the "
                "realized covariance Frobenius norm"
            ),
            "post2023_data_loaded": False,
        },
        "input_checks": checks,
        "frozen_hashes": {
            "base_config_sha256": sha256(base_path),
            "registry_sha256": sha256(registry_path),
            "pre2024_smoke_sha256": sha256(smoke_path),
            "design_sha256": sha256(design),
            "freeze_runner_sha256": sha256(__file__),
            "evaluation_runner_sha256": sha256(evaluation_runner),
            "shared_helper_sha256": sha256(helper),
        },
        "markets": market_reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen prospective protocol saved to {output}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="research_multimarket_base.yaml")
    parser.add_argument("--registry", default="research_multimarket_markets.json")
    parser.add_argument(
        "--smoke",
        default="research_output/prospective_multimarket/pre2024_smoke.json",
    )
    parser.add_argument(
        "--output",
        default="research_output/prospective_multimarket/frozen_protocol.json",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--pool-weights",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
