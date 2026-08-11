"""Freeze the FTSE 100 dual-base pooled protocol using data ending in 2023.

One shared set of residual-diffusion checkpoints supports two validation-
selected pools: the Gaussian-VAR-base pool (primary endpoint, transplant of
the locked S&P Phase 2F specification) and the Student-t-VAR-base pool
(prespecified secondary endpoint addressing the strong-baseline question).
All architecture, budget, and selection rules are transplanted from the
locked S&P protocol; nothing is tuned to FTSE outcomes.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

from config.config import load_config
from data.yf_loader import YFinanceDataPipeline
from diffusion.conditional_path import VARResidualPathDiffusion
from phase2c_benchmark import _checkpoint_payload
from research.freeze_phase2f import _file_sha256, _panel_fingerprint
from research.phase2f_pooling import _linear_pool
from sequences.baselines import Phase2ABaselines
from sequences.dataset import FactorStateSequenceBuilder
from sequences.evaluation import _energy_score


PRIMARY_BASE = "VAR-GARCH"
SECONDARY_BASE = "Student-t-VAR"
POOL_BASES = {
    "gaussian_base_pool": PRIMARY_BASE,
    "student_t_base_pool": SECONDARY_BASE,
}


def _select_weight_for_base(
    model, seed, config, splits, baselines, grid, base
):
    n_paths = config.temporal.path_validation_paths
    rows = {weight: [] for weight in grid}
    for origin in range(len(splits.validation)):
        context = splits.validation.context[origin]
        baseline_states = baselines.forecast(
            context,
            config.temporal.horizon,
            n_paths=n_paths,
            seed=50000 + origin * n_paths,
        ).states[base]
        diffusion = model.sample(
            context,
            n_paths=n_paths,
            seed=seed * 10000 + 60000 + origin * n_paths,
            sampling_steps=config.temporal.path_validation_steps,
        )
        target = splits.validation.target[origin]
        for weight in grid:
            rows[weight].append(
                _energy_score(
                    _linear_pool(diffusion, baseline_states, weight),
                    target,
                )
            )
    scores = {
        weight: float(np.mean(values)) for weight, values in rows.items()
    }
    selected = min(grid, key=lambda weight: (scores[weight], weight))
    return selected, scores


def run(args):
    output = Path(args.output)
    if output.exists() and not args.allow_overwrite:
        raise FileExistsError(
            f"Frozen FTSE protocol already exists: {output}. "
            "Refusing to overwrite it."
        )
    config = load_config(args.config)
    if pd.to_datetime(config.data.end_date) >= pd.Timestamp("2024-01-02"):
        raise ValueError("Freeze data must end before the evaluation period.")

    returns, market = YFinanceDataPipeline(config).load_all_data(
        max_stocks=args.max_stocks
    )
    splits = FactorStateSequenceBuilder(
        config,
        context_length=config.temporal.context_length,
        horizon=config.temporal.horizon,
        evaluation_stride=config.temporal.horizon,
        train_end_date=args.train_end,
        validation_end_date=args.validation_end,
        allow_empty_test=True,
    ).build(returns, market)
    baselines = Phase2ABaselines(config, splits)
    grid = sorted(set(args.pool_weights))
    seeds = args.seeds or config.temporal.validation_seeds
    checkpoint_dir = output.parent / "checkpoints"
    seed_reports = []
    for seed in seeds:
        print(f"Freezing FTSE dual-pool seed {seed}")
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
            training_steps=args.steps,
            selection_metric="sampled_path_energy",
        )
        pool_weights = {}
        validation_energies = {}
        for pool_name, base in POOL_BASES.items():
            weight, validation_scores = _select_weight_for_base(
                model, seed, config, splits, baselines, grid, base
            )
            pool_weights[pool_name] = weight
            validation_energies[pool_name] = {
                str(key): value
                for key, value in validation_scores.items()
            }
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_dir / f"ftse_frozen_seed{seed}.pt"
        torch.save(_checkpoint_payload(model, splits, seed), checkpoint)
        seed_reports.append(
            {
                "seed": seed,
                "best_step": model.best_step_,
                "best_validation_loss": model.best_validation_loss_,
                "best_validation_state_energy": model.best_selection_score_,
                "pool_weights": pool_weights,
                "validation_pool_energy_by_weight": validation_energies,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _file_sha256(checkpoint),
                "training_history": history,
            }
        )

    manifest = Path(config.data.universe_manifest)
    report = {
        "frozen_at": datetime.now().isoformat(),
        "status": "frozen_before_external_evaluation",
        "model": "FTSE100-Dual-Base-Validation-Pooled-Diffusion",
        "protocol": {
            "role": "one-time untouched-market external evaluation",
            "primary_market_unchanged": "S&P 500",
            "evaluation_market": "FTSE 100",
            "universe": (
                f"{config.data.universe} {config.data.universe_snapshot_date} "
                "snapshot, leading 100 eligible"
            ),
            "universe_manifest": str(manifest),
            "universe_manifest_sha256": _file_sha256(manifest),
            "market_benchmark": config.data.market_benchmark,
            "ticker_convention": config.data.ticker_convention,
            "hyperparameters_transplanted_from": (
                "locked S&P 500 Phase 2F protocol; no FTSE-specific tuning"
            ),
            "data_start": str(returns.index.min().date()),
            "data_end": str(returns.index.max().date()),
            "train_end": args.train_end,
            "pool_and_checkpoint_validation_start": str(
                splits.validation_returns.index.min().date()
            ),
            "pool_and_checkpoint_validation_end": str(
                splits.validation_returns.index.max().date()
            ),
            "evaluation_start": "2024-01-01",
            "evaluation_end": "latest complete session at one-time run",
            "n_assets": len(baselines.stocks),
            "asset_columns": list(baselines.stocks),
            "state_dim": splits.train_states.shape[1],
            "context_length": config.temporal.context_length,
            "horizon": config.temporal.horizon,
            "evaluation_stride": config.temporal.horizon,
            "ensemble_paths_per_seed": config.temporal.ensemble_paths,
            "seeds": seeds,
            "training_steps_requested": (
                args.steps or config.temporal.training_steps
            ),
            "checkpoint_selection": "2023 sampled-path state energy",
            "pool_weight_selection": "2023 pooled state energy per base",
            "pool_weight_grid": grid,
            "weight_zero_included": 0.0 in grid,
            "pool_bases": POOL_BASES,
            "preprocess_fit_end_date": (
                config.data.preprocess_fit_end_date
            ),
            "eligibility_end_date": config.data.eligibility_end_date,
            "pre2024_panel_fingerprint": _panel_fingerprint(
                returns, market
            ),
            "post2023_data_loaded": False,
        },
        "confirmatory_endpoints": {
            "primary_pool": "gaussian_base_pool",
            "primary_baseline": PRIMARY_BASE,
            "secondary_pool": "student_t_base_pool",
            "secondary_baseline": SECONDARY_BASE,
            "co_primary": [
                "state_energy_score",
                "state_rmse",
            ],
            "composite": (
                "geometric mean of pool/baseline ratios for the two "
                "co-primary losses, computed separately per pool"
            ),
            "success_rule": (
                "for each pool: composite < 1.0 and both co-primary point "
                "estimates do not exceed that pool's classical baseline; "
                "paired origin bootstrap intervals reported. The primary "
                "decision uses the Gaussian-VAR-base pool against VAR-GARCH "
                "(identical S&P Phase 2F rule); the Student-t-VAR-base pool "
                "against Student-t VAR is the prespecified strong-baseline "
                "secondary decision."
            ),
            "secondary": [
                "mean_factor_rmse",
                "logvol_factor_rmse",
                "state_variogram_score",
                "return_rmse_scaled",
                "daily_volatility_mae",
                "tail_quantile_error",
                "max_drawdown_error",
            ],
            "secondary_interpretation": (
                "exploratory with Benjamini-Hochberg adjusted paired p-values"
            ),
        },
        "seed_reports": seed_reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen FTSE protocol saved to {output}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_ftse100_freeze.yaml")
    parser.add_argument(
        "--output",
        default="research_output/ftse100_frozen/frozen_protocol.json",
    )
    parser.add_argument("--train-end", default="2022-12-30")
    parser.add_argument("--validation-end", default="2023-12-31")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--max-stocks", type=int, default=100)
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
