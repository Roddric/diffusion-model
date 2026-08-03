"""Freeze the Phase 2F model and protocol using data ending in 2023."""

import argparse
import hashlib
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
from research.phase2f_pooling import _select_weight
from sequences.baselines import Phase2ABaselines
from sequences.dataset import FactorStateSequenceBuilder


def _panel_fingerprint(returns, market):
    digest = hashlib.sha256()
    for values in (returns, market.to_frame("market")):
        hashed = pd.util.hash_pandas_object(
            values, index=True
        ).values.tobytes()
        digest.update(hashed)
    return digest.hexdigest()


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    output = Path(args.output)
    if output.exists() and not args.allow_overwrite:
        raise FileExistsError(
            f"Frozen protocol already exists: {output}. "
            "Refusing to overwrite it."
        )
    config = load_config(args.config)
    if pd.to_datetime(config.data.end_date) >= pd.Timestamp("2024-01-02"):
        raise ValueError("Freeze data must end before the confirmation period.")

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
        print(f"Freezing Phase 2F seed {seed}")
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
        weight, validation_scores = _select_weight(
            model, seed, config, splits, baselines, grid
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_dir / f"phase2f_frozen_seed{seed}.pt"
        torch.save(_checkpoint_payload(model, splits, seed), checkpoint)
        seed_reports.append(
            {
                "seed": seed,
                "best_step": model.best_step_,
                "best_validation_loss": model.best_validation_loss_,
                "best_validation_state_energy": (
                    model.best_selection_score_
                ),
                "selected_diffusion_weight": weight,
                "validation_pool_energy_by_weight": {
                    str(key): value
                    for key, value in validation_scores.items()
                },
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _file_sha256(checkpoint),
                "training_history": history,
            }
        )

    manifest = Path(config.data.universe_manifest)
    report = {
        "frozen_at": datetime.now().isoformat(),
        "status": "frozen_before_confirmation",
        "model": "Phase2F-Validation-Pooled-Diffusion",
        "protocol": {
            "universe": "S&P 500 December 2023 snapshot, leading 100 eligible",
            "universe_manifest": str(manifest),
            "universe_manifest_sha256": _file_sha256(manifest),
            "data_start": str(returns.index.min().date()),
            "data_end": str(returns.index.max().date()),
            "train_end": args.train_end,
            "pool_and_checkpoint_validation_start": str(
                splits.validation_returns.index.min().date()
            ),
            "pool_and_checkpoint_validation_end": str(
                splits.validation_returns.index.max().date()
            ),
            "confirmation_start": "2024-01-01",
            "confirmation_end": "latest complete session at one-time run",
            "n_assets": len(baselines.stocks),
            "state_dim": splits.train_states.shape[1],
            "context_length": config.temporal.context_length,
            "horizon": config.temporal.horizon,
            "evaluation_stride": config.temporal.horizon,
            "ensemble_paths_per_seed": config.temporal.ensemble_paths,
            "seeds": seeds,
            "training_steps_requested": (
                args.steps or config.temporal.training_steps
            ),
            "checkpoint_selection": "2023 reconstructed-state energy",
            "pool_weight_selection": "2023 state energy",
            "pool_weight_grid": grid,
            "weight_zero_included": 0.0 in grid,
            "preprocess_fit_end_date": (
                config.data.preprocess_fit_end_date
            ),
            "eligibility_end_date": config.data.eligibility_end_date,
            "pre2024_panel_fingerprint": _panel_fingerprint(
                returns, market
            ),
            "confirmation_data_loaded": False,
        },
        "confirmatory_endpoints": {
            "co_primary": [
                "state_energy_score",
                "state_rmse",
            ],
            "composite": (
                "geometric mean of pooled/VAR-GARCH ratios for co-primary "
                "losses"
            ),
            "success_rule": (
                "composite < 1.0 and both co-primary point estimates do not "
                "exceed VAR-GARCH; paired origin bootstrap intervals reported"
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
    print(f"Frozen protocol saved to {output}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_sp500_freeze.yaml")
    parser.add_argument(
        "--output",
        default="research_output/sp500_frozen/frozen_protocol.json",
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

