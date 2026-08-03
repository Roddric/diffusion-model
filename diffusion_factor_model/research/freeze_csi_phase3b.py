"""Freeze the S&P-selected Phase 3B specification for CSI replication."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

torch.set_num_threads(1)

from config.config import load_config
from data.loader import DataPipeline
from diffusion.conditional_path import VARResidualPathDiffusion
from phase2c_benchmark import _checkpoint_payload
from research.freeze_phase2f import (
    _file_sha256,
    _panel_fingerprint,
)
from sequences.baselines import Phase2ABaselines
from sequences.dataset import FactorStateSequenceBuilder


def _seed_weights(report, field):
    weights = {}
    for seed, value in report.items():
        weights[int(seed)] = float(value)
    if not weights:
        raise ValueError(f"No weights found in {field}.")
    return weights


def run(args):
    output = Path(args.output)
    if output.exists() and not args.allow_overwrite:
        raise FileExistsError(
            f"Frozen CSI replication protocol already exists: {output}."
        )
    config = load_config(args.config)
    if config.data.source != "akshare":
        raise ValueError("CSI external replication requires AKShare.")
    if pd.to_datetime(config.data.end_date) >= pd.Timestamp("2024-01-02"):
        raise ValueError("CSI freeze data must end before 2024.")

    phase3b_path = Path(args.phase3b_report)
    phase2f_path = Path(args.phase2f_report)
    phase3b = json.loads(phase3b_path.read_text(encoding="utf-8"))
    phase2f = json.loads(phase2f_path.read_text(encoding="utf-8"))
    if phase3b["validation"]["selected_base"] != "Student-t-VAR":
        raise ValueError("The fixed Phase 3B base is not Student-t VAR.")
    candidate_weights = _seed_weights(
        phase3b["validation"]["selected_diffusion_weights"],
        "Phase 3B selected weights",
    )
    incumbent_weights = {
        int(row["seed"]): float(row["selected_diffusion_weight"])
        for row in phase2f["seed_reports"]
    }
    if set(candidate_weights) != set(incumbent_weights):
        raise ValueError("Candidate and incumbent seed sets differ.")

    returns, market = DataPipeline(config).load_all_data(
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
    checkpoint_dir = output.parent / "checkpoints"
    seed_reports = []
    for seed in sorted(candidate_weights):
        print(f"Freezing CSI Phase 3B seed {seed}")
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
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_dir / f"csi_phase3b_seed{seed}.pt"
        torch.save(_checkpoint_payload(model, splits, seed), checkpoint)
        seed_reports.append(
            {
                "seed": seed,
                "best_step": model.best_step_,
                "best_validation_loss": model.best_validation_loss_,
                "best_validation_state_energy": model.best_selection_score_,
                "candidate_diffusion_weight": candidate_weights[seed],
                "incumbent_diffusion_weight": incumbent_weights[seed],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _file_sha256(checkpoint),
                "training_history": history,
            }
        )

    manifest = Path(config.data.universe_manifest)
    report = {
        "frozen_at": datetime.now().isoformat(),
        "status": "frozen_before_external_replication",
        "model": "Phase3B-Student-t-Base-Diffusion-Pool",
        "protocol": {
            "role": "one-time cross-market external replication",
            "primary_market_unchanged": "S&P 500",
            "replication_market": "CSI 300",
            "universe": (
                "December 2023 CSI 300 snapshot; first 100 manifest entries "
                "screened using information through 2020"
            ),
            "universe_manifest": str(manifest),
            "universe_manifest_sha256": _file_sha256(manifest),
            "freeze_config": args.config,
            "freeze_config_sha256": _file_sha256(args.config),
            "data_start": str(returns.index.min().date()),
            "data_end": str(returns.index.max().date()),
            "train_end": args.train_end,
            "checkpoint_validation_start": str(
                splits.validation_returns.index.min().date()
            ),
            "checkpoint_validation_end": str(
                splits.validation_returns.index.max().date()
            ),
            "external_replication_start": "2024-01-01",
            "external_replication_end": (
                "latest complete session at one-time run"
            ),
            "n_assets": len(baselines.stocks),
            "asset_columns": list(baselines.stocks),
            "state_dim": splits.train_states.shape[1],
            "context_length": config.temporal.context_length,
            "horizon": config.temporal.horizon,
            "evaluation_stride": config.temporal.horizon,
            "ensemble_paths_per_seed": config.temporal.ensemble_paths,
            "seeds": sorted(candidate_weights),
            "training_steps_requested": (
                args.steps or config.temporal.training_steps
            ),
            "checkpoint_selection": "2023 sampled-path state energy",
            "candidate_classical_base": "Student-t-VAR",
            "incumbent_classical_base": "Gaussian-VAR",
            "candidate_diffusion_weights": candidate_weights,
            "incumbent_diffusion_weights": incumbent_weights,
            "weights_reselected_for_csi": False,
            "preprocess_fit_end_date": (
                config.data.preprocess_fit_end_date
            ),
            "eligibility_end_date": config.data.eligibility_end_date,
            "pre2024_panel_fingerprint": _panel_fingerprint(
                returns, market
            ),
            "post2023_data_loaded": False,
            "s_and_p_phase3b_report": str(phase3b_path),
            "s_and_p_phase3b_report_sha256": _file_sha256(phase3b_path),
            "s_and_p_phase2f_report": str(phase2f_path),
            "s_and_p_phase2f_report_sha256": _file_sha256(phase2f_path),
        },
        "external_replication_endpoints": {
            "co_primary": ["state_energy_score", "state_rmse"],
            "state_composite": (
                "geometric mean of candidate/incumbent ratios for "
                "state energy and state RMSE"
            ),
            "return_composite_metrics": [
                "return_energy_score",
                "return_variogram_score",
                "daily_volatility_mae",
                "tail_quantile_error",
                "max_drawdown_error",
            ],
            "success_rule": (
                "state composite < 1.0; neither co-primary state loss exceeds "
                "the Phase 2F incumbent; paired one-sided state-energy "
                "bootstrap p < 0.10; return composite <= 1.02"
            ),
            "strong_baseline": "Student-t-VAR",
        },
        "seed_reports": seed_reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen CSI external protocol saved to {output}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="research_csi300_freeze.yaml"
    )
    parser.add_argument(
        "--phase3b-report",
        default="research_output/sp500/phase3b_state_pool.json",
    )
    parser.add_argument(
        "--phase2f-report",
        default="research_output/sp500/phase2f_development.json",
    )
    parser.add_argument(
        "--output",
        default="research_output/csi300_external/frozen_protocol.json",
    )
    parser.add_argument("--train-end", default="2022-12-30")
    parser.add_argument("--validation-end", default="2023-12-31")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--max-stocks", type=int, default=100)
    parser.add_argument("--allow-overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

