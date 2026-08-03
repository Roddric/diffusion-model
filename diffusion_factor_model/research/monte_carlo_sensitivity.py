"""Post-hoc path-count stability audit for the frozen Phase 2F state forecast."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import load_config
from diffusion.conditional_path import VARResidualPathDiffusion
from dynamics.var import LatentVAR
from phase2c_benchmark import _restore_checkpoint
from research.confirm_phase2f import (
    _load_frozen_history_with_current_extension,
    _sha256,
)
from research.freeze_phase2f import _panel_fingerprint
from research.phase2f_pooling import _linear_pool
from sequences.dataset import FactorStateSequenceBuilder
from sequences.evaluation import _energy_score


MODEL_NAME = "Phase2F-Validation-Pooled-Diffusion"
METRICS = ("state_energy_score", "state_rmse")


def _score(paths, target):
    return {
        "state_energy_score": _energy_score(paths, target),
        "state_rmse": float(
            np.sqrt(np.mean((paths.mean(axis=0) - target) ** 2))
        ),
    }


def _aggregate(rows):
    return {
        metric: float(np.mean([row[metric] for row in rows]))
        for metric in METRICS
    }


def _summarize_replicates(replicates, path_counts):
    output = {}
    for count in path_counts:
        count_rows = [row[str(count)] for row in replicates]
        methods = count_rows[0]
        output[str(count)] = {}
        for method in methods:
            output[str(count)][method] = {}
            for metric in METRICS:
                values = np.asarray(
                    [row[method][metric] for row in count_rows], dtype=float
                )
                output[str(count)][method][metric] = {
                    "mean": float(values.mean()),
                    "std_across_mc_replicates": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
        for baseline in ("Gaussian-VAR", "Student-t-VAR"):
            key = f"ratios_vs_{baseline}"
            output[str(count)][key] = {}
            for metric in METRICS:
                ratios = np.asarray(
                    [
                        row[MODEL_NAME][metric] / row[baseline][metric]
                        for row in count_rows
                    ]
                )
                output[str(count)][key][metric] = {
                    "mean": float(ratios.mean()),
                    "std_across_mc_replicates": float(ratios.std()),
                    "min": float(ratios.min()),
                    "max": float(ratios.max()),
                }
    return output


def run(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Monte Carlo audit: {output}")
    frozen = json.loads(Path(args.frozen_protocol).read_text(encoding="utf-8"))
    config = load_config(args.config)
    frozen_config = load_config(args.freeze_config)
    returns, market = _load_frozen_history_with_current_extension(
        frozen_config, config, args.max_stocks
    )
    pre2024 = returns.loc[returns.index < "2024-01-01"]
    if _panel_fingerprint(
        pre2024, market.reindex(pre2024.index)
    ) != frozen["protocol"]["pre2024_panel_fingerprint"]:
        raise ValueError("Frozen pre-2024 panel fingerprint mismatch.")

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
    gaussian_var = LatentVAR().fit(splits.train_states.values)
    models = []
    for frozen_seed in frozen["seed_reports"]:
        checkpoint = Path(frozen_seed["checkpoint"])
        if _sha256(checkpoint) != frozen_seed["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
        model = VARResidualPathDiffusion(
            config,
            state_dim=splits.train_states.shape[1],
            horizon=config.temporal.horizon,
            n_mean_factors=splits.latent_metadata["n_mean_factors"],
        )
        _restore_checkpoint(model, splits, checkpoint)
        models.append((frozen_seed, model))

    path_counts = sorted(set(args.path_counts))
    if path_counts[0] < 2:
        raise ValueError("Path counts must be at least two.")
    max_paths = max(path_counts)
    replicate_reports = []
    for replicate in range(args.replicates):
        print(f"Monte Carlo replicate {replicate + 1}/{args.replicates}")
        scores = {
            str(count): {
                MODEL_NAME: [],
                "Gaussian-VAR": [],
                "Student-t-VAR": [],
            }
            for count in path_counts
        }
        for origin in range(len(splits.test)):
            context = splits.test.context[origin]
            target = splits.test.target[origin]
            seed_root = 910000 + replicate * 100000 + origin * 1000
            gaussian_paths = np.stack(
                [
                    gaussian_var.simulate(
                        config.temporal.horizon,
                        seed=seed_root + path,
                        initial_state=context[-1],
                    )
                    for path in range(max_paths)
                ]
            )
            student_paths = np.stack(
                [
                    gaussian_var.simulate_student_t(
                        config.temporal.horizon,
                        seed=seed_root + 50000 + path,
                        initial_state=context[-1],
                    )
                    for path in range(max_paths)
                ]
            )
            diffusion_by_seed = []
            for frozen_seed, model in models:
                diffusion_by_seed.append(
                    (
                        frozen_seed,
                        model.sample(
                            context,
                            n_paths=max_paths,
                            seed=(
                                frozen_seed["seed"] * 1000000
                                + seed_root
                            ),
                        ),
                    )
                )
            for count in path_counts:
                key = str(count)
                scores[key]["Gaussian-VAR"].append(
                    _score(gaussian_paths[:count], target)
                )
                scores[key]["Student-t-VAR"].append(
                    _score(student_paths[:count], target)
                )
                seed_scores = []
                for frozen_seed, diffusion_paths in diffusion_by_seed:
                    pooled = _linear_pool(
                        diffusion_paths[:count],
                        gaussian_paths[:count],
                        frozen_seed["selected_diffusion_weight"],
                    )
                    seed_scores.append(_score(pooled, target))
                scores[key][MODEL_NAME].append(
                    {
                        metric: float(
                            np.mean([row[metric] for row in seed_scores])
                        )
                        for metric in METRICS
                    }
                )
        replicate_reports.append(
            {
                str(count): {
                    method: _aggregate(rows)
                    for method, rows in scores[str(count)].items()
                }
                for count in path_counts
            }
        )

    report = {
        "analyzed_at": datetime.now().isoformat(),
        "status": "posthoc_monte_carlo_sensitivity",
        "interpretation_boundary": (
            "Frozen models and validation-selected weights; post-hoc path-count "
            "sensitivity on a consumed holdout, not a new confirmation."
        ),
        "protocol": {
            "path_counts": path_counts,
            "replicates": args.replicates,
            "nested_paths_within_replicate": True,
            "n_origins": len(splits.test),
            "horizon": config.temporal.horizon,
            "seeds": [row["seed"] for row, _ in models],
            "weights_reselected": False,
        },
        "replicate_metrics": replicate_reports,
        "summary": _summarize_replicates(
            replicate_reports, path_counts
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research_sp500_confirmation.yaml")
    parser.add_argument("--freeze-config", default="research_sp500_freeze.yaml")
    parser.add_argument(
        "--frozen-protocol",
        default="research_output/sp500_frozen/frozen_protocol.json",
    )
    parser.add_argument(
        "--output",
        default=(
            "research_output/sp500_confirmation/"
            "posthoc_monte_carlo_sensitivity.json"
        ),
    )
    parser.add_argument(
        "--path-counts", type=int, nargs="+", default=[20, 50, 100]
    )
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--max-stocks", type=int, default=100)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
