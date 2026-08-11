"""Post-hoc calibration and overlapping-origin power audit of the locked result.

Three diagnostics on the frozen Phase 2F specification, none of which changes
the locked decision or performs model selection:

1. Exact reproduction of the locked 29-origin score, with saved forecast paths.
2. Calibration diagnostics on the locked origins: rank (PIT) histograms of the
   observed state inside each ensemble, and an energy-score decomposition into
   ensemble-mean distance and ensemble spread terms.
3. Overlapping-origin scoring across the full 2024-2026 window at a fixed
   stride, with Newey-West HAC and circular moving-block inference, to increase
   statistical power against the locked 29 non-overlapping origins.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import chisquare

from config.config import load_config
from diffusion.conditional_path import VARResidualPathDiffusion
from phase2b_benchmark import reconstruct_paths
from phase2c_benchmark import _restore_checkpoint, _score_baselines
from research.confirm_phase2f import (
    _load_frozen_history_with_current_extension,
    _sha256,
)
from research.freeze_phase2f import _panel_fingerprint
from research.market_selection import (
    PRIMARY_SELECTION_METRICS,
    _aggregate_seed_scores,
)
from research.phase2f_pooling import MODEL_NAME, _linear_pool
from research.robustness import dependence_robust_comparison
from sequences.baselines import Phase2ABaselines
from sequences.dataset import FactorStateSequenceBuilder
from sequences.evaluation import PathForecastEvaluator, _energy_score


CALIBRATION_METHODS = (MODEL_NAME, "Gaussian-VAR", "Student-t-VAR")


def _energy_terms(paths, target):
    """Return the normalized distance-to-observation and spread terms."""
    samples = paths.reshape(len(paths), -1)
    observed = target.reshape(-1)
    scale = np.sqrt(len(observed))
    first = np.linalg.norm(samples - observed, axis=1).mean() / scale
    pairwise = np.linalg.norm(
        samples[:, None, :] - samples[None, :, :], axis=2
    ).mean() / scale
    return float(first), float(pairwise)


def _observation_ranks(paths, target):
    """Integer rank of each observation within its ensemble.

    For an ensemble of M paths the rank counts ensemble members strictly
    below the observation, so it lies in {0, ..., M}. Exact ties add half the
    tied count, which is conservative; ties have measure zero in the
    continuous state space. Under calibration the ranks are exchangeable and
    therefore uniform over the M + 1 possible values.
    """
    below = (paths < target).sum(axis=0)
    equal = (paths == target).sum(axis=0)
    ranks = below + (equal > 0) * np.ceil(0.5 * equal)
    return ranks.astype(int), len(paths)


def _rank_histogram(ranks, ensemble_size):
    """Talagrand rank histogram with one bin per possible rank."""
    bins = ensemble_size + 1
    counts = np.bincount(ranks.reshape(-1), minlength=bins)[:bins]
    expected = np.full(bins, len(ranks) / bins)
    statistic, p_value = chisquare(counts, expected)
    # U-shaped under-dispersion puts mass on the two edge bins; over-dispersion
    # concentrates it in the middle. Report both for interpretation.
    edge_share = float((counts[0] + counts[-1]) / counts.sum())
    return {
        "ensemble_size": int(ensemble_size),
        "bins": int(bins),
        "counts": [int(value) for value in counts],
        "expected_per_bin": float(expected[0]),
        "chi_square_statistic": float(statistic),
        "chi_square_p_uniform": float(p_value),
        "edge_bin_share": edge_share,
        "uniform_edge_bin_share": float(2.0 / bins),
        "max_abs_deviation_from_uniform": float(
            np.max(np.abs(counts - expected[0])) / expected[0]
        ),
        "n_values": int(len(ranks)),
    }


def _score_locked_origins(
    config,
    splits,
    baselines,
    evaluator,
    n_paths,
    frozen_seeds,
):
    """Replay the locked scoring exactly, retaining pooled state paths."""
    baseline_scores, evaluation_rows, origins = _score_baselines(
        config,
        splits,
        baselines,
        evaluator,
        n_paths,
        len(splits.test),
    )
    # Per-origin, per-seed pooled ensembles: calibration diagnostics must use
    # the same 20-path ensembles that the locked score averaged over.
    pooled_paths_by_origin = [[] for _ in range(len(splits.test))]
    seed_scores = []
    timing = {"diffusion_sampling_seconds": []}
    for frozen_seed in frozen_seeds:
        seed = frozen_seed["seed"]
        checkpoint = Path(frozen_seed["checkpoint"])
        if _sha256(checkpoint) != frozen_seed["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch for seed {seed}.")
        model = VARResidualPathDiffusion(
            config,
            state_dim=splits.train_states.shape[1],
            horizon=config.temporal.horizon,
            n_mean_factors=splits.latent_metadata["n_mean_factors"],
        )
        _restore_checkpoint(model, splits, checkpoint)
        weight = frozen_seed["selected_diffusion_weight"]
        scores = []
        for origin, row in enumerate(evaluation_rows):
            start = time.perf_counter()
            diffusion_paths = model.sample(
                row["context"],
                n_paths=n_paths,
                seed=seed * 10000 + 80000 + origin * n_paths,
            )
            timing["diffusion_sampling_seconds"].append(
                time.perf_counter() - start
            )
            states = _linear_pool(
                diffusion_paths,
                row["baseline_state_paths"]["VAR-GARCH"],
                weight,
            )
            generated_returns = reconstruct_paths(
                splits,
                states,
                row["innovations"],
                baselines.stocks,
            )
            scores.append(
                evaluator.score(
                    states,
                    generated_returns,
                    row["target_states"],
                    row["target_returns"],
                )
            )
            pooled_paths_by_origin[origin].append(states)
        seed_scores.append(scores)
    return (
        baseline_scores,
        evaluation_rows,
        origins,
        pooled_paths_by_origin,
        seed_scores,
        timing,
    )


def run(args):
    output = Path(args.output)
    if output.exists() and not args.allow_overwrite:
        raise FileExistsError(f"Audit output already exists: {output}.")

    frozen_path = Path(args.frozen_protocol)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen["status"] != "frozen_before_confirmation":
        raise ValueError("Protocol is not in the frozen pre-confirmation state.")
    locked_result = json.loads(
        Path(args.locked_result_reference).read_text(encoding="utf-8")
    )
    locked_composite = locked_result["confirmatory_decision"]["composite_ratio"]

    config = load_config(args.config)
    frozen_config = load_config(args.freeze_config)
    returns, market = _load_frozen_history_with_current_extension(
        frozen_config, config, args.max_stocks
    )
    pre2024_returns = returns.loc[returns.index < "2024-01-01"]
    pre2024_market = market.reindex(pre2024_returns.index)
    actual_fingerprint = _panel_fingerprint(pre2024_returns, pre2024_market)
    if actual_fingerprint != frozen["protocol"]["pre2024_panel_fingerprint"]:
        raise ValueError("Pre-2024 panel fingerprint changed; refusing audit.")
    manifest = Path(config.data.universe_manifest)
    if _sha256(manifest) != frozen["protocol"]["universe_manifest_sha256"]:
        raise ValueError("Universe manifest hash differs from frozen protocol.")

    def _build_splits(evaluation_stride):
        return FactorStateSequenceBuilder(
            config,
            context_length=config.temporal.context_length,
            horizon=config.temporal.horizon,
            evaluation_stride=evaluation_stride,
            train_end_date=frozen["protocol"]["train_end"],
            validation_end_date=frozen["protocol"][
                "pool_and_checkpoint_validation_end"
            ],
        ).build(returns, market)

    splits = _build_splits(config.temporal.horizon)
    baselines = Phase2ABaselines(config, splits)
    evaluator = PathForecastEvaluator(
        splits.latent_metadata["n_mean_factors"]
    )
    n_paths = frozen["protocol"]["ensemble_paths_per_seed"]

    (
        baseline_scores,
        evaluation_rows,
        origins,
        pooled_paths_by_origin,
        seed_scores,
        timing,
    ) = _score_locked_origins(
        config,
        splits,
        baselines,
        evaluator,
        n_paths,
        frozen["seed_reports"],
    )

    pooled_by_origin = _aggregate_seed_scores(seed_scores)
    ratios = {
        metric: (
            np.mean(pooled_by_origin[metric])
            / np.mean(
                [row[metric] for row in baseline_scores["VAR-GARCH"]]
            )
        )
        for metric in PRIMARY_SELECTION_METRICS
    }
    composite = float(np.exp(np.mean(np.log(list(ratios.values())))))
    if composite != locked_composite:
        raise ValueError(
            "Locked composite not reproduced exactly: "
            f"expected {locked_composite}, got {composite}."
        )

    # --- Calibration diagnostics on the locked non-overlapping origins ---
    ranks_by_method = {name: [] for name in CALIBRATION_METHODS}
    ensemble_size_by_method = {}
    energy_terms_by_method = {
        name: {"distance_to_observation": [], "spread": []}
        for name in CALIBRATION_METHODS
    }
    for origin, row in enumerate(evaluation_rows):
        ensembles_by_method = {
            MODEL_NAME: pooled_paths_by_origin[origin],
            "Gaussian-VAR": [row["baseline_state_paths"]["VAR-GARCH"]],
            "Student-t-VAR": [row["baseline_state_paths"]["Student-t-VAR"]],
        }
        for name, ensembles in ensembles_by_method.items():
            for paths in ensembles:
                ranks, ensemble_size = _observation_ranks(
                    paths, row["target_states"]
                )
                ranks_by_method[name].append(ranks.reshape(-1))
                ensemble_size_by_method[name] = ensemble_size
                first_term, spread_term = _energy_terms(
                    paths, row["target_states"]
                )
                energy_terms_by_method[name][
                    "distance_to_observation"
                ].append(first_term)
                energy_terms_by_method[name]["spread"].append(spread_term)
    calibration = {
        "definition": (
            "Rank (Talagrand) diagnostics pool all locked origins, horizons, "
            "and state dimensions. For an ensemble of M paths the observation "
            "rank lies in {0,...,M}; a calibrated ensemble yields a uniform "
            "rank histogram. Edge-bin excess indicates under-dispersion "
            "(the observation falls outside the ensemble). Pool diagnostics "
            "use the same per-seed 20-path ensembles the locked score "
            "averages over. Energy terms use the same sqrt(H*d) "
            "normalization as the state energy score: "
            "energy = distance_to_observation - 0.5 * spread."
        ),
        "n_origins": len(evaluation_rows),
        "rank_histograms": {
            name: _rank_histogram(
                np.concatenate(ranks_by_method[name]),
                ensemble_size_by_method[name],
            )
            for name in CALIBRATION_METHODS
        },
        "energy_decomposition": {
            name: {
                term: float(np.mean(values))
                for term, values in terms.items()
            }
            for name, terms in energy_terms_by_method.items()
        },
        "energy_recomposition_check": {
            name: float(
                np.mean(terms["distance_to_observation"])
                - 0.5 * np.mean(terms["spread"])
            )
            for name, terms in energy_terms_by_method.items()
        },
    }

    # --- Overlapping-origin power audit ---
    stride = int(args.stride)
    if stride < 1 or stride >= config.temporal.horizon:
        raise ValueError("stride must lie in [1, horizon).")
    overlap_start = time.perf_counter()
    overlapping_splits = _build_splits(stride)
    overlapping_baselines = Phase2ABaselines(config, overlapping_splits)
    overlap_baseline_scores, overlap_rows, overlap_origins = _score_baselines(
        config,
        overlapping_splits,
        overlapping_baselines,
        evaluator,
        n_paths,
        len(overlapping_splits.test),
    )
    checkpoint_lookup = {
        row["seed"]: row for row in frozen["seed_reports"]
    }
    overlap_seed_scores = []
    overlap_sampling_seconds = []
    for frozen_seed in frozen["seed_reports"]:
        seed = frozen_seed["seed"]
        frozen_row = checkpoint_lookup[seed]
        checkpoint = Path(frozen_row["checkpoint"])
        model = VARResidualPathDiffusion(
            config,
            state_dim=overlapping_splits.train_states.shape[1],
            horizon=config.temporal.horizon,
            n_mean_factors=overlapping_splits.latent_metadata[
                "n_mean_factors"
            ],
        )
        _restore_checkpoint(model, overlapping_splits, checkpoint)
        weight = frozen_row["selected_diffusion_weight"]
        scores = []
        for origin, row in enumerate(overlap_rows):
            start = time.perf_counter()
            diffusion_paths = model.sample(
                row["context"],
                n_paths=n_paths,
                seed=seed * 10000 + 90000 + origin * n_paths,
            )
            overlap_sampling_seconds.append(time.perf_counter() - start)
            states = _linear_pool(
                diffusion_paths,
                row["baseline_state_paths"]["VAR-GARCH"],
                weight,
            )
            generated_returns = reconstruct_paths(
                overlapping_splits,
                states,
                row["innovations"],
                overlapping_baselines.stocks,
            )
            scores.append(
                evaluator.score(
                    states,
                    generated_returns,
                    row["target_states"],
                    row["target_returns"],
                )
            )
        overlap_seed_scores.append(scores)
    overlap_pooled = _aggregate_seed_scores(overlap_seed_scores)
    overlap_ratios = {
        metric: (
            np.mean(overlap_pooled[metric])
            / np.mean(
                [row[metric] for row in overlap_baseline_scores["VAR-GARCH"]]
            )
        )
        for metric in PRIMARY_SELECTION_METRICS
    }
    overlap_composite = float(
        np.exp(np.mean(np.log(list(overlap_ratios.values()))))
    )
    overlap_comparisons = {}
    for baseline_index, (baseline, display) in enumerate(
        (
            ("Gaussian-VAR", "vs_gaussian_var"),
            ("Student-t-VAR", "vs_student_t_var"),
        )
    ):
        overlap_comparisons[display] = {
            metric: dependence_robust_comparison(
                overlap_pooled[metric],
                [row[metric] for row in overlap_baseline_scores[baseline]],
                block_lengths=(5, 10, 20),
                seed=20260811 + baseline_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(PRIMARY_SELECTION_METRICS)
        }
    overlap_summary = {
        metric: {
            "pool": float(np.mean(overlap_pooled[metric])),
            **{
                method: float(
                    np.mean([row[metric] for row in rows])
                )
                for method, rows in overlap_baseline_scores.items()
            },
        }
        for metric in PRIMARY_SELECTION_METRICS
    }

    compute_cost = {
        "locked_origins": {
            "n_origins": len(evaluation_rows),
            "mean_diffusion_sampling_seconds_per_origin_per_seed": float(
                np.mean(timing["diffusion_sampling_seconds"])
            ),
        },
        "overlapping_origins": {
            "stride": stride,
            "n_origins": len(overlap_rows),
            "mean_baseline_forecast_note": (
                "baseline forecasts for all five methods are produced by one "
                "Phase2ABaselines.forecast call per origin"
            ),
            "mean_diffusion_sampling_seconds_per_origin_per_seed": float(
                np.mean(overlap_sampling_seconds)
            ),
            "wall_seconds_for_overlapping_stage": float(
                time.perf_counter() - overlap_start
            ),
        },
    }

    report = {
        "analyzed_at": datetime.now().isoformat(),
        "report_role": "posthoc_robustness_audit",
        "status": "posthoc_calibration_power_audit",
        "locked_result_reference": args.locked_result_reference,
        "frozen_protocol": str(frozen_path),
        "frozen_protocol_sha256": _sha256(frozen_path),
        "interpretation": (
            "Diagnostics only. No model, checkpoint, or weight selection is "
            "performed; the locked decision is unchanged. Overlapping origins "
            "share target days, so inference uses HAC and moving-block "
            "bootstrap only."
        ),
        "reproduction_checks": {
            "composite_ratio_exact_match": bool(composite == locked_composite),
            "composite_ratio": composite,
            "locked_composite_ratio": locked_composite,
            "co_primary_ratios_vs_var_garch": ratios,
        },
        "calibration": calibration,
        "overlapping_origin_audit": {
            "stride": stride,
            "n_origins": len(overlap_rows),
            "evaluation_period": {
                "start": str(overlapping_splits.test_returns.index.min().date()),
                "end": str(overlapping_splits.test_returns.index.max().date()),
                "n_days": len(overlapping_splits.test_returns),
            },
            "co_primary_ratios_vs_var_garch": overlap_ratios,
            "composite_ratio": overlap_composite,
            "mean_losses": overlap_summary,
            "dependence_robust_comparisons": overlap_comparisons,
        },
        "compute_cost": compute_cost,
        "origins": origins,
        "overlap_origin_windows": overlap_origins,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    _plot_rank_histograms(calibration, output.with_suffix(".png"))
    print(
        f"Audit saved to {output}; locked composite reproduced "
        f"({composite:.6f}); overlapping origins={len(overlap_rows)} "
        f"(stride {stride}), composite={overlap_composite:.6f}"
    )
    return report


def _plot_rank_histograms(calibration, png_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    histograms = calibration["rank_histograms"]
    methods = list(histograms)
    bins = histograms[methods[0]]["bins"]
    width = 0.8 / len(methods)
    positions = np.arange(bins)
    fig, ax = plt.subplots(figsize=(8, 4))
    for method_index, method in enumerate(methods):
        counts = np.asarray(histograms[method]["counts"], dtype=float)
        expected = histograms[method]["expected_per_bin"]
        ax.bar(
            positions + method_index * width,
            counts / expected,
            width=width,
            label=method,
        )
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Observation rank in ensemble (0 = below all paths)")
    ax.set_ylabel("Relative frequency (1.0 = uniform)")
    ax.set_title("Locked S&P holdout: state forecast rank histograms")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="research_sp500_confirmation.yaml"
    )
    parser.add_argument(
        "--freeze-config", default="research_sp500_freeze.yaml"
    )
    parser.add_argument(
        "--frozen-protocol",
        default="research_output/sp500_frozen/frozen_protocol.json",
    )
    parser.add_argument(
        "--locked-result-reference",
        default="research_output/sp500_confirmation/confirmation.json",
    )
    parser.add_argument(
        "--output",
        default=(
            "research_output/sp500_confirmation/"
            "posthoc_calibration_power_audit.json"
        ),
    )
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-stocks", type=int, default=100)
    parser.add_argument("--allow-overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
