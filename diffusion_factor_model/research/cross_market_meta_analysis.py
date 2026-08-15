"""Post-hoc cross-market synthesis of the locked Phase 2F comparisons.

The market is the unit of replication.  The primary synthesis is the
equal-market mean of log composite ratios with a Student-t interval across
markets.  Paired-origin bootstrap intervals are shown for the individual
markets, but they are not treated as additional independent replications.

This analysis is necessarily post-hoc: the S&P 500, FTSE 100, and Hang Seng
results were known before this script and protocol were written.  It must not
be described as a preregistered pooled test.
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


METRICS = ("state_energy_score", "state_rmse")
MARKETS = (
    {
        "market": "S&P 500",
        "evidence_class": "retrospective locked holdout",
        "result": "research_output/sp500_confirmation/confirmation.json",
        "origin_scores": (
            "research_output/sp500_confirmation/"
            "posthoc_dependence_robustness.json"
        ),
        "pool": "Phase2F-Validation-Pooled-Diffusion",
        "baseline": "VAR-GARCH",
        "decision_key": "confirmatory_decision",
    },
    {
        "market": "FTSE 100",
        "evidence_class": "preregistered external confirmation",
        "result": "research_output/ftse100_confirmation/confirmation.json",
        "origin_scores": "research_output/ftse100_confirmation/confirmation.json",
        "pool": "FTSE100-Gaussian-Base-Pool",
        "baseline": "VAR-GARCH",
        "decision_key": "primary_decision",
    },
    {
        "market": "Hang Seng",
        "evidence_class": "preregistered external confirmation",
        "result": "research_output/hsi_confirmation/confirmation.json",
        "origin_scores": "research_output/hsi_confirmation/confirmation.json",
        "pool": "HSI-Gaussian-Base-Pool",
        "baseline": "VAR-GARCH",
        "decision_key": "primary_decision",
    },
)


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _composite_ratio(pool_scores, baseline_scores, indices=None):
    """Geometric mean of the two co-primary mean-loss ratios."""
    ratios = []
    for metric in METRICS:
        pool = np.asarray(pool_scores[metric], dtype=float)
        baseline = np.asarray(baseline_scores[metric], dtype=float)
        if indices is not None:
            pool = pool[indices]
            baseline = baseline[indices]
        ratios.append(float(pool.mean() / baseline.mean()))
    return float(math.sqrt(np.prod(ratios)))


def _origin_bootstrap(pool_scores, baseline_scores, samples, seed):
    """Paired-origin bootstrap distribution of the log composite ratio."""
    lengths = {
        len(pool_scores[metric]) for metric in METRICS
    } | {
        len(baseline_scores[metric]) for metric in METRICS
    }
    if len(lengths) != 1:
        raise ValueError("All paired origin-score arrays must have equal length.")
    n_origins = lengths.pop()
    if n_origins < 2:
        raise ValueError("At least two origins are required for bootstrap inference.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n_origins, size=(samples, n_origins))
    log_draws = np.zeros(samples, dtype=float)
    for metric in METRICS:
        pool = np.asarray(pool_scores[metric], dtype=float)
        baseline = np.asarray(baseline_scores[metric], dtype=float)
        log_draws += 0.5 * np.log(
            pool[indices].mean(axis=1) / baseline[indices].mean(axis=1)
        )
    return log_draws


def _equal_market_summary(log_effects):
    """Equal-market log-effect mean and small-K Student-t interval."""
    values = np.asarray(log_effects, dtype=float)
    if len(values) < 2:
        raise ValueError("At least two markets are required for a market-level CI.")
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    critical = float(t.ppf(0.975, df=len(values) - 1))
    lower, upper = mean - critical * se, mean + critical * se
    return {
        "k_markets": int(len(values)),
        "mean_log_composite": mean,
        "standard_error_across_markets": se,
        "t_critical_95": critical,
        "ci_95_log": [float(lower), float(upper)],
        "composite_ratio": float(math.exp(mean)),
        "ci_95_composite_ratio": [float(math.exp(lower)), float(math.exp(upper))],
    }


def _dersimonian_laird(log_effects, standard_errors):
    """DL random-effects sensitivity with an unmodified HKSJ interval."""
    effects = np.asarray(log_effects, dtype=float)
    ses = np.asarray(standard_errors, dtype=float)
    if len(effects) < 2 or len(effects) != len(ses) or np.any(ses <= 0):
        raise ValueError("Effects and positive SEs are required for at least two markets.")
    fixed_weights = 1.0 / np.square(ses)
    fixed_mean = float(np.sum(fixed_weights * effects) / fixed_weights.sum())
    q = float(np.sum(fixed_weights * np.square(effects - fixed_mean)))
    c = float(fixed_weights.sum() - np.square(fixed_weights).sum() / fixed_weights.sum())
    tau_squared = float(max(0.0, (q - (len(effects) - 1)) / c))
    random_weights = 1.0 / (np.square(ses) + tau_squared)
    random_mean = float(np.sum(random_weights * effects) / random_weights.sum())
    hksj_scale = float(
        np.sum(random_weights * np.square(effects - random_mean))
        / (len(effects) - 1)
    )
    hksj_se = float(math.sqrt(hksj_scale / random_weights.sum()))
    critical = float(t.ppf(0.975, df=len(effects) - 1))
    lower = random_mean - critical * hksj_se
    upper = random_mean + critical * hksj_se
    i_squared = float(max(0.0, (q - (len(effects) - 1)) / max(q, 1e-12)))
    return {
        "method": "DerSimonian-Laird tau-squared; unmodified HKSJ t interval",
        "k_markets": int(len(effects)),
        "fixed_effect_mean_log": fixed_mean,
        "cochran_q": q,
        "i_squared": i_squared,
        "tau_squared": tau_squared,
        "random_effect_mean_log": random_mean,
        "hksj_standard_error": hksj_se,
        "ci_95_log": [float(lower), float(upper)],
        "composite_ratio": float(math.exp(random_mean)),
        "ci_95_composite_ratio": [float(math.exp(lower)), float(math.exp(upper))],
        "interpretation": (
            "Sensitivity analysis only; heterogeneity estimates and HKSJ intervals "
            "are unstable with three markets."
        ),
    }


def _market_record(spec, bootstrap_samples, seed):
    result = _load_json(spec["result"])
    score_artifact = _load_json(spec["origin_scores"])
    origin_scores = score_artifact["origin_level_scores"]
    pool_scores = origin_scores[spec["pool"]]
    baseline_scores = origin_scores[spec["baseline"]]
    point = _composite_ratio(pool_scores, baseline_scores)
    decision = result[spec["decision_key"]]
    if not np.isclose(point, decision["composite_ratio"], rtol=0.0, atol=1e-12):
        raise ValueError(
            f'{spec["market"]} raw scores do not reproduce the locked composite.'
        )
    draws = _origin_bootstrap(pool_scores, baseline_scores, bootstrap_samples, seed)
    ci = np.quantile(draws, [0.025, 0.975])
    horizon_ratios = {}
    for horizon, scores in result["horizon_robustness"].items():
        component_ratios = [
            scores[spec["pool"]][metric] / scores[spec["baseline"]][metric]
            for metric in METRICS
        ]
        horizon_ratios[horizon] = float(math.sqrt(np.prod(component_ratios)))
    return {
        "market": spec["market"],
        "evidence_class": spec["evidence_class"],
        "pool": spec["pool"],
        "baseline": spec["baseline"],
        "n_origins": len(pool_scores[METRICS[0]]),
        "locked_gate_success": bool(decision["success"]),
        "composite_ratio": point,
        "log_composite_ratio": float(math.log(point)),
        "paired_origin_bootstrap_standard_error_log": float(draws.std(ddof=1)),
        "paired_origin_bootstrap_ci_95_composite_ratio": [
            float(math.exp(ci[0])),
            float(math.exp(ci[1])),
        ],
        "horizon_composite_ratios": horizon_ratios,
        "source_result": spec["result"],
        "source_origin_scores": spec["origin_scores"],
    }


def _plot_forest(markets, pooled, external, output_stem):
    labels = [row["market"] for row in markets] + [
        "All markets (equal weight)",
        "External only (equal weight)",
    ]
    values = [row["composite_ratio"] for row in markets] + [
        pooled["composite_ratio"],
        external["composite_ratio"],
    ]
    intervals = [
        row["paired_origin_bootstrap_ci_95_composite_ratio"] for row in markets
    ] + [pooled["ci_95_composite_ratio"], external["ci_95_composite_ratio"]]
    y = np.arange(len(labels))[::-1]
    fig, axis = plt.subplots(figsize=(8.4, 5.1))
    colors = ["#4c78a8", "#2a9d8f", "#e76f51", "#222222", "#6c5ce7"]
    markers = ["o", "o", "o", "D", "D"]
    for index, (value, interval) in enumerate(zip(values, intervals)):
        axis.errorbar(
            value,
            y[index],
            xerr=[[value - interval[0]], [interval[1] - value]],
            fmt=markers[index],
            color=colors[index],
            ecolor=colors[index],
            capsize=3,
            markersize=6.5 if index < 3 else 7.5,
            linewidth=1.4,
        )
    axis.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_yticks(y, labels)
    axis.set_xlabel("Composite loss ratio (pool / Gaussian VAR baseline)")
    axis.set_title("Cross-market synthesis of the locked Phase 2F comparison")
    axis.grid(axis="x", alpha=0.2)
    axis.text(
        0.01,
        -0.18,
        "Market intervals: paired-origin bootstrap. Pooled intervals: Student-t across markets.\n"
        "Ratios below 1 favor pooling; synthesis is post-hoc.",
        transform=axis.transAxes,
        fontsize=8.5,
        va="top",
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            f"{output_stem}.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def _plot_horizons(markets, output_stem):
    fig, axis = plt.subplots(figsize=(7.7, 4.7))
    colors = ("#4c78a8", "#2a9d8f", "#e76f51")
    for market, color in zip(markets, colors):
        horizons = sorted(int(value) for value in market["horizon_composite_ratios"])
        ratios = [market["horizon_composite_ratios"][str(value)] for value in horizons]
        axis.plot(horizons, ratios, marker="o", linewidth=1.8, color=color, label=market["market"])
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_xticks([5, 10, 20])
    axis.set_xlabel("Cumulative forecast horizon (trading days)")
    axis.set_ylabel("Composite loss ratio")
    axis.set_title("Horizon robustness of the Gaussian-base pool")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    axis.text(
        0.01,
        -0.18,
        "Geometric mean of cumulative energy-score and RMSE ratios; below 1 favors pooling.",
        transform=axis.transAxes,
        fontsize=8.5,
        va="top",
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            f"{output_stem}.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "cross_market_meta_analysis.json"
    if output_json.exists() and not args.allow_overwrite:
        raise FileExistsError(f"Output already exists: {output_json}")

    markets = [
        _market_record(spec, args.bootstrap_samples, args.seed + index)
        for index, spec in enumerate(MARKETS)
    ]
    log_effects = [row["log_composite_ratio"] for row in markets]
    pooled = _equal_market_summary(log_effects)
    external_rows = [
        row for row in markets if row["evidence_class"].startswith("preregistered")
    ]
    external = _equal_market_summary(
        [row["log_composite_ratio"] for row in external_rows]
    )
    random_effects = _dersimonian_laird(
        log_effects,
        [row["paired_origin_bootstrap_standard_error_log"] for row in markets],
    )
    report = {
        "status": "completed_posthoc_synthesis",
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "analysis_role": "post-hoc descriptive synthesis; not preregistered",
            "estimand": "equal-market mean log composite loss ratio",
            "market_unit": (
                "one locked Gaussian-base pool versus its own Gaussian VAR/VAR-GARCH "
                "baseline per market"
            ),
            "primary_interval": (
                "two-sided 95% Student-t interval across market log ratios"
            ),
            "individual_market_interval": (
                "paired nonparametric bootstrap of locked origin indices"
            ),
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.seed,
            "exclusions": (
                "CSI 300 is excluded because it evaluates a different Phase 3B state-pool "
                "candidate and therefore does not share this estimand."
            ),
        },
        "markets": markets,
        "primary_mixed_evidence_summary": pooled,
        "preregistered_external_only_sensitivity": external,
        "random_effects_sensitivity": random_effects,
        "claim_boundary": (
            "The three-market point estimate is descriptive and imprecise. It cannot "
            "establish a universal pooling benefit; the external-only interval is based "
            "on two markets and includes no effect."
        ),
    }
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _plot_forest(markets, pooled, external, output_dir / "cross_market_forest")
    _plot_horizons(markets, output_dir / "horizon_robustness")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="research_output/cross_market_meta_analysis"
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--allow-overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
