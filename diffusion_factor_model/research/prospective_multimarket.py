"""Shared locked-study helpers for the prospective multi-market evaluation."""

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats


STATE_METRICS = ("state_energy_score", "state_rmse")
STRONG_BASELINES = ("Gaussian-VAR", "Student-t-VAR")
CANDIDATE_NAME = "Student-t-Base-Pool"
DIRECT_COVARIANCE_METRIC = "direct_covariance_frobenius_scaled_error"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def market_config(base_config, market, *, current=False, evaluation_end=None):
    """Create an isolated config for one immutable market manifest."""
    config = copy.deepcopy(base_config)
    code = market["code"]
    config.data.universe = code
    config.data.universe_manifest = market["universe_manifest"]
    config.data.market_benchmark = market["market_benchmark"]
    config.data.ticker_convention = "verbatim"
    config.data_dir = f"./research_data/prospective_multimarket/{code}"
    config.model_dir = f"./research_models/prospective_multimarket/{code}"
    config.output_dir = f"./research_output/prospective_multimarket/{code}"
    if current:
        if evaluation_end is None:
            raise ValueError("Current configs require an explicit evaluation end.")
        config.data.end_date = evaluation_end
    return config


def validate_pre2024_inputs(base_path, registry_path, smoke_path):
    """Verify that the eligibility report matches the exact registered inputs."""
    smoke = load_json(smoke_path)
    registry = load_json(registry_path)
    checks = {
        "smoke_complete": smoke.get("status") == "pre2024_feasibility_complete",
        "no_post2023_data_loaded": smoke.get("post2023_data_loaded") is False,
        "base_config_hash_match": smoke.get("base_config_sha256") == sha256(base_path),
        "registry_hash_match": smoke.get("registry_sha256") == sha256(registry_path),
        "study_id_match": smoke.get("study_id") == registry.get("study_id"),
    }
    if not all(checks.values()):
        raise ValueError(f"Pre-2024 input validation failed: {checks}")
    registered = {row["code"]: row for row in registry["primary_markets"]}
    eligible = [row for row in smoke["markets"] if row["eligible"]]
    if len(eligible) < 5:
        raise ValueError("Fewer than five eligible markets; population study stops.")
    if any(row["code"] not in registered for row in eligible):
        raise ValueError("Smoke report contains an unregistered eligible market.")
    required_passes = math.ceil(0.70 * len(eligible))
    return registry, smoke, registered, eligible, required_passes, checks


def geometric_ratio(candidate_metrics, baseline_metrics, metric_names=STATE_METRICS):
    ratios = {
        metric: float(
            candidate_metrics[metric]["mean"]
            / max(baseline_metrics[metric]["mean"], 1e-12)
        )
        for metric in metric_names
    }
    return ratios, float(np.exp(np.mean(np.log(list(ratios.values())))))


def strong_baseline_state_decision(metrics, candidate=CANDIDATE_NAME):
    """Apply the per-market beat-both-baselines state decision exactly."""
    comparisons = {}
    for baseline in STRONG_BASELINES:
        ratios, composite = geometric_ratio(metrics[candidate], metrics[baseline])
        comparisons[baseline] = {
            "component_ratios": ratios,
            "composite_ratio": composite,
        }
    strong_ratio = max(row["composite_ratio"] for row in comparisons.values())
    all_components = [
        ratio
        for row in comparisons.values()
        for ratio in row["component_ratios"].values()
    ]
    passed = bool(
        all(row["composite_ratio"] < 1.0 for row in comparisons.values())
        and all(ratio <= 1.0 for ratio in all_components)
    )
    return {
        "passed": passed,
        "strong_baseline_ratio": float(strong_ratio),
        "comparisons": comparisons,
    }


def strong_baseline_scalar_decision(candidate, baselines):
    """Apply the worse-of-two ratio rule to a scalar loss endpoint."""
    ratios = {
        baseline: float(candidate / max(baselines[baseline], 1e-12))
        for baseline in STRONG_BASELINES
    }
    strong_ratio = max(ratios.values())
    return {
        "passed": bool(all(value < 1.0 for value in ratios.values())),
        "strong_baseline_ratio": float(strong_ratio),
        "ratios": ratios,
    }


def market_t_interval(ratios, confidence=0.95):
    """Equal-market Student-t interval for the mean log loss ratio."""
    values = np.asarray(ratios, dtype=float)
    if len(values) < 2 or np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("At least two finite positive market ratios are required.")
    logs = np.log(values)
    mean = float(np.mean(logs))
    standard_error = float(stats.sem(logs))
    critical = float(stats.t.ppf((1.0 + confidence) / 2.0, len(logs) - 1))
    lower = mean - critical * standard_error
    upper = mean + critical * standard_error
    return {
        "n_markets": int(len(values)),
        "mean_log_ratio": mean,
        "ci_level": confidence,
        "ci_log_lower": float(lower),
        "ci_log_upper": float(upper),
        "geometric_mean_ratio": float(np.exp(mean)),
        "ci_ratio_lower": float(np.exp(lower)),
        "ci_ratio_upper": float(np.exp(upper)),
    }


def population_decision(market_decisions, required_passes):
    ratios = [row["strong_baseline_ratio"] for row in market_decisions]
    interval = market_t_interval(ratios)
    individual_passes = sum(bool(row["passed"]) for row in market_decisions)
    passed = bool(
        interval["ci_log_upper"] < 0.0
        and individual_passes >= required_passes
    )
    return {
        "passed": passed,
        "required_individual_passes": int(required_passes),
        "individual_passes": int(individual_passes),
        "interval": interval,
    }
