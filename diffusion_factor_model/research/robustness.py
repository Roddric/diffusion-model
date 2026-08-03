"""Dependence-robust summaries for paired rolling-origin forecast losses."""

import numpy as np
from scipy.stats import norm


def _paired_difference(candidate, baseline):
    candidate = np.asarray(candidate, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if candidate.shape != baseline.shape or candidate.ndim != 1:
        raise ValueError(
            "Paired score vectors must be one-dimensional and aligned."
        )
    if len(candidate) < 3:
        raise ValueError("At least three paired origins are required.")
    if not np.isfinite(candidate).all() or not np.isfinite(baseline).all():
        raise ValueError("Paired score vectors must contain finite values.")
    return candidate - baseline


def circular_block_bootstrap_mean(
    difference, block_length, seed=0, draws=10000
):
    """Bootstrap a sample mean using fixed-length circular moving blocks."""
    difference = np.asarray(difference, dtype=float)
    if difference.ndim != 1 or len(difference) < 3:
        raise ValueError("difference must be a one-dimensional vector.")
    if block_length < 1 or block_length > len(difference):
        raise ValueError("block_length must lie between one and sample size.")
    if draws < 1:
        raise ValueError("draws must be positive.")

    rng = np.random.default_rng(seed)
    n = len(difference)
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=(draws, n_blocks))
    offsets = np.arange(block_length)
    indices = (starts[..., None] + offsets) % n
    samples = difference[indices.reshape(draws, -1)[:, :n]]
    means = samples.mean(axis=1)
    return {
        "block_length": int(block_length),
        "draws": int(draws),
        "mean_difference": float(difference.mean()),
        "ci_95": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "bootstrap_tail_probability_mean_not_better": float(
            np.mean(means >= 0.0)
        ),
    }


def hac_mean_test(difference, max_lag=None):
    """Newey-West standard error for the mean paired loss difference.

    The reported one-sided normal-approximation p-value tests a non-negative
    expected candidate-minus-baseline loss against the alternative that the
    candidate has lower loss.
    """
    difference = np.asarray(difference, dtype=float)
    if difference.ndim != 1 or len(difference) < 3:
        raise ValueError("difference must be a one-dimensional vector.")
    n = len(difference)
    if max_lag is None:
        max_lag = min(int(np.floor(4 * (n / 100) ** (2 / 9))), n - 1)
    if max_lag < 0 or max_lag >= n:
        raise ValueError("max_lag must lie between zero and n - 1.")

    centered = difference - difference.mean()
    long_run_variance = float(np.dot(centered, centered) / n)
    for lag in range(1, max_lag + 1):
        autocovariance = float(
            np.dot(centered[lag:], centered[:-lag]) / n
        )
        weight = 1.0 - lag / (max_lag + 1.0)
        long_run_variance += 2.0 * weight * autocovariance
    long_run_variance = max(long_run_variance, 0.0)
    standard_error = float(np.sqrt(long_run_variance / n))
    if standard_error == 0.0:
        statistic = float("-inf" if difference.mean() < 0 else "inf")
        p_value = float(0.0 if difference.mean() < 0 else 1.0)
    else:
        statistic = float(difference.mean() / standard_error)
        p_value = float(norm.cdf(statistic))
    return {
        "max_lag": int(max_lag),
        "mean_difference": float(difference.mean()),
        "newey_west_standard_error": standard_error,
        "z_statistic": statistic,
        "one_sided_normal_p_candidate_not_better": p_value,
    }


def dependence_robust_comparison(
    candidate,
    baseline,
    block_lengths=(2, 3, 4, 5),
    seed=20260803,
    draws=10000,
):
    """Summarize paired losses under several dependence assumptions."""
    difference = _paired_difference(candidate, baseline)
    lengths = sorted(set(int(value) for value in block_lengths))
    return {
        "difference_definition": "candidate loss minus baseline loss",
        "lower_is_better": True,
        "n_origins": int(len(difference)),
        "origin_loss_differences": difference.tolist(),
        "hac": hac_mean_test(difference),
        "circular_block_bootstrap": {
            str(length): circular_block_bootstrap_mean(
                difference,
                block_length=length,
                seed=seed + length,
                draws=draws,
            )
            for length in lengths
        },
    }
