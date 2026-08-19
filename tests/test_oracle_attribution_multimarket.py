import numpy as np

from research.oracle_ceiling_attribution_multimarket import (
    PER_MARKET_SHARE_FLOOR,
    PROCEED_MIN_MARKETS_ABOVE_FLOOR,
    PROCEED_POINT_THRESHOLD,
    _mean_t_interval,
    _population_summary,
)


def _market_report(mean_share, vol_share, innovation_share):
    return {
        "shapley_attribution": {
            "components": {
                "mean_state": {"share_of_full_oracle_reduction": mean_share},
                "volatility_state": {"share_of_full_oracle_reduction": vol_share},
                "innovation": {"share_of_full_oracle_reduction": innovation_share},
            }
        }
    }


def test_mean_t_interval_brackets_point_estimate():
    values = [0.90, 0.92, 0.88, 0.91, 0.89, 0.93]
    interval = _mean_t_interval(values)
    assert interval["n"] == 6
    assert interval["ci_lower"] < interval["mean"] < interval["ci_upper"]
    assert interval["mean"] == np.mean(values)


def test_gate_passes_when_dominant_and_consistent():
    reports = [_market_report(0.42, 0.08, 0.50) for _ in range(6)]
    summary = _population_summary(reports)
    gate = summary["proceed_gate"]
    assert gate["passed"] is True
    assert (
        summary["mean_state_plus_innovation"]["markets_above_floor"]
        == len(reports)
    )


def test_gate_fails_when_point_estimate_below_threshold():
    reports = [_market_report(0.30, 0.10, 0.35) for _ in range(6)]
    summary = _population_summary(reports)
    combined = summary["mean_state_plus_innovation"]
    assert combined["mean"] < PROCEED_POINT_THRESHOLD
    assert summary["proceed_gate"]["passed"] is False


def test_gate_fails_when_too_few_markets_above_floor():
    reports = [_market_report(0.45, 0.05, 0.50) for _ in range(4)]
    reports += [_market_report(0.20, 0.10, 0.20) for _ in range(2)]
    summary = _population_summary(reports)
    combined = summary["mean_state_plus_innovation"]
    assert combined["markets_above_floor"] < PROCEED_MIN_MARKETS_ABOVE_FLOOR
    assert summary["proceed_gate"]["passed"] is False


def test_per_market_floor_is_applied_to_combined_share():
    reports = [_market_report(0.30, 0.30, 0.25) for _ in range(6)]
    summary = _population_summary(reports)
    combined = summary["mean_state_plus_innovation"]
    expected = [
        value > PER_MARKET_SHARE_FLOOR for value in combined["per_market"]
    ]
    assert combined["markets_above_floor"] == sum(expected)
