"""Phase 3A return-reconstruction selection gates."""

from research.phase3a_reconstruction import (
    _accept_candidate,
    _geometric_ratio,
)


def _metrics(values):
    return {
        name: {"mean": value}
        for name, value in values.items()
    }


def test_geometric_ratio_uses_requested_metrics():
    incumbent = _metrics({"a": 2.0, "b": 8.0})
    candidate = _metrics({"a": 1.0, "b": 4.0})
    composite, ratios = _geometric_ratio(
        candidate, incumbent, ("a", "b")
    )

    assert composite == 0.5
    assert ratios == {"a": 0.5, "b": 0.5}


def test_acceptance_gate_requires_broad_and_supported_improvement():
    ratios = {
        "return_energy_score": 0.98,
        "return_variogram_score": 0.99,
        "daily_volatility_mae": 0.97,
        "tail_quantile_error": 1.01,
        "max_drawdown_error": 1.02,
    }
    assert _accept_candidate(0.994, ratios, 0.06)["accepted"]
    assert not _accept_candidate(0.994, ratios, 0.15)["accepted"]


def test_acceptance_gate_rejects_large_single_metric_degradation():
    ratios = {
        "return_energy_score": 0.95,
        "return_variogram_score": 0.95,
        "daily_volatility_mae": 0.95,
        "tail_quantile_error": 0.95,
        "max_drawdown_error": 1.06,
    }
    assert not _accept_candidate(0.97, ratios, 0.01)["accepted"]
