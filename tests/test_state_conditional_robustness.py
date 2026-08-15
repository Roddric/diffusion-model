"""Locked state-conditional robustness audit tests."""

import numpy as np

from research.state_conditional_robustness_audit import (
    _direct_covariance_scores,
    _survival_gate,
)


def test_direct_covariance_score_is_zero_for_exact_paths():
    rng = np.random.default_rng(4)
    target = rng.normal(size=(20, 6))
    paths = np.repeat(target[None, :, :], 10, axis=0)
    weights = rng.dirichlet(np.ones(6), size=12)

    scores = _direct_covariance_scores(paths, target, weights)

    assert scores["direct_covariance_frobenius_scaled_error"] < 1e-12
    assert scores["direct_correlation_rmse"] < 1e-12
    assert scores["random_portfolio_variance_scaled_mae"] < 1e-12


def _configuration(composite=0.98, energy=0.99):
    paired = {
        "return_energy_score": {
            "one_sided_p_diffusion_not_better": 0.04
        }
    }
    return {
        "candidate_vs_incumbent": {
            "return_composite_ratio": composite,
            "return_metric_ratios": {"return_energy_score": energy},
            "direct_risk_ratios": {
                "direct_covariance_frobenius_scaled_error": 1.01,
                "random_portfolio_variance_scaled_mae": 1.02,
                "portfolio_var_05_pinball": 0.99,
            },
            "paired_origin_bootstrap": paired,
        },
        "candidate_vs_unconditional": {"return_composite_ratio": 0.97},
        "regime_composites_vs_incumbent": {
            "low": {"composite_ratio": 0.99},
            "high": {"composite_ratio": 1.01},
        },
    }


def test_survival_gate_requires_every_safety_condition():
    configurations = {
        f"paths_20_repeat_{repeat}": _configuration()
        for repeat in range(5)
    }
    configurations["paths_50_repeat_0"] = _configuration()
    configurations["paths_100_repeat_0"] = _configuration()

    assert _survival_gate(configurations)["survives"]

    configurations["paths_100_repeat_0"]["candidate_vs_incumbent"][
        "direct_risk_ratios"
    ]["portfolio_var_05_pinball"] = 1.06
    assert not _survival_gate(configurations)["survives"]
