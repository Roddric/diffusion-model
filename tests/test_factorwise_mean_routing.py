"""Factorwise mean-state routing tests."""

import numpy as np

from research.factorwise_mean_routing import (
    _gate,
    _route_dimensions,
    _select_student_factors,
)


def test_selection_requires_joint_validation_margin():
    diagnostic = {
        "validation": {
            "pooled_ratios": {
                "Student-t-VAR": {
                    "market": {"rmse_ratio": 1.03, "crps_ratio": 1.04},
                    "momentum": {"rmse_ratio": 1.03, "crps_ratio": 1.01},
                    "reversal": {"rmse_ratio": 0.90, "crps_ratio": 0.91},
                }
            }
        }
    }

    assert _select_student_factors(diagnostic) == ["market"]


def test_routing_changes_only_selected_dimensions():
    pooled = np.zeros((3, 4, 5))
    student = np.ones((3, 4, 5))

    routed = _route_dimensions(pooled, student, [0, 4])

    np.testing.assert_array_equal(routed[:, :, [0, 4]], 1.0)
    np.testing.assert_array_equal(routed[:, :, 1:4], 0.0)


def test_gate_requires_supported_energy_improvement():
    paired = {
        "state_energy_score": {"one_sided_p_diffusion_not_better": 0.05}
    }
    assert _gate(0.99, {"a": 0.99, "b": 1.0}, 0.98, 1.01, paired)["accepted"]
    paired["state_energy_score"]["one_sided_p_diffusion_not_better"] = 0.2
    assert not _gate(0.99, {"a": 0.99, "b": 1.0}, 0.98, 1.01, paired)[
        "accepted"
    ]
