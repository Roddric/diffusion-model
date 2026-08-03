"""Phase 4A multi-market regime baseline gates."""

from research.phase4a_regime_var import _accept_integration


def _market(state_energy, state_rmse, return_composite, energy_p):
    return {
        "state_composite_vs_student_t": (
            state_energy * state_rmse
        ) ** 0.5,
        "state_ratios_vs_student_t": {
            "state_energy_score": state_energy,
            "state_rmse": state_rmse,
        },
        "return_composite_vs_student_t": return_composite,
        "paired_candidate_vs_student_t": {
            "state_energy_score": {
                "one_sided_p_diffusion_not_better": energy_p
            }
        },
    }


def test_regime_integration_gate_accepts_broad_material_gain():
    reports = {
        "a": _market(0.97, 0.98, 1.00, 0.04),
        "b": _market(0.98, 0.99, 1.01, 0.20),
    }
    decision = _accept_integration(reports)
    assert decision["accepted_for_diffusion_integration"]


def test_regime_integration_gate_rejects_one_market_reversal():
    reports = {
        "a": _market(0.95, 0.96, 1.00, 0.01),
        "b": _market(1.01, 1.00, 1.00, 0.20),
    }
    assert not _accept_integration(reports)[
        "accepted_for_diffusion_integration"
    ]
