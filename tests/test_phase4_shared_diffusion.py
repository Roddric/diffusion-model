"""Phase 4B shared-market integration rule."""

from research.phase4b_shared_diffusion import (
    SEPARATE,
    STUDENT,
    _integration_decision,
)


def _comparison(energy, rmse, return_composite, p=0.2):
    return {
        "state_composite": (energy * rmse) ** 0.5,
        "state_ratios": {
            "state_energy_score": energy,
            "state_rmse": rmse,
        },
        "return_composite": return_composite,
        "paired": {
            "state_energy_score": {
                "one_sided_p_diffusion_not_better": p
            }
        },
    }


def _market(separate, student):
    return {"comparisons": {SEPARATE: separate, STUDENT: student}}


def test_shared_diffusion_gate_accepts_cross_market_gain():
    reports = {
        "a": _market(
            _comparison(0.97, 0.98, 1.00, p=0.04),
            _comparison(0.98, 0.99, 1.00),
        ),
        "b": _market(
            _comparison(0.98, 0.99, 1.01),
            _comparison(0.99, 0.99, 1.00),
        ),
    }
    assert _integration_decision(reports)[
        "accepted_for_third_market_freeze"
    ]


def test_shared_diffusion_gate_rejects_market_specific_reversal():
    reports = {
        "a": _market(
            _comparison(0.95, 0.96, 1.00, p=0.01),
            _comparison(0.98, 0.99, 1.00),
        ),
        "b": _market(
            _comparison(1.02, 1.01, 1.00),
            _comparison(0.99, 0.99, 1.00),
        ),
    }
    assert not _integration_decision(reports)[
        "accepted_for_third_market_freeze"
    ]
