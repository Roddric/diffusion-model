"""Phase 4C adapter advancement rule."""

from research.phase4b_shared_diffusion import SEPARATE, STUDENT
from research.phase4c_adapter_diffusion import (
    FULLY_SHARED,
    _adapter_decision,
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


def _market(separate, shared, student):
    return {
        "comparisons": {
            SEPARATE: separate,
            FULLY_SHARED: shared,
            STUDENT: student,
        }
    }


def test_adapter_gate_accepts_protected_cross_market_gain():
    reports = {
        "sp500": _market(
            _comparison(0.97, 0.98, 1.00, p=0.04),
            _comparison(0.98, 0.99, 1.00),
            _comparison(0.98, 0.99, 1.00),
        ),
        "csi300": _market(
            _comparison(0.98, 0.99, 1.01),
            _comparison(1.00, 1.00, 1.00),
            _comparison(0.99, 0.99, 1.00),
        ),
    }
    assert _adapter_decision(reports)[
        "accepted_for_third_market_freeze"
    ]


def test_adapter_gate_rejects_failure_to_protect_sp500():
    reports = {
        "sp500": _market(
            _comparison(0.97, 0.98, 1.00, p=0.04),
            _comparison(1.01, 1.01, 1.00),
            _comparison(0.98, 0.99, 1.00),
        ),
        "csi300": _market(
            _comparison(0.98, 0.99, 1.00),
            _comparison(0.99, 0.99, 1.00),
            _comparison(0.99, 0.99, 1.00),
        ),
    }
    assert not _adapter_decision(reports)[
        "accepted_for_third_market_freeze"
    ]
