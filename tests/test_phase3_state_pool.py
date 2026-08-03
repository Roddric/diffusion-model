"""Phase 3B state-pool acceptance rule."""

from research.phase3b_state_pool import _accept


def _paired(p):
    return {
        "state_energy_score": {
            "one_sided_p_diffusion_not_better": p
        }
    }


def test_state_pool_accepts_broad_supported_gain():
    result = _accept(
        0.98,
        {"state_energy_score": 0.97, "state_rmse": 0.99},
        _paired(0.04),
        1.01,
    )
    assert result["accepted"]


def test_state_pool_rejects_return_damage_or_weak_evidence():
    ratios = {"state_energy_score": 0.97, "state_rmse": 0.99}
    assert not _accept(0.98, ratios, _paired(0.11), 1.00)["accepted"]
    assert not _accept(0.98, ratios, _paired(0.04), 1.03)["accepted"]
