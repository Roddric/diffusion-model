import json
from types import SimpleNamespace

import numpy as np
import pytest

from research.evaluate_prospective_multimarket import _verify_notarization
from research.prospective_multimarket import (
    market_config,
    market_t_interval,
    population_decision,
    strong_baseline_scalar_decision,
    strong_baseline_state_decision,
    validate_pre2024_inputs,
)


def _metrics(candidate=(0.8, 0.9), gaussian=(1.0, 1.0), student=(0.9, 0.95)):
    def row(values):
        return {
            "state_energy_score": {"mean": values[0]},
            "state_rmse": {"mean": values[1]},
        }

    return {
        "Student-t-Base-Pool": row(candidate),
        "Gaussian-VAR": row(gaussian),
        "Student-t-VAR": row(student),
    }


def test_strong_baseline_state_decision_requires_all_four_components():
    passed = strong_baseline_state_decision(_metrics())
    assert passed["passed"] is True
    failed = strong_baseline_state_decision(
        _metrics(candidate=(1.01, 0.70), gaussian=(1.0, 1.0), student=(1.1, 1.0))
    )
    assert failed["comparisons"]["Gaussian-VAR"]["composite_ratio"] < 1.0
    assert failed["passed"] is False


def test_scalar_decision_uses_worse_baseline():
    result = strong_baseline_scalar_decision(
        0.95, {"Gaussian-VAR": 1.0, "Student-t-VAR": 0.90}
    )
    assert result["strong_baseline_ratio"] == pytest.approx(0.95 / 0.90)
    assert result["passed"] is False


def test_population_decision_requires_interval_and_count():
    winners = [
        {"strong_baseline_ratio": value, "passed": True}
        for value in (0.70, 0.72, 0.74, 0.76, 0.78)
    ]
    loser = {"strong_baseline_ratio": 0.80, "passed": False}
    result = population_decision(winners + [loser], required_passes=5)
    assert result["passed"] is True
    assert result["interval"]["ci_log_upper"] < 0
    result = population_decision(winners[:4] + [loser, loser], required_passes=5)
    assert result["passed"] is False


def test_t_interval_rejects_nonpositive_ratios():
    with pytest.raises(ValueError):
        market_t_interval([0.9, 0.0])


def test_market_config_does_not_mutate_template():
    base = SimpleNamespace(
        data=SimpleNamespace(end_date="20240101"),
        data_dir="template",
        model_dir="template",
        output_dir="template",
    )
    market = {
        "code": "tsx60",
        "universe_manifest": "tsx.csv",
        "market_benchmark": "^GSPTSE",
    }
    configured = market_config(base, market)
    assert configured.data.universe_manifest == "tsx.csv"
    assert base.data.end_date == "20240101"
    with pytest.raises(ValueError):
        market_config(base, market, current=True)


def test_validate_pre2024_inputs_locks_required_count(tmp_path):
    base = tmp_path / "base.yaml"
    registry_path = tmp_path / "registry.json"
    smoke_path = tmp_path / "smoke.json"
    base.write_text("data: {}\n")
    markets = [{"code": f"m{i}"} for i in range(6)]
    registry = {"study_id": "study", "primary_markets": markets}
    registry_path.write_text(json.dumps(registry))
    import hashlib

    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    smoke = {
        "status": "pre2024_feasibility_complete",
        "post2023_data_loaded": False,
        "base_config_sha256": digest(base),
        "registry_sha256": digest(registry_path),
        "study_id": "study",
        "markets": [{"code": f"m{i}", "eligible": True} for i in range(6)],
    }
    smoke_path.write_text(json.dumps(smoke))
    _, _, _, eligible, required, checks = validate_pre2024_inputs(
        base, registry_path, smoke_path
    )
    assert len(eligible) == 6
    assert required == 5
    assert all(checks.values())


def test_public_notarization_must_bind_exact_protocol(tmp_path):
    protocol = tmp_path / "protocol.json"
    notarization = tmp_path / "notarization.json"
    protocol.write_text('{"study_id": "study"}\n')
    import hashlib

    protocol_hash = hashlib.sha256(protocol.read_bytes()).hexdigest()
    notarization.write_text(
        json.dumps(
            {
                "study_id": "study",
                "frozen_protocol_sha256": protocol_hash,
                "public_commit": "abcdef123456",
                "public_url": "https://example.test/commit/abcdef1",
                "post2023_evaluation_started": False,
            }
        )
    )
    _, checks = _verify_notarization(
        notarization, protocol, {"study_id": "study"}
    )
    assert all(checks.values())
    record = json.loads(notarization.read_text())
    record["frozen_protocol_sha256"] = "wrong"
    notarization.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        _verify_notarization(notarization, protocol, {"study_id": "study"})
