import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from research.evaluate_prospective_multimarket import (
    _post2023_survivor_coverage,
    _read_market_artifact,
    _verify_notarization,
    _write_atomic_json,
)
from research.prospective_multimarket import (
    EUROPE_BLOCK_CODES,
    POST2023_MIN_SESSION_COVERAGE,
    block_sensitivity_interval,
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


def test_population_claim_trichotomy_is_prespecified():
    winners = [
        {"strong_baseline_ratio": value, "passed": True}
        for value in (0.70, 0.72, 0.74, 0.76, 0.78)
    ]
    loser = {"strong_baseline_ratio": 0.80, "passed": False}
    result = population_decision(winners + [loser], required_passes=5)
    assert result["claim_classification"] == "pass"

    losers = [
        {"strong_baseline_ratio": value, "passed": False}
        for value in (1.20, 1.25, 1.30, 1.35, 1.40, 1.45)
    ]
    result = population_decision(losers, required_passes=5)
    assert result["passed"] is False
    assert result["claim_classification"] == "fail"
    assert result["interval"]["ci_log_lower"] > 0

    mixed = [
        {"strong_baseline_ratio": value, "passed": value < 1.0}
        for value in (0.90, 1.10, 0.95, 1.05, 0.92, 1.08)
    ]
    result = population_decision(mixed, required_passes=5)
    assert result["passed"] is False
    assert result["claim_classification"] == "inconclusive"


def test_claim_not_established_when_count_rule_fails():
    winners = [
        {"strong_baseline_ratio": value, "passed": True}
        for value in (0.70, 0.72, 0.74, 0.76)
    ]
    loser = {"strong_baseline_ratio": 0.80, "passed": False}
    result = population_decision(winners + [loser, loser], required_passes=5)
    assert result["passed"] is False
    assert result["claim_classification"] != "pass"


def test_block_sensitivity_collapses_prespecified_markets():
    codes = ("dax", "asx200", "cac40", "nifty50", "smi", "tsx60")
    ratios = (0.95, 0.97, 0.99, 1.01, 0.98, 1.02)
    decisions = {
        code: {"strong_baseline_ratio": value} for code, value in zip(codes, ratios)
    }
    result = block_sensitivity_interval(decisions)
    assert result["n_markets"] == 4
    assert result["units"] == [
        "asx200",
        "nifty50",
        "tsx60",
        "block:cac40+dax+smi",
    ]
    block_log = float(np.mean(np.log([0.99, 0.95, 0.98])))
    expected_logs = [np.log(0.97), np.log(1.01), np.log(1.02), block_log]
    assert result["mean_log_ratio"] == pytest.approx(float(np.mean(expected_logs)))
    assert result["block_codes"] == list(EUROPE_BLOCK_CODES)


def test_block_sensitivity_rejects_incomplete_blocks():
    decisions = {
        code: {"strong_baseline_ratio": 0.98}
        for code in ("dax", "asx200", "nifty50", "tsx60")
    }
    with pytest.raises(ValueError):
        block_sensitivity_interval(decisions)


def _synthetic_prices():
    dates = pd.bdate_range("2023-12-01", "2024-06-28")
    n = len(dates)
    prices = pd.DataFrame(index=dates)
    prices["^BENCH"] = 100.0 + np.arange(n) * 0.10
    prices["ALIVE"] = 50.0 + np.arange(n) * 0.05
    delisted = prices["ALIVE"].copy()
    delisted.loc[delisted.index >= "2024-03-01"] = np.nan
    prices["DELISTED"] = delisted
    return prices


def test_post2023_survivor_coverage_rule():
    prices = _synthetic_prices()
    survivors, dropped, coverage = _post2023_survivor_coverage(
        ["ALIVE", "DELISTED", "MISSING"], prices, "^BENCH"
    )
    assert survivors == ["ALIVE"]
    assert dropped == ["DELISTED", "MISSING"]
    assert coverage["MISSING"] == 0.0
    assert coverage["ALIVE"] == pytest.approx(1.0)
    assert coverage["DELISTED"] < POST2023_MIN_SESSION_COVERAGE


def test_post2023_survivor_coverage_requires_benchmark_sessions():
    prices = _synthetic_prices().loc[:"2023-12-29"]
    with pytest.raises(ValueError):
        _post2023_survivor_coverage(["ALIVE"], prices, "^BENCH")
    with pytest.raises(ValueError):
        _post2023_survivor_coverage(
            ["ALIVE"], _synthetic_prices(), "^NOT_PRESENT"
        )


def test_market_artifact_round_trip_and_nonterminal_rejection(tmp_path):
    path = tmp_path / "dax.json"
    payload = {"code": "dax", "status": "evaluated"}
    _write_atomic_json(path, payload)
    assert _read_market_artifact(path) == payload
    assert _read_market_artifact(tmp_path / "absent.json") is None
    _write_atomic_json(path, {"code": "dax", "status": "scoring"})
    with pytest.raises(ValueError):
        _read_market_artifact(path)


def test_mean_factor_r2_recovers_factor_structure():
    from config.config import Config
    from factors.extractor import FactorExtractor
    from research.prospective_multimarket_factor_quality import mean_factor_r2

    rng = np.random.default_rng(0)
    n_days = 400
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    market = pd.Series(rng.normal(0.0, 0.012, n_days), index=dates)
    betas = rng.uniform(0.8, 1.2, 6)
    returns = pd.DataFrame(
        market.values[:, None] * betas[None, :]
        + rng.normal(0.0, 0.002, (n_days, 6)),
        index=dates,
        columns=[f"S{i}" for i in range(6)],
    )
    extractor = FactorExtractor(Config())
    extractor.fit_mean_factors(returns, market)
    result = mean_factor_r2(extractor, returns, market)
    assert result["n_fitted"] == 6
    assert result["mean"] > 0.8
    assert result["min"] <= result["median"] <= result["max"]
