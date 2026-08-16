"""Tests for the prospective study's pre-outcome configuration guards."""

import json

import pytest

from config.config import load_config
from research.prospective_multimarket_smoke import load_registry, market_config


def test_market_config_preserves_pre2024_boundary(tmp_path):
    base = load_config("research_multimarket_base.yaml")
    market = {
        "code": "test",
        "universe_manifest": "manifest.csv",
        "market_benchmark": "^TEST",
    }

    config = market_config(base, market)

    assert config.data.end_date == "20240101"
    assert config.data.eligibility_end_date == "20221230"
    assert config.data.preprocess_fit_end_date == "20221230"
    assert config.data.ticker_convention == "verbatim"
    assert config.data.market_benchmark == "^TEST"


def test_registry_rejects_duplicate_market_codes(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps({"primary_markets": [{"code": "x"}, {"code": "x"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_registry(path)
