"""Tests for revision-pinned international universe construction."""

import pandas as pd
import pytest

from research.prospective_multimarket_universes import (
    _normalize_symbol,
    _render_manifest,
    _select_constituent_table,
)


def test_normalize_symbol_removes_exchange_prefix_and_adds_suffix():
    assert _normalize_symbol("EPA:AIR[1]", ".PA") == "AIR.PA"
    assert _normalize_symbol("7203", ".T") == "7203.T"
    assert _normalize_symbol("BHP.AX", ".AX") == "BHP.AX"
    assert _normalize_symbol("NAN", ".AX") == "NAN.AX"
    assert _normalize_symbol("TECK.B", ".TO", "-") == "TECK-B.TO"
    assert _normalize_symbol("BIP.UN.TO", ".TO", "-") == "BIP-UN.TO"


def test_table_selection_requires_one_in_range_symbol_table():
    noise = pd.DataFrame({"Year": [2020, 2021], "Value": [1, 2]})
    constituents = pd.DataFrame(
        {"Ticker": ["A", "B", "C"], "Company": ["A plc", "B plc", "C plc"]}
    )
    frame, symbol, name = _select_constituent_table(
        [noise, constituents], (3, 4)
    )

    assert frame is constituents
    assert symbol == "Ticker"
    assert name == "Company"


def test_render_manifest_applies_deterministic_suffix(monkeypatch):
    table = pd.DataFrame(
        {"Code": ["1", "2"], "Company name": ["One", "Two"]}
    )
    monkeypatch.setattr(pd, "read_html", lambda *args, **kwargs: [table])

    output, excluded = _render_manifest(
        "unused", {"suffix": ".T", "row_bounds": (2, 2)}
    )

    assert output.to_dict(orient="records") == [
        {"Symbol": "1.T", "Name": "One"},
        {"Symbol": "2.T", "Name": "Two"},
    ]
    assert excluded == []


def test_render_manifest_records_missing_symbol_exclusion(monkeypatch):
    table = pd.DataFrame(
        {"Ticker": ["A", None, "B"], "Company": ["Alpha", "Note", "Beta"]}
    )
    monkeypatch.setattr(pd, "read_html", lambda *args, **kwargs: [table])

    output, excluded = _render_manifest(
        "unused", {"suffix": ".SW", "row_bounds": (2, 3)}
    )

    assert output["Symbol"].tolist() == ["A.SW", "B.SW"]
    assert excluded == [
        {
            "source_row_index": 1,
            "values": {"ticker": None, "company": "Note"},
        }
    ]


def test_table_selection_rejects_ambiguous_tables():
    first = pd.DataFrame({"Ticker": ["A"], "Company": ["A plc"]})
    second = pd.DataFrame({"Symbol": ["B"], "Name": ["B plc"]})

    with pytest.raises(ValueError, match="exactly one"):
        _select_constituent_table([first, second], (1, 2))
