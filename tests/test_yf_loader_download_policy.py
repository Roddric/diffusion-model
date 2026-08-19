import numpy as np
import pandas as pd

from data import yf_loader
from data.yf_loader import (
    MAX_MISSING_FRACTION_FOR_CACHE,
    _classify_incomplete_download,
    _merge_refetch,
    _missing_tickers,
)


def test_missing_tickers_detects_absent_and_empty_columns():
    df = pd.DataFrame(
        {"A": [1.0, np.nan], "B": [np.nan, np.nan], "^X": [1.0, 2.0]},
        index=pd.date_range("2024-01-01", periods=2),
    )
    assert _missing_tickers(df, ["A", "B", "C", "^X"]) == ["B", "C"]


def test_classify_complete_download_as_asset_level():
    assert _classify_incomplete_download([], ["A", "B"], "^X") == "asset_level"


def test_classify_missing_benchmark_as_transport():
    assert _classify_incomplete_download(["^X"], ["A", "^X"], "^X") == "transport"


def test_classify_wide_missing_fraction_as_transport():
    tickers = [f"S{i}" for i in range(10)] + ["^X"]
    missing = [f"S{i}" for i in range(3)]
    assert len(missing) / len(tickers) > MAX_MISSING_FRACTION_FOR_CACHE
    assert _classify_incomplete_download(missing, tickers, "^X") == "transport"


def test_classify_isolated_missing_as_asset_level():
    tickers = [f"S{i}" for i in range(20)] + ["^X"]
    missing = ["S3"]
    assert _classify_incomplete_download(missing, tickers, "^X") == "asset_level"


def _frame(**columns):
    return pd.DataFrame(
        columns, index=pd.date_range("2024-01-01", periods=2)
    )


def test_merge_refetch_tolerates_response_without_close(monkeypatch):
    base = _frame(A=[1.0, 2.0])
    for bad_response in (
        None,
        pd.DataFrame(index=pd.date_range("2024-01-01", periods=2)),
        _frame(Open=[1.0, 2.0]),
    ):
        monkeypatch.setattr(
            yf_loader.yf, "download", lambda *a, **k: bad_response
        )
        result = _merge_refetch(base.copy(), ["B"], "2024-01-01", "2024-02-01")
        assert list(result.columns) == ["A"]


def test_merge_refetch_merges_close_columns(monkeypatch):
    base = _frame(A=[1.0, 2.0])
    values = {"A": [10.0, 20.0], "B": [3.0, 4.0], "^X": [5.0, 6.0]}
    flat = pd.DataFrame(values, index=pd.date_range("2024-01-01", periods=2))
    price_first = pd.DataFrame(
        values, index=flat.index
    )
    price_first.columns = pd.MultiIndex.from_product([["Close"], flat.columns])
    ticker_first = pd.DataFrame(
        values, index=flat.index
    )
    ticker_first.columns = pd.MultiIndex.from_product([flat.columns, ["Close"]])
    for response in (price_first, ticker_first):
        monkeypatch.setattr(yf_loader.yf, "download", lambda *a, **k: response)
        result = _merge_refetch(base.copy(), ["B", "^X"], "2024-01-01", "2024-02-01")
        assert list(result.columns) == ["A", "B", "^X"]
        assert result["B"].tolist() == [3.0, 4.0]
