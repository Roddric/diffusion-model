import numpy as np
import pandas as pd

from data.yf_loader import (
    MAX_MISSING_FRACTION_FOR_CACHE,
    _classify_incomplete_download,
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
