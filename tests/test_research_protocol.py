"""Locked-universe, split-safe preprocessing, and calendar split tests."""

import numpy as np
import pandas as pd
import pytest

from data.loader import DataPipeline
from data.yf_loader import YFinanceDataPipeline
from sequences.dataset import FactorStateSequenceBuilder


def test_akshare_manifest_is_unique_and_normalized(config, tmp_path):
    manifest = tmp_path / "csi.csv"
    manifest.write_text(
        "Symbol,Name\n600519.SS,A\n000001.SZ,B\n", encoding="utf-8"
    )
    config.data.universe_manifest = str(manifest)
    pipeline = DataPipeline(config)
    assert pipeline.get_stock_list() == ["600519", "000001"]

    manifest.write_text(
        "Symbol,Name\n600519.SS,A\n600519.SS,A\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate"):
        pipeline.get_stock_list()


def test_yahoo_manifest_normalizes_class_tickers(config, tmp_path):
    manifest = tmp_path / "sp.csv"
    manifest.write_text(
        "Symbol,Name\nBRK.B,Berkshire\nBF.B,Brown-Forman\n", encoding="utf-8"
    )
    config.data.universe_manifest = str(manifest)
    assert YFinanceDataPipeline(config)._stock_list() == ["BRK-B", "BF-B"]


def test_winsorization_uses_only_prespecified_reference(config):
    config.data.winsorize_std = 1.0
    pipeline = DataPipeline(config)
    dates = pd.bdate_range("2020-01-01", periods=6)
    returns = pd.DataFrame(
        {"A": [-1.0, 0.0, 1.0, 2.0, 100.0, -100.0]}, index=dates
    )
    reference = returns.iloc[:4]
    original = pipeline.winsorize(returns, reference)

    changed_future = returns.copy()
    changed_future.iloc[4:] *= 1000
    after = pipeline.winsorize(changed_future, reference)

    np.testing.assert_allclose(original.iloc[:4], after.iloc[:4])
    assert original.iloc[4, 0] == after.iloc[4, 0]
    assert original.iloc[5, 0] == after.iloc[5, 0]


def test_akshare_load_all_data_applies_frozen_preprocessing_reference(
    config, monkeypatch
):
    config.data.preprocess_fit_end_date = "2020-01-03"
    config.data.winsorize_std = 1.0
    pipeline = DataPipeline(config)
    dates = pd.bdate_range("2020-01-01", periods=5)
    raw = pd.DataFrame(
        {"A": [-1.0, 0.0, 1.0, 100.0, -100.0]}, index=dates
    )
    market = pd.Series(0.0, index=dates)
    monkeypatch.setattr(
        pipeline,
        "prepare_raw_returns",
        lambda max_stocks=None: (raw.copy(), market.copy()),
    )

    actual, actual_market = pipeline.load_all_data(max_stocks=1)
    expected = pipeline.winsorize(raw, raw.iloc[:3])

    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_series_equal(actual_market, market)


def test_explicit_calendar_splits_are_respected(config):
    config.data.synthetic_n_stocks = 8
    config.data.synthetic_n_days = 900
    config.factors.n_vol_factors = 3
    config.factors.vol_window = 10
    from data.loader import SyntheticDataPipeline

    returns, market = SyntheticDataPipeline(config).load_all_data()
    train_end = returns.index[499]
    validation_end = returns.index[699]
    splits = FactorStateSequenceBuilder(
        config,
        context_length=20,
        horizon=5,
        evaluation_stride=5,
        train_end_date=str(train_end.date()),
        validation_end_date=str(validation_end.date()),
    ).build(returns, market)

    assert splits.train_returns.index.max() <= train_end
    assert splits.validation_returns.index.min() > train_end
    assert splits.validation_returns.index.max() <= validation_end
    assert splits.test_returns.index.min() > validation_end
    assert splits.validation.target_dates.max() <= validation_end.to_datetime64()
    assert splits.test.target_dates.min() > validation_end.to_datetime64()


def test_training_builder_can_freeze_without_loading_test_period(config):
    config.data.synthetic_n_stocks = 8
    config.data.synthetic_n_days = 750
    config.factors.n_vol_factors = 3
    config.factors.vol_window = 10
    from data.loader import SyntheticDataPipeline

    returns, market = SyntheticDataPipeline(config).load_all_data()
    train_end = returns.index[499]
    validation_end = returns.index[-1]
    splits = FactorStateSequenceBuilder(
        config,
        context_length=20,
        horizon=5,
        train_end_date=str(train_end.date()),
        validation_end_date=str(validation_end.date()),
        allow_empty_test=True,
    ).build(returns, market)

    assert len(splits.validation) > 0
    assert len(splits.test_returns) == 0
    assert len(splits.test) == 0
