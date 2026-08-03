"""US equity data via yfinance, with the same interface as the AKShare DataPipeline."""

import time
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# S&P 100 constituents that have traded continuously since 2010. Deliberately a fixed,
# hardcoded list: fetching the *current* index membership and applying it to history would
# introduce survivorship bias, and a universe that changes between runs makes the benchmark
# irreproducible (which is exactly what bit us on the A-share side).
SP100 = [
    "AAPL", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN", "AXP",
    "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "C", "CAT", "CHTR", "CL",
    "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS", "CVX", "DE", "DHR",
    "DIS", "DUK", "EMR", "EXC", "F", "FDX", "GD", "GE", "GILD", "GM",
    "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "LIN",
    "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT", "MET", "MMM", "MO",
    "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE",
    "PG", "PM", "QCOM", "RTX", "SBUX", "SO", "SPG", "T", "TGT", "TXN",
    "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT", "XOM",
]
MARKET = "SPY"


class YFinanceDataPipeline:
    def __init__(self, config):
        self.config = config
        self.start_date = config.data.start_date
        self.end_date = config.data.end_date
        self.cache_dir = Path(config.data_dir) / "cache_us"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_coverage = config.data.min_coverage
        self.winsorize_std = config.data.winsorize_std
        self.universe_manifest = config.data.universe_manifest
        self.eligibility_end_date = config.data.eligibility_end_date
        self.preprocess_fit_end_date = config.data.preprocess_fit_end_date

    def _stock_list(self):
        if not self.universe_manifest:
            return SP100.copy()
        path = Path(self.universe_manifest)
        if not path.exists():
            raise FileNotFoundError(f"Universe manifest not found: {path}")
        frame = pd.read_csv(path)
        column = "Symbol" if "Symbol" in frame else "code"
        if column not in frame:
            raise ValueError(
                f"{path} must contain a 'Symbol' or 'code' column."
            )
        symbols = (
            frame[column].astype(str).str.strip().str.replace(".", "-", regex=False)
        )
        duplicates = symbols[symbols.duplicated()].unique().tolist()
        if duplicates:
            raise ValueError(
                f"Universe manifest contains duplicate symbols: {duplicates[:5]}"
            )
        return symbols.tolist()

    def _fmt(self, d):
        return f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else d

    def _download(self):
        if self.universe_manifest:
            manifest_bytes = Path(self.universe_manifest).read_bytes()
            universe_key = hashlib.sha256(manifest_bytes).hexdigest()[:12]
        else:
            universe_key = "sp100"
        cache = self.cache_dir / (
            f"prices_{universe_key}_{self.start_date}_{self.end_date}.parquet"
        )
        if cache.exists():
            return pd.read_parquet(cache)

        start, end = self._fmt(self.start_date), self._fmt(self.end_date)
        tickers = self._stock_list() + [MARKET]
        df = yf.download(tickers, start=start, end=end,
                         auto_adjust=True, progress=False)["Close"]

        # Yahoo drops tickers on transient network/rate-limit errors. Retry them explicitly:
        # silently accepting the gap would let the universe vary run to run.
        for attempt in range(3):
            missing = [t for t in tickers if t not in df.columns or df[t].isna().all()]
            if not missing:
                break
            print(f"  retry {attempt + 1}: refetching {len(missing)} tickers {missing}")
            time.sleep(2)
            retry = yf.download(missing, start=start, end=end,
                                auto_adjust=True, progress=False)["Close"]
            if isinstance(retry, pd.Series):
                retry = retry.to_frame(missing[0])
            for t in retry.columns:
                df[t] = retry[t]

        still_missing = [t for t in tickers if t not in df.columns or df[t].isna().all()]
        if still_missing:
            print(f"  WARNING: {len(still_missing)} tickers unavailable after retries: {still_missing}")

        df.to_parquet(cache)
        return df

    def winsorize(self, returns, reference=None):
        reference = returns if reference is None else reference
        if reference.empty:
            raise ValueError("Winsorization reference period is empty.")
        mean, std = reference.mean(), reference.std()
        return returns.clip(mean - self.winsorize_std * std,
                            mean + self.winsorize_std * std, axis=1)

    def prepare_raw_returns(self, prices, max_stocks=None):
        """Create an eligible, aligned panel without fitting winsorization."""
        market = prices[MARKET].pct_change(fill_method=None).dropna()
        stocks = prices.drop(columns=[MARKET])
        # yfinance commonly alphabetizes multi-ticker output. Restore the
        # archived manifest order so a prespecified leading-N screen means the
        # same thing across markets and across downloads.
        stocks = stocks.reindex(columns=self._stock_list())
        returns = stocks.pct_change(fill_method=None).dropna(how="all")

        eligibility = returns
        if self.eligibility_end_date:
            eligibility = eligibility.loc[
                eligibility.index <= pd.to_datetime(self.eligibility_end_date)
            ]
            if eligibility.empty:
                raise ValueError(
                    "eligibility_end_date precedes the available return data."
                )
        coverage = eligibility.notna().mean()
        keep = coverage[coverage >= self.min_coverage].index
        dropped = len(returns.columns) - len(keep)
        if dropped:
            print(f"Dropping {dropped} tickers with <{self.min_coverage:.0%} history; "
                  f"{len(keep)} remain")
        returns = returns[keep].fillna(0.0)

        if max_stocks:
            returns = returns.iloc[:, :max_stocks]

        idx = returns.index.intersection(market.index)
        returns, market = returns.loc[idx], market.loc[idx]
        return returns, market

    def load_all_data(self, max_stocks=None):
        prices = self._download()
        returns, market = self.prepare_raw_returns(
            prices, max_stocks=max_stocks
        )

        reference = returns
        if self.preprocess_fit_end_date:
            reference = reference.loc[
                reference.index <= pd.to_datetime(
                    self.preprocess_fit_end_date
                )
            ]
            if reference.empty:
                raise ValueError(
                    "preprocess_fit_end_date precedes the available return data."
                )
        returns = self.winsorize(returns, reference)

        idx = returns.index
        print(f"Loaded {returns.shape[1]} US stocks, {len(returns)} trading days "
              f"({idx.min().date()} -> {idx.max().date()})")
        return returns, market
