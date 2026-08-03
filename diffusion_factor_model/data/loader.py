import akshare as ak
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor


class DataPipeline:
    def __init__(self, config):
        self.config = config
        self.universe = config.data.universe
        self.start_date = config.data.start_date
        self.end_date = config.data.end_date
        self.cache_dir = Path(config.data_dir) / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_history = config.data.min_history
        self.winsorize_std = config.data.winsorize_std
        self.min_coverage = config.data.min_coverage
        self.universe_manifest = config.data.universe_manifest
        self.eligibility_end_date = config.data.eligibility_end_date
        self.preprocess_fit_end_date = config.data.preprocess_fit_end_date
        self.adjustment = config.data.adjustment
        self.download_workers = config.data.download_workers
        self.akshare_endpoint = config.data.akshare_endpoint
        if self.download_workers < 1:
            raise ValueError("data.download_workers must be positive.")
        if self.akshare_endpoint not in {"eastmoney", "sina"}:
            raise ValueError(
                "data.akshare_endpoint must be 'eastmoney' or 'sina'."
            )
        if self.akshare_endpoint == "sina" and self.download_workers != 1:
            raise ValueError(
                "AKShare's Sina adjustment engine is not thread-safe; set "
                "data.download_workers to 1."
            )

    @staticmethod
    def _manifest_codes(path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Universe manifest not found: {path}")
        frame = pd.read_csv(path)
        if "code" in frame:
            raw = frame["code"].astype(str)
        elif "Symbol" in frame:
            raw = frame["Symbol"].astype(str).str.extract(
                r"^(\d{6})(?:\.(?:SS|SZ))?$", expand=False
            )
            if raw.isna().any():
                bad = frame.loc[raw.isna(), "Symbol"].astype(str).tolist()[:5]
                raise ValueError(
                    f"Invalid A-share symbols in {path}: {bad}"
                )
        else:
            raise ValueError(
                f"{path} must contain a 'code' or 'Symbol' column."
            )
        codes = raw.str.zfill(6)
        duplicates = codes[codes.duplicated()].unique().tolist()
        if duplicates:
            raise ValueError(
                f"Universe manifest contains duplicate codes: {duplicates[:5]}"
            )
        return codes.tolist()

    def get_stock_list(self):
        if self.universe_manifest:
            return self._manifest_codes(self.universe_manifest)

        cache_file = self.cache_dir / f'universe_{self.universe}.csv'
        if cache_file.exists():
            df = pd.read_csv(cache_file)
            codes = df['code'].astype(str).str.zfill(6)
            duplicates = int(codes.duplicated().sum())
            if duplicates:
                print(
                    f"WARNING: dropping {duplicates} duplicate codes from "
                    f"cached latest-membership universe {cache_file}."
                )
            return codes.drop_duplicates().tolist()
        
        if self.universe == 'hs300':
            df = ak.index_stock_cons(symbol='000300')
            codes = (
                df['品种代码'].astype(str).str.zfill(6).drop_duplicates().tolist()
            )
        elif self.universe == 'hs500':
            df = ak.index_stock_cons(symbol='000905')
            codes = (
                df['品种代码'].astype(str).str.zfill(6).drop_duplicates().tolist()
            )
        else:
            raise ValueError(f'Unknown universe: {self.universe}')
        
        pd.DataFrame({'code': codes}).to_csv(cache_file, index=False)
        return codes

    def _slice(self, df):
        lo = pd.to_datetime(self.start_date)
        hi = pd.to_datetime(self.end_date)
        return df[(df.index >= lo) & (df.index <= hi)]

    def load_single_stock(self, stock_code, max_retries=3, allow_fallback=True):
        stock_code = str(stock_code).zfill(6)
        # The date range is part of the cache key. Keying on the code alone lets a run
        # with one date range silently serve stale, wrongly-ranged data to another.
        cache_file = self.cache_dir / f'{stock_code}_{self.start_date}_{self.end_date}.parquet'
        if cache_file.exists():
            return self._slice(pd.read_parquet(cache_file))

        if self.akshare_endpoint == "sina":
            return self._load_sina(stock_code, cache_file)

        for attempt in range(max_retries):
            try:
                df = ak.stock_zh_a_hist(
                    symbol=stock_code, period='daily',
                    start_date=self.start_date, end_date=self.end_date,
                    adjust=self.adjustment
                )
                if df is None or df.empty:
                    return None
                
                df['日期'] = pd.to_datetime(df['日期'])
                df = df.set_index('日期')
                df = df.rename(columns={
                    '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low',
                    '成交量': 'volume', '成交额': 'amount', '换手率': 'turnover'
                })
                df.to_parquet(cache_file)
                return df
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                print(f'Failed to load {stock_code}: {e}')
                break

        if not allow_fallback:
            return None

        # Eastmoney occasionally drops long historical requests. AKShare's
        # Sina-backed daily endpoint provides an independent, schema-compatible
        # fallback while keeping the data source inside AKShare.
        return self._load_sina(stock_code, cache_file)

    def _load_sina(self, stock_code, cache_file):
        try:
            exchange = "sh" if stock_code.startswith(("6", "9")) else "sz"
            df = ak.stock_zh_a_daily(
                symbol=f"{exchange}{stock_code}",
                start_date=self.start_date,
                end_date=self.end_date,
                adjust=self.adjustment,
            )
            if df is None or df.empty:
                return None
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df.to_parquet(cache_file)
            return df
        except Exception as e:
            print(f"Fallback failed for {stock_code}: {e}")
            return None

    def load_market_index(self, index_code='000300'):
        cache_file = self.cache_dir / f'index_{index_code}_{self.start_date}_{self.end_date}.parquet'
        if cache_file.exists():
            df = self._slice(pd.read_parquet(cache_file))
            return df['close'].pct_change().dropna()

        try:
            df = ak.stock_zh_index_daily(symbol=f'sh{index_code}')
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df = self._slice(df)
            df.to_parquet(cache_file)
            return df['close'].pct_change().dropna()
        except Exception as e:
            print(f'Failed to load index {index_code}: {e}')
            return None

    def winsorize(self, returns, reference=None):
        reference = returns if reference is None else reference
        if reference.empty:
            raise ValueError("Winsorization reference period is empty.")
        mean = reference.mean()
        std = reference.std()
        lower = mean - self.winsorize_std * std
        upper = mean + self.winsorize_std * std
        return returns.clip(lower, upper, axis=1)

    def _eligibility_slice(self, values):
        if not self.eligibility_end_date:
            return values
        cutoff = pd.to_datetime(self.eligibility_end_date)
        eligible = values.loc[values.index <= cutoff]
        if eligible.empty:
            raise ValueError(
                "eligibility_end_date precedes the available return data."
            )
        return eligible

    def _preprocessing_reference(self, values):
        if not self.preprocess_fit_end_date:
            return values
        cutoff = pd.to_datetime(self.preprocess_fit_end_date)
        reference = values.loc[values.index <= cutoff]
        if reference.empty:
            raise ValueError(
                "preprocess_fit_end_date precedes the available return data."
            )
        return reference

    def _has_minimum_eligibility_history(self, returns):
        if self.eligibility_end_date:
            cutoff = pd.to_datetime(self.eligibility_end_date)
            returns = returns.loc[returns.index <= cutoff]
        return len(returns) >= self.min_history

    def prepare_raw_returns(self, max_stocks=None):
        """Load an eligible aligned panel without fitting winsorization.

        Keeping this step separate allows a frozen historical panel to be
        combined with newly downloaded observations while retaining the
        original preprocessing reference.
        """
        stock_list = self.get_stock_list()
        if max_stocks:
            stock_list = stock_list[:max_stocks]
        
        print(f'Loading data for {len(stock_list)} stocks...')
        
        def load_return(code):
            df = self.load_single_stock(
                code, allow_fallback=self.download_workers == 1
            )
            if df is not None and 'close' in df.columns:
                returns = df['close'].pct_change(fill_method=None).dropna()
                if self._has_minimum_eligibility_history(returns):
                    return code, returns
            return code, None

        returns_dict = {}
        if self.download_workers == 1:
            loaded = (
                load_return(code)
                for code in tqdm(stock_list, desc="Loading stocks")
            )
        else:
            executor = ThreadPoolExecutor(max_workers=self.download_workers)
            loaded = tqdm(
                executor.map(load_return, stock_list),
                total=len(stock_list),
                desc=f"Loading stocks ({self.download_workers} workers)",
            )
        try:
            for code, returns in loaded:
                if returns is not None:
                    returns_dict[code] = returns
        finally:
            if self.download_workers != 1:
                executor.shutdown(wait=True)

        # AKShare's Sina adjustment engine embeds a native JavaScript runtime
        # that is not thread-safe. Retry misses serially after all Eastmoney
        # worker calls have completed.
        if self.download_workers != 1:
            missing = [code for code in stock_list if code not in returns_dict]
            if missing:
                print(
                    f"Serial AKShare fallback for {len(missing)} unavailable "
                    "Eastmoney histories..."
                )
            for code in tqdm(missing, desc="Fallback stocks"):
                df = self.load_single_stock(
                    code, max_retries=0, allow_fallback=True
                )
                if df is None or "close" not in df.columns:
                    continue
                returns = df["close"].pct_change(fill_method=None).dropna()
                if self._has_minimum_eligibility_history(returns):
                    returns_dict[code] = returns
        
        if len(returns_dict) < 2:
            raise ValueError(f'Only {len(returns_dict)} stocks loaded successfully, need at least 2')

        returns_df = pd.DataFrame(returns_dict).dropna(how='all')

        # A stock that listed mid-window has no returns before its IPO. Filling those days
        # with 0.0 fabricates flat returns: it shrinks the stock's variance, dilutes its
        # correlations toward zero, and corrupts its GARCH fit. Keep only stocks that
        # actually traded for (nearly) the whole window.
        eligibility = self._eligibility_slice(returns_df)
        coverage = eligibility.notna().mean()
        keep = coverage[coverage >= self.min_coverage].index
        dropped = len(returns_df.columns) - len(keep)
        if dropped:
            print(f'Dropping {dropped} stocks with <{self.min_coverage:.0%} history in window '
                  f'(late listings / long suspensions); {len(keep)} remain')
        returns_df = returns_df[keep]

        # Remaining gaps are short suspensions, where a zero return is the honest value.
        returns_df = returns_df.fillna(0)
        
        market_returns = self.load_market_index()
        if market_returns is None:
            market_returns = returns_df.mean(axis=1)
        
        common_idx = returns_df.index.intersection(market_returns.index)
        dropped = 1 - len(common_idx) / len(returns_df)
        if dropped > 0.05:
            print(
                f'WARNING: aligning to the market index drops {dropped:.0%} of stock days '
                f'({len(returns_df)} -> {len(common_idx)}). The index series likely covers a '
                f'shorter range than the stocks.'
            )
        returns_df = returns_df.loc[common_idx]
        market_returns = market_returns.loc[common_idx]

        return returns_df, market_returns

    def load_all_data(self, max_stocks=None):
        returns_df, market_returns = self.prepare_raw_returns(
            max_stocks=max_stocks
        )
        returns_df = self.winsorize(
            returns_df, self._preprocessing_reference(returns_df)
        )
        print(f'Loaded {len(returns_df.columns)} stocks, {len(returns_df)} trading days')
        return returns_df, market_returns


class SyntheticDataPipeline:
    """Generate a reproducible factor-model panel for offline smoke tests."""

    def __init__(self, config):
        self.config = config

    def load_all_data(self, max_stocks=None):
        n_stocks = max_stocks or self.config.data.synthetic_n_stocks
        n_days = self.config.data.synthetic_n_days
        if n_stocks < 2:
            raise ValueError("Synthetic data requires at least 2 stocks.")
        if n_days < 300:
            raise ValueError(
                "Synthetic data requires at least 300 days for the rolling factors."
            )

        rng = np.random.default_rng(self.config.data.random_seed)
        dates = pd.bdate_range(self.config.data.start_date, periods=n_days)

        market = rng.standard_t(7, n_days) * 0.012
        style = rng.normal(0.0, 0.008, (n_days, 2))
        loadings = rng.normal(0.0, 0.5, (n_stocks, 2))
        betas = rng.uniform(0.6, 1.4, n_stocks)

        # A persistent volatility state makes the smoke data exercise the GARCH path.
        log_vol = np.empty(n_days)
        log_vol[0] = np.log(0.012)
        shocks = rng.normal(0.0, 0.12, n_days)
        for i in range(1, n_days):
            log_vol[i] = 0.97 * log_vol[i - 1] + 0.03 * np.log(0.012) + shocks[i]
        idio = rng.standard_t(6, (n_days, n_stocks)) * np.exp(log_vol)[:, None]

        values = (
            market[:, None] * betas[None, :]
            + style @ loadings.T
            + idio
        )
        columns = [f"SYN_{i:03d}" for i in range(n_stocks)]
        returns = pd.DataFrame(values, index=dates, columns=columns)
        market_returns = pd.Series(market, index=dates, name="market")

        print(f"Generated {n_stocks} synthetic stocks, {n_days} trading days")
        return returns, market_returns
