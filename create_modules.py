import pathlib

# ============================================================================
# DATA LAYER
# ============================================================================

data_loader_code = """\"\"\"Data loading and preprocessing for Chinese A-shares.\"\"\"

import akshare as ak
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm
import warnings

class AShareDataLoader:
    \"\"\"Load and preprocess Chinese A-share stock data using AKShare.\"\"\"
    
    def __init__(self, config):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def load_universe(self, universe: str = "hs300") -> List[str]:
        \"\"\"Load stock universe constituents.\"\"\"
        if universe == "hs300":
            df = ak.index_stock_cons(symbol="000300")
            return df['品种代码'].tolist()
        elif universe == "hs500":
            df = ak.index_stock_cons(symbol="000905")
            return df['品种代码'].tolist()
        elif universe == "hs1000":
            df = ak.index_stock_cons(symbol="000852")
            return df['品种代码'].tolist()
        else:
            raise ValueError(f"Unknown universe: {universe}")
    
    def load_stock_data(self, stock_list: List[str]) -> pd.DataFrame:
        \"\"\"Load daily OHLCV data for all stocks.\"\"\"
        all_data = []
        cache_file = self.data_dir / f"raw_data_{self.config.data.start_date}_{self.config.data.end_date}.parquet"
        
        if cache_file.exists():
            print(f"Loading cached data from {cache_file}")
            return pd.read_parquet(cache_file)
        
        print(f"Downloading data for {len(stock_list)} stocks...")
        for stock_code in tqdm(stock_list, desc="Loading stocks"):
            try:
                df = ak.stock_zh_a_hist(
                    symbol=stock_code,
                    period="daily",
                    start_date=self.config.data.start_date,
                    end_date=self.config.data.end_date,
                    adjust="hfq"  # 后复权
                )
                df['stock_code'] = stock_code
                all_data.append(df)
            except Exception as e:
                warnings.warn(f"Failed to load {stock_code}: {e}")
                continue
        
        if not all_data:
            raise RuntimeError("No data loaded")
        
        combined = pd.concat(all_data, ignore_index=True)
        combined['日期'] = pd.to_datetime(combined['日期'])
        combined = combined.set_index(['日期', 'stock_code'])
        
        # Cache to disk
        combined.to_parquet(cache_file)
        print(f"Cached data to {cache_file}")
        
        return combined
    
    def load_index_data(self, index_code: str = "000300") -> pd.DataFrame:
        \"\"\"Load index data (e.g., CSI 300) for market factor.\"\"\"
        df = ak.stock_zh_index_daily(symbol=f"sh{index_code}")
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        return df
    
    def load_market_data(self) -> pd.DataFrame:
        \"\"\"Load market-wide data for conditioning.\"\"\"
        # CSI 300 as market proxy
        csi300 = self.load_index_data("000300")
        csi300['market_return'] = csi300['close'].pct_change()
        
        # Add macro indicators if available
        # GDP, CPI, PMI can be added here
        
        return csi300
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Preprocess raw data.\"\"\"
        # Calculate returns
        df = df.copy()
        df['returns'] = df.groupby('stock_code')['收盘'].pct_change()
        
        # Calculate additional features
        df['log_volume'] = np.log1p(df['成交量'])
        df['turnover'] = df['换手率'] / 100
        
        # Handle limit days (涨跌停)
        df['limit_flag'] = self._flag_limit_days(df)
        
        # Drop NaN and limit days
        df = df[~df['limit_flag']].dropna()
        
        # Winsorize extreme returns
        df = self._winsorize_returns(df, 'returns', self.config.data.winsorize_std)
        
        # Pivot to wide format
        returns_wide = df.reset_index().pivot(
            index='日期', 
            columns='stock_code', 
            values='returns'
        )
        
        # Filter stocks with enough history
        valid_stocks = returns_wide.dropna(axis=1, thresh=self.config.data.min_history).columns
        returns_wide = returns_wide[valid_stocks]
        
        print(f"Final universe: {len(valid_stocks)} stocks")
        
        return returns_wide
    
    def _flag_limit_days(self, df: pd.DataFrame) -> pd.Series:
        \"\"\"Flag days with limit up/down (±10% or ±20% for ChiNext/STAR).\"\"\"
        # Simplified: flag returns > 9.5% or < -9.5%
        return df['returns'].abs() > 0.095
    
    def _winsorize_returns(self, df: pd.DataFrame, col: str, n_std: float) -> pd.DataFrame:
        \"\"\"Winsorize returns at n standard deviations.\"\"\"
        mean = df[col].mean()
        std = df[col].std()
        lower = mean - n_std * std
        upper = mean + n_std * std
        df[col] = df[col].clip(lower, upper)
        return df
    
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        \"\"\"Load all data needed for the model.\"\"\"
        # Load universe
        stock_list = self.load_universe(self.config.data.universe)
        print(f"Universe: {len(stock_list)} stocks")
        
        # Load stock data
        raw_data = self.load_stock_data(stock_list)
        
        # Preprocess
        returns = self.preprocess(raw_data)
        
        # Load market data
        market_data = self.load_market_data()
        
        # Load factor characteristics (size, value, etc.)
        characteristics = self._load_characteristics(stock_list)
        
        return returns, market_data, characteristics
    
    def _load_characteristics(self, stock_list: List[str]) -> pd.DataFrame:
        \"\"\"Load stock characteristics for factor construction.\"\"\"
        # This would load market cap, book-to-market, etc.
        # For now, return empty DataFrame
        return pd.DataFrame()

class SyntheticDataLoader:
    \"\"\"Generate synthetic data for testing.\"\"\"
    
    def __init__(self, config):
        self.config = config
        
    def generate(self, n_stocks: int = 100, n_days: int = 2520, 
                 n_factors: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        \"\"\"Generate synthetic returns from factor model.\"\"\"
        np.random.seed(42)
        
        # Generate factor returns
        factor_returns = pd.DataFrame(
            np.random.randn(n_days, n_factors) * 0.02,
            index=pd.date_range('2015-01-01', periods=n_days, freq='D'),
            columns=[f'F{i}' for i in range(n_factors)]
        )
        
        # Generate factor loadings
        betas = np.random.randn(n_stocks, n_factors) * 0.5
        
        # Generate idiosyncratic returns
        idio = np.random.randn(n_days, n_stocks) * 0.03
        
        # Combine
        returns = pd.DataFrame(
            factor_returns.values @ betas.T + idio,
            index=factor_returns.index,
            columns=[f'S{i:03d}' for i in range(n_stocks)]
        )
        
        # Synthetic market data
        market_data = pd.DataFrame(
            {'market_return': factor_returns['F0']},
            index=factor_returns.index
        )
        
        return returns, market_data
"""

pathlib.Path('diffusion_factor_model/data/loader.py').write_text(data_loader_code)
print("Created data/loader.py")

# ============================================================================
# FACTOR EXTRACTION
# ============================================================================

factor_code = """\"\"\"Factor extraction: mean factors and volatility factors.\"\"\"

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from typing import Tuple, Dict
import statsmodels.api as sm

class FactorExtractor:
    \"\"\"Extract mean and volatility factors from returns.\"\"\"
    
    def __init__(self, config):
        self.config = config
        self.n_vol_factors = config.factors.n_vol_factors
        self.explained_var_threshold = config.factors.pca_explained_var_threshold
        
    def extract_mean_factors(self, returns: pd.DataFrame, 
                            characteristics: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        \"\"\"Extract mean factors using cross-sectional regression (Fama-MacBeth).\"\"\"
        # For now, use market factor as proxy
        # Full implementation would use size, value, momentum, etc.
        
        market_return = returns.mean(axis=1)
        market_return.name = 'MKT'
        
        # Calculate size factor (small minus big)
        # Would need market cap data
        
        # Calculate value factor (high BM minus low BM)
        # Would need book-to-market data
        
        # Calculate momentum factor
        # Would need past returns
        
        mean_factors = pd.DataFrame({
            'MKT': market_return
        })
        
        # Time-series regression to get betas
        betas_mu = {}
        for stock in returns.columns:
            y = returns[stock].dropna()
            X = sm.add_constant(mean_factors.loc[y.index])
            model = sm.OLS(y, X).fit()
            betas_mu[stock] = model.params.iloc[1:]  # exclude constant
        
        betas_mu_df = pd.DataFrame(betas_mu).T
        
        return {
            'factors': mean_factors,
            'loadings': betas_mu_df,
            'residuals': returns - mean_factors['MKT'].values.reshape(-1, 1) * betas_mu_df['MKT'].values
        }
    
    def extract_vol_factors(self, returns: pd.DataFrame) -> Dict[str, np.ndarray]:
        \"\"\"Extract volatility factors using PCA on residuals.\"\"\"
        # First extract mean factors to get residuals
        mean_result = self.extract_mean_factors(returns, pd.DataFrame())
        residuals = mean_result['residuals'].fillna(0)
        
        # PCA on residuals
        n_components = min(self.n_vol_factors, min(residuals.shape) - 1)
        pca = PCA(n_components=n_components)
        vol_factors = pca.fit_transform(residuals)
        
        # Calculate explained variance
        explained_var = np.cumsum(pca.explained_variance_ratio_)
        
        print(f"Vol factors explain {explained_var[-1]:.2%} of variance")
        
        # Factor loadings
        loadings = pca.components_.T  # (n_stocks, n_factors)
        
        return {
            'factors': vol_factors,
            'loadings': loadings,
            'explained_var': explained_var,
            'pca': pca
        }
    
    def extract_all(self, returns: pd.DataFrame, 
                   characteristics: pd.DataFrame = None) -> Dict:
        \"\"\"Extract all factors.\"\"\"
        mean_result = self.extract_mean_factors(returns, characteristics or pd.DataFrame())
        vol_result = self.extract_vol_factors(returns)
        
        return {
            'mean': mean_result,
            'vol': vol_result,
            'returns': returns
        }

class RobustFactorExtractor:
    \"\"\"Robust factor extraction using Ledoit-Wolf shrinkage.\"\"\"
    
    def __init__(self, config):
        self.config = config
        self.shrinkage = LedoitWolf()
        
    def robust_pca(self, returns: pd.DataFrame, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
        \"\"\"PCA with Ledoit-Wolf shrinkage for robust covariance.\"\"\"
        # Shrink covariance
        cov_shrunk = self.shrinkage.fit(returns).covariance_
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_shrunk)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Take top n_components
        loadings = eigenvectors[:, :n_components]
        
        # Project returns
        factors = returns.values @ loadings
        
        return factors, loadings
"""

pathlib.Path('diffusion_factor_model/factors/extractor.py').write_text(factor_code)
print("Created factors/extractor.py")

# ============================================================================
# LATENT PARAMETRIZATION
# ============================================================================

latent_code = """\"\"\"Latent parametrization for diffusion model.\"\"\"

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from typing import Tuple, Dict

class LatentParametrizer:
    \"\"\"Convert factor returns to latent representation for diffusion.\"\"\"
    
    def __init__(self, config):
        self.config = config
        self.use_log_vol = config.latent.use_log_vol
        self.shrinkage_method = config.latent.shrinkage
        
    def parametrize(self, factor_result: Dict) -> Dict[str, np.ndarray]:
        \"\"\"Convert factors to latent vector Z_t.\"\"\"
        mean_factors = factor_result['mean']['factors']
        vol_factors = factor_result['vol']['factors']
        
        # Concatenate mean factors and volatility factors
        if self.use_log_vol:
            # Use log-volatility for PSD guarantee
            vol_factors = np.log(np.abs(vol_factors) + 1e-8)
        
        # Combine into latent vector
        Z = np.hstack([
            mean_factors.values,
            vol_factors
        ])
        
        print(f"Latent dimension: {Z.shape[1]}")
        print(f"  - Mean factors: {mean_factors.shape[1]}")
        print(f"  - Vol factors: {vol_factors.shape[1]}")
        
        return {
            'latent': Z,
            'mean_factors': mean_factors.values,
            'vol_factors': vol_factors,
            'loadings': factor_result['vol']['loadings']
        }
    
    def reconstruct_covariance(self, Z: np.ndarray, loadings: np.ndarray,
                               idio_var: np.ndarray) -> np.ndarray:
        \"\"\"Reconstruct covariance matrix from latent representation.
        
        Args:
            Z: latent vector (T, k_mean + k_vol)
            loadings: factor loadings (n_stocks, k_vol)
            idio_var: idiosyncratic variance (n_stocks,)
            
        Returns:
            Sigma: covariance matrix (n_stocks, n_stocks)
        \"\"\"
        # Extract volatility part
        if self.use_log_vol:
            vol_factor = np.exp(Z[:, -loadings.shape[1]:])
        else:
            vol_factor = Z[:, -loadings.shape[1]:]
        
        # Reconstruct factor covariance
        # For each time step
        n_stocks = loadings.shape[0]
        n_times = Z.shape[0]
        
        Sigma_list = []
        for t in range(n_times):
            # Factor covariance at time t
            F_t = np.diag(vol_factor[t])
            factor_cov = loadings @ F_t @ loadings.T
            
            # Add idiosyncratic variance
            Sigma_t = factor_cov + np.diag(idio_var)
            
            # Apply shrinkage if needed
            if self.shrinkage_method == "ledoit_wolf":
                lw = LedoitWolf()
                Sigma_t = lw.shrinkage(Sigma_t)
            
            Sigma_list.append(Sigma_t)
        
        return np.array(Sigma_list)
    
    def inverse_transform(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        \"\"\"Inverse transform latent to mean and vol factors.\"\"\"
        # Split mean and vol
        n_mean = self.config.factors.mean_factors.shape[0] if hasattr(self.config.factors.mean_factors, 'shape') else 1
        
        mean_factors = Z[:, :n_mean]
        vol_factors = Z[:, n_mean:]
        
        if self.use_log_vol:
            vol_factors = np.exp(vol_factors)
        
        return mean_factors, vol_factors
"""

pathlib.Path('diffusion_factor_model/latent/parametrizer.py').write_text(latent_code)
print("Created latent/parametrizer.py")

# ============================================================================
# DIFFUSION MODEL - SDE
# ============================================================================

sde_code = """\"\"\"Stochastic Differential Equations for diffusion models.\"\"\"

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Callable
from abc import ABC, abstractmethod

class SDE(ABC):
    \"\"\"Base class for SDE.\"\"\"
    
    @abstractmethod
    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def diffusion(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def marginal_params(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
    
    def marginal(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        \"\"\"Sample from marginal distribution p(x_t | x_0).\"\"\"
        mean, std = self.marginal_params(x0, t)
        eps = torch.randn_like(x0)
        return mean + std * eps

class VPSDE(SDE):
    \"\"\"Variance-Preserving SDE (DDPM-style).\"\"\"
    
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, T: float = 1.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        \"\"\"Noise schedule.\"\"\"
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def integral_beta(self, t: torch.Tensor) -> torch.Tensor:
        \"\"\"Integral of beta from 0 to t.\"\"\"
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2
    
    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        \"\"\"Drift coefficient: -0.5 * beta(t) * x.\"\"\"
        return -0.5 * self.beta(t).unsqueeze(-1) * x
    
    def diffusion(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        \"\"\"Diffusion coefficient: sqrt(beta(t)).\"\"\"
        return torch.sqrt(self.beta(t)).unsqueeze(-1)
    
    def marginal_params(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        \"\"\"Parameters of marginal p(x_t | x_0).\"\"\"
        log_mean_coeff = -0.5 * self.integral_beta(t).unsqueeze(-1)
        mean = torch.exp(log_mean_coeff) * x0
        std = torch.sqrt(1 - torch.exp(2 * log_mean_coeff))
        return mean, std

class VESDE(SDE):
    \"\"\"Variance-Exploding SDE (SDEdit-style).\"\"\"
    
    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0, T: float = 1.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.T = T
        
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        \"\"\"Noise schedule.\"\"\"
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    
    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        \"\"\"Drift is zero for VE-SDE.\"\"\"
        return torch.zeros_like(x)
    
    def diffusion(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        \"\"\"Diffusion coefficient.\"\"\"
        sigma = self.sigma(t)
        dsigma = sigma * torch.log(torch.tensor(self.sigma_max / self.sigma_min))
        return torch.sqrt(2 * torch.log(sigma) * dsigma).unsqueeze(-1)
    
    def marginal_params(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        \"\"\"Parameters of marginal p(x_t | x_0).\"\"\"
        mean = x0
        std = self.sigma(t).unsqueeze(-1)
        return mean, std

def get_sde(config) -> SDE:
    \"\"\"Factory function to create SDE.\"\"\"
    if config.diffusion.sde_type == "vp":
        return VPSDE(
            beta_min=config.diffusion.beta_min,
            beta_max=config.diffusion.beta_max,
            T=config.diffusion.T
        )
    elif config.diffusion.sde_type == "ve":
        return VESDE(
            sigma_min=config.diffusion.sigma_min,
            sigma_max=config.diffusion.sigma_max,
            T=config.diffusion.T
        )
    else:
        raise ValueError(f"Unknown SDE type: {config.diffusion.sde_type}")
"""

pathlib.Path('diffusion_factor_model/diffusion/sde.py').write_text(sde_code)
print("Created diffusion/sde.py")

print("Created core modules (part 1/2)")
