import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# The package imports its own submodules as top-level ("from config.config import ...")
# so it must be run with diffusion_factor_model/ on the path.
PKG = Path(__file__).resolve().parents[1] / "diffusion_factor_model"
sys.path.insert(0, str(PKG))

from config.config import Config  # noqa: E402


@pytest.fixture
def config():
    cfg = Config()
    cfg.factors.n_vol_factors = 3
    return cfg


@pytest.fixture
def returns():
    """Synthetic panel with a real factor structure. Long enough for the 252d rollings."""
    rng = np.random.default_rng(0)
    n_days, n_stocks = 400, 8
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    market = rng.normal(0, 0.012, n_days)
    betas = rng.uniform(0.6, 1.4, n_stocks)
    idio = rng.normal(0, 0.015, (n_days, n_stocks))
    data = market[:, None] * betas[None, :] + idio
    codes = [f"{600000 + i:06d}" for i in range(n_stocks)]
    return pd.DataFrame(data, index=dates, columns=codes)


@pytest.fixture
def market(returns):
    return returns.mean(axis=1)
