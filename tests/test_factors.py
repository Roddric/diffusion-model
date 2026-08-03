"""Factor construction: conditioning, no-lookahead, and the idiosyncratic split."""

import numpy as np

from factors.extractor import FactorExtractor


def test_momentum_and_reversal_are_not_collinear(config, returns, market):
    """Regression: momentum defined as mean(252) - mean(21) shares its short leg with the
    reversal factor, making the two ~0.995 correlated and blowing the OLS loadings up into
    large offsetting betas."""
    factors = FactorExtractor(config).extract_mean_factors(returns, market)["factors"]

    corr = factors["momentum"].corr(factors["reversal"])
    assert abs(corr) < 0.9, f"momentum/reversal collinear (corr={corr:.3f})"


def test_no_factor_is_explained_by_the_others(config, returns, market):
    """Variance inflation factor: VIF_i = 1 / (1 - R^2) from regressing factor i on the rest.
    The usual flag is VIF > 10 (the collinear momentum/reversal pair scored ~100). Unlike the
    condition number of the raw design matrix, VIF is scale-invariant, so it does not fire
    merely because a factor is modelled in logs."""
    factors = FactorExtractor(config).extract_mean_factors(returns, market)["factors"]
    X = (factors - factors.mean()) / factors.std()

    for col in X.columns:
        y = X[col].values
        others = np.column_stack([np.ones(len(X)), X.drop(columns=[col]).values])
        resid = y - others @ np.linalg.lstsq(others, y, rcond=None)[0]
        r2 = 1 - resid.var() / y.var()
        vif = 1 / max(1 - r2, 1e-12)
        assert vif < 10, f"{col} is collinear with the other factors (VIF={vif:.1f})"


def test_factors_do_not_look_ahead(config, returns, market):
    """A factor value at date t must not change when returns after t change."""
    fe = FactorExtractor(config)
    base = fe.extract_mean_factors(returns, market)["factors"]

    tampered = returns.copy()
    tampered.iloc[-30:] *= 10.0  # blow up the future
    after = fe.extract_mean_factors(tampered, market)["factors"]

    cutoff = returns.index[-31]
    past = base.index[base.index <= cutoff]
    np.testing.assert_allclose(
        base.loc[past].values, after.loc[past].values, atol=1e-12
    )


def test_idiosyncratic_residual_is_standardized_by_log_volatility(
    config, returns, market
):
    """Residual innovations should be unit-scale after removing common log volatility."""
    fe = FactorExtractor(config)
    mfr = fe.extract_mean_factors(returns, market)
    resid = fe.compute_residuals(returns, mfr)
    vfr = fe.extract_volatility_factors(resid)

    idio = fe.compute_idiosyncratic(resid, vfr)

    assert 0.7 < idio.std().mean() < 1.3
    assert list(vfr["factors"].columns)[0] == "logvol_pc1"


def test_volatility_pca_uses_magnitude_not_residual_sign(config):
    rng = np.random.default_rng(11)
    idx = np.arange(500)
    scale = np.where(idx < 250, 0.005, 0.04)
    residuals = returns = np.column_stack(
        [rng.normal(0, scale) for _ in range(6)]
    )
    import pandas as pd

    frame = pd.DataFrame(
        residuals,
        index=pd.bdate_range("2020-01-01", periods=len(idx)),
        columns=[f"S{i}" for i in range(6)],
    )
    fe = FactorExtractor(config)
    vfr = fe.fit_volatility_factors(frame)
    log_var = fe.reconstruct_log_variance(vfr)

    assert log_var.iloc[-100:].mean().mean() > log_var.iloc[:100].mean().mean()


def test_transform_reuses_train_fitted_mappings(config, returns, market):
    split = 330
    train_r, test_r = returns.iloc[:split], returns.iloc[split:]
    train_m, test_m = market.iloc[:split], market.iloc[split:]

    fe = FactorExtractor(config)
    train_mean = fe.fit_mean_factors(train_r, train_m)
    fitted_loadings = train_mean["loadings"].copy()
    transformed = fe.transform_mean_factors(
        test_r, test_m, history_returns=train_r, history_market=train_m
    )

    np.testing.assert_allclose(transformed["loadings"], fitted_loadings)

    train_resid = fe.compute_residuals(train_r, train_mean)
    fe.fit_volatility_factors(train_resid)
    pca_components = fe.vol_pca_.components_.copy()
    test_resid = fe.compute_residuals(test_r, transformed)
    fe.transform_volatility_factors(
        test_resid, history_residuals=train_resid
    )
    np.testing.assert_allclose(fe.vol_pca_.components_, pca_components)
