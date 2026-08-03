"""Return reconstruction from sampled latents."""

import numpy as np
import pandas as pd
import pytest

from reconstruction.reconstructor import ReturnReconstructor

STOCKS = ["600000", "600001", "600002"]


@pytest.fixture
def loadings():
    rng = np.random.default_rng(2)
    mfr = {
        "loadings": pd.DataFrame(
            rng.normal(1, 0.2, (3, 2)),
            index=STOCKS,
            columns=["market", "momentum"],
        ),
        "intercepts": pd.Series(0.0, index=STOCKS),
    }
    vfr = {
        "loadings": pd.DataFrame(
            rng.normal(0, 0.5, (3, 2)),
            index=STOCKS,
            columns=["logvol_pc1", "logvol_pc2"],
        ),
        "log_variance_mean": pd.Series(0.0, index=STOCKS),
        "log_variance_bounds": pd.DataFrame(
            {"lower": -10.0, "upper": 10.0}, index=STOCKS
        ),
    }
    return mfr, vfr


def test_volatility_factors_affect_output(config, loadings):
    """Log-volatility factors must scale innovations, not add expected returns."""
    mfr, vfr = loadings
    rec = ReturnReconstructor(config, mfr, vfr)

    base = np.zeros((5, 4))
    perturbed = base.copy()
    perturbed[:, 2:] = 1.0
    innovations = pd.DataFrame(1.0, index=range(5), columns=STOCKS)

    r_base = rec.reconstruct(base, innovations)
    r_pert = rec.reconstruct(perturbed, innovations)

    assert not np.allclose(r_base.values, r_pert.values)


def test_volatility_does_not_change_conditional_mean(config, loadings):
    mfr, vfr = loadings
    rec = ReturnReconstructor(config, mfr, vfr)
    base = np.zeros((5, 4))
    perturbed = base.copy()
    perturbed[:, 2:] = 2.0

    np.testing.assert_allclose(
        rec.reconstruct(base, None), rec.reconstruct(perturbed, None)
    )


def test_mean_factors_affect_output(config, loadings):
    mfr, vfr = loadings
    rec = ReturnReconstructor(config, mfr, vfr)

    base = np.zeros((5, 4))
    perturbed = base.copy()
    perturbed[:, :2] = 1.0

    assert not np.allclose(
        rec.reconstruct(base, None).values, rec.reconstruct(perturbed, None).values
    )


def test_residuals_align_by_name_not_position(config, loadings):
    """Regression: residuals were sliced positionally, so a residual frame whose columns
    came back in a different order was silently attached to the wrong stocks."""
    mfr, vfr = loadings
    rec = ReturnReconstructor(config, mfr, vfr)

    latent = np.zeros((4, 4))
    resid = pd.DataFrame(
        {"600000": [0.1] * 4, "600001": [0.2] * 4, "600002": [0.3] * 4}
    )
    shuffled = resid[["600002", "600000", "600001"]]

    in_order = rec.reconstruct(latent, resid)
    out_of_order = rec.reconstruct(latent, shuffled)

    pd.testing.assert_frame_equal(in_order, out_of_order)
    assert in_order["600000"].iloc[0] == pytest.approx(0.1)


def test_output_columns_are_stock_names(config, loadings):
    mfr, vfr = loadings
    rec = ReturnReconstructor(config, mfr, vfr)
    out = rec.reconstruct(np.zeros((6, 4)), None)

    assert list(out.columns) == STOCKS
    assert len(out) == 6
