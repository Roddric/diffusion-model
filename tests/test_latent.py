"""Latent parametrization: standardization and invertibility."""

import numpy as np
import pandas as pd

from latent.parametrizer import LatentParametrizer


def _factor_results(seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=120)
    # Deliberately mismatched scales, as in the real pipeline: mean factors ~1e-3,
    # PCA vol scores ~1e-1.
    mean = pd.DataFrame(rng.normal(0, 1e-3, (120, 4)), index=idx,
                        columns=["market", "momentum", "reversal", "dispersion"])
    vol = pd.DataFrame(rng.normal(0, 1e-1, (120, 3)), index=idx)
    return {"factors": mean}, {"factors": vol}


def test_latent_is_standardized(config):
    """Regression: an unstandardized latent mixes 1e-3 and 1e-1 scales, which a diffusion
    model trained against N(0,I) noise cannot represent."""
    mfr, vfr = _factor_results()
    latent, _ = LatentParametrizer(config).parametrize(mfr, vfr)

    assert np.allclose(latent.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(latent.std(axis=0), 1, atol=1e-6)


def test_inverse_transform_round_trips(config):
    """The latent the model samples must map back to the factor values it came from."""
    mfr, vfr = _factor_results()
    lp = LatentParametrizer(config)
    latent, meta = lp.parametrize(mfr, vfr)

    mean_out, vol_out = lp.inverse_transform(latent, meta)

    assert np.allclose(mean_out, mfr["factors"].values, atol=1e-8)
    assert np.allclose(vol_out, vfr["factors"].values, atol=1e-8)


def test_metadata_preserves_factor_split(config):
    mfr, vfr = _factor_results()
    _, meta = LatentParametrizer(config).parametrize(mfr, vfr)

    assert meta["n_mean_factors"] == 4
    assert meta["n_vol_factors"] == 3
    assert meta["mean_names"][0] == "market"


def test_transform_uses_training_scaler(config):
    train_mean, train_vol = _factor_results(seed=1)
    test_mean, test_vol = _factor_results(seed=2)
    test_mean["factors"] = test_mean["factors"] + 100
    test_vol["factors"] = test_vol["factors"] - 50

    lp = LatentParametrizer(config).fit(train_mean, train_vol)
    train_loc = lp.metadata_["loc"].copy()
    train_scale = lp.metadata_["scale"].copy()
    transformed, metadata = lp.transform(test_mean, test_vol)

    np.testing.assert_allclose(metadata["loc"], train_loc)
    np.testing.assert_allclose(metadata["scale"], train_scale)
    assert abs(transformed.mean()) > 10
