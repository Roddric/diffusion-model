"""Configuration loading, compatibility aliases, and typo visibility."""

from pathlib import Path
import warnings

import pytest

from config.config import Config


def test_example_config_uses_supported_keys():
    path = Path(__file__).resolve().parents[1] / "config_example.yaml"

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        config = Config.from_yaml(path)

    assert not captured
    assert config.data.source == "synthetic"
    assert config.factors.n_vol_factors == 5
    assert config.diffusion.n_layers == 6
    assert config.sampling.solver == "dpm_solver_pp"
    assert config.sampling.n_steps == 20
    assert config.sampling.n_samples == 1000
    assert config.data_dir == "./data"


def test_deprecated_aliases_are_applied(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text(
        """
data:
  source: synthetic
  data_dir: ./legacy-data
factors:
  num_vol_factors: 4
latent:
  shrinkage_method: oracle
diffusion:
  num_layers: 3
sampling:
  method: euler_maruyama
  num_steps: 12
  num_samples: 25
""",
        encoding="utf-8",
    )

    with pytest.warns(FutureWarning):
        config = Config.from_yaml(path)

    assert config.data_dir == "./legacy-data"
    assert config.factors.n_vol_factors == 4
    assert config.latent.shrinkage == "oracle"
    assert config.diffusion.n_layers == 3
    assert config.sampling.solver == "euler_maruyama"
    assert config.sampling.n_steps == 12
    assert config.sampling.n_samples == 25


def test_unsupported_key_emits_warning(tmp_path):
    path = tmp_path / "typo.yaml"
    path.write_text("diffusion:\n  hidden_dims: 999\n", encoding="utf-8")

    with pytest.warns(UserWarning, match="diffusion.hidden_dims"):
        config = Config.from_yaml(path)

    assert config.diffusion.hidden_dim == 256
