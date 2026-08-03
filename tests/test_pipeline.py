"""Offline end-to-end pipeline smoke test."""

import pandas as pd

from pipeline import run_pipeline


def test_pipeline_is_importable_as_a_package():
    from diffusion_factor_model.pipeline import run_pipeline as package_run_pipeline

    assert callable(package_run_pipeline)


def test_synthetic_pipeline_generates_configured_sample_count(tmp_path):
    data_dir = tmp_path / "output"
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(
        f"""
data:
  source: synthetic
  start_date: "2020-01-01"
  synthetic_n_stocks: 6
  synthetic_n_days: 320
  random_seed: 7
factors:
  n_vol_factors: 2
diffusion:
  n_epochs: 2
  batch_size: 32
  hidden_dim: 16
  n_layers: 2
  dropout: 0.0
residuals:
  min_obs: 30
sampling:
  solver: dpm_solver_pp
  n_steps: 2
  n_samples: 12
data_dir: "{data_dir}"
""",
        encoding="utf-8",
    )

    run_pipeline(str(config_path), generate_samples=True)

    generated = pd.read_csv(data_dir / "generated_returns.csv", index_col=0)
    assert generated.shape == (12, 6)
    assert (data_dir / "score_net.pt").exists()
    assert (data_dir / "evaluation_plots.png").exists()
