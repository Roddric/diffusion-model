# Diffusion-Enhanced Factor-State Forecasting

Research code for a validation-selected pool of classical VAR and conditional
diffusion forecasts. The active study evaluates S&P 500 latent factor-state paths;
the original Chinese A-share synthetic-generation experiments are retained as
legacy development history.

## Current research status

The authoritative write-up is `Research_Paper_Draft.md`. The primary result is a
**retrospective locked holdout**, not a prospectively preregistered confirmation:

- the December 2023 universe archive, protocol, checkpoints, and decision rule were
  locked on 2026-07-30 before post-2023 observations were loaded into the scoring
  pipeline;
- the evaluated 2024–2026 period had already occurred, so the study cannot claim a
  prospective freeze;
- the VAR–diffusion pool improves state energy by 1.83% and state RMSE by 2.14%
  relative to Gaussian VAR, but is statistically tied with Student-t VAR;
- 100-path post-hoc scoring supports the Gaussian comparison and beats several
  added baselines, but remains inconclusive against Student-t VAR;
- it shows no consistent observable portfolio-risk, return-tail, or drawdown
  improvement, and 5% VaR pinball loss is worse than Gaussian VAR;
- the CSI 300 cross-market promotion rule fails;
- a post-hoc calibration/power audit reproduces the locked composite exactly,
  shows the pool's Gaussian-relative gain comes from the ensemble-spread term
  rather than mean-path accuracy, and finds the pool also separates from
  Student-t VAR at 114 stride-5 overlapping origins under HAC and moving-block
  inference (post-hoc sensitivity evidence only);
- a preregistered, externally notarized FTSE 100 untouched-market evaluation
  reproduces the pattern on 2024–2026: the Gaussian-base pool passes the primary
  rule with composite 0.98204 versus VAR-GARCH (HAC p=0.0017 on both co-primary
  metrics), and the Student-t-base pool passes its prespecified rule
  (composite 0.99631) but is statistically tied with Student-t VAR, mirroring the
  S&P result;
- the matching FTSE post-hoc audit reproduces both locked composites exactly and
  shows the same calibration signature (pool edge coverage at the uniform value,
  VAR baselines under-dispersed); at 116 overlapping origins the pool stays
  significant versus Gaussian VAR (HAC p≈0.03) and tied with Student-t VAR. The
  consumed FTSE sample may only support post-hoc audits.

Primary artifacts:

- `research_output/sp500_frozen/frozen_protocol.json`
- `research_output/sp500_confirmation/confirmation.json`
- `research_output/sp500_confirmation/posthoc_dependence_robustness.json`
- `research_output/sp500_confirmation/posthoc_monte_carlo_sensitivity.json`
- `research_output/sp500_confirmation/posthoc_monte_carlo_seed_variation.json`
- `research_output/sp500_confirmation/posthoc_expanded_baselines.json`
- `research_output/sp500_confirmation/posthoc_observable_risk_audit.json`
- `research_output/sp500_confirmation/posthoc_calibration_power_audit.json`
- `research_output/sp500_confirmation/paired_origin_robustness.png`
- `research_output/ftse100_frozen/ftse100_external.protocol.json` (preregistered protocol)
- `research_output/ftse100_frozen/frozen_protocol.json` (one-time FTSE freeze)
- `research_output/ftse100_confirmation/confirmation.json` (one-time FTSE 2024–2026 result)
- `research_output/ftse100_confirmation/posthoc_ftse_calibration_power_audit.json`
- `REPRODUCIBILITY.md` and `ARTIFACT_MANIFEST.sha256`
- `PROFESSOR_BRIEF.md`

Do not use the consumed S&P, CSI, or FTSE 2024–2026 observations for further model
selection. New confirmatory evidence requires a genuinely untouched market or a
future prospective evaluation period.

## Implementation overview

The codebase implements a **Diffusion Factor Model** that:
- Extracts market, momentum, reversal, dispersion, and illiquidity factors
- Identifies common log-volatility states using PCA of causal realized variance
- Trains a score-based diffusion model on the latent factor space
- Generates high-dimensional synthetic returns with realistic statistical properties

The early implementation was based on "Diffusion Factor Models" (Chen et al.,
2026) and adapted for Chinese A-share experiments. Those early results below are
diagnostic history, not the active headline result.

## Architecture

```
Raw Returns → Mean Factors + Log-Volatility States → Latent Dynamics
            → Diffusion/VAR Sampling → Volatility-Scaled Reconstruction
```

**Key Components:**
1. **Factor Extraction**: Mean factors plus PCA of causal rolling log residual variance
2. **Latent Space**: Train-standardized mean and log-volatility states
3. **Score Network**: Residual MLP with FiLM conditioning
4. **Diffusion Model**: VP-SDE with DPM-Solver++ sampling
5. **Residual Modeling**: GJR-GARCH(1,1) for standardized innovations
6. **Dynamic Baseline**: Stable VAR(1) factor states with GARCH innovations

## Installation

```bash
cd diffusion-model

# Install dependencies
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

For exact reproduction of the audited 2026-08-03 environment, install
`requirements-lock.txt` instead. The shorter requirements files intentionally
describe supported minimum versions rather than the frozen research environment.

## Quick Start

### 1. Run with Synthetic Data (Testing)

```bash
python diffusion_factor_model/pipeline.py --config config_example.yaml --generate
```

This will:
- Generate synthetic factor model data
- Extract factors
- Train diffusion model (1000 epochs)
- Generate 1000 synthetic samples
- Evaluate and save results

### 2. Run with Real A-Share Data

```bash
python diffusion_factor_model/pipeline.py --config prod_config.yaml --generate
```

### 3. Custom Configuration

```bash
# Create your config
cp config_example.yaml my_config.yaml

# Edit settings (universe, date range, model parameters)
python diffusion_factor_model/pipeline.py --config my_config.yaml --generate
```

## Project Structure

```
diffusion-model/
├── diffusion_factor_model/
│   ├── config.py              # Configuration classes
│   ├── pipeline.py            # Main orchestration
│   ├── data/
│   │   └── loader.py          # AKShare + synthetic data loaders
│   ├── factors/
│   │   └── extractor.py       # Mean + volatility factor extraction
│   ├── latent/
│   │   └── parametrizer.py    # Latent space construction
│   ├── diffusion/
│   │   ├── sde.py             # VP-SDE implementation
│   │   ├── score_net.py       # Residual MLP score network
│   │   ├── conditional_path.py # Context-conditioned path diffusion
│   │   └── trainer.py         # Training loop
│   ├── dynamics/
│   │   └── var.py             # VAR/GARCH factor-state baseline
│   ├── residuals/
│   │   └── garch.py           # GJR-GARCH modeling
│   ├── sampling/
│   │   └── sampler.py         # DPM-Solver++ + Euler-Maruyama
│   ├── reconstruction/
│   │   └── reconstructor.py   # Return reconstruction
│   ├── sequences/
│   │   ├── dataset.py         # Leak-free 60/20 temporal windows
│   │   ├── baselines.py       # VAR/GARCH and block path forecasts
│   │   └── evaluation.py      # Ensemble path scoring
│   ├── evaluation/
│   │   └── metrics.py         # Statistical metrics + visualization
│   ├── phase1_benchmark.py    # Strict temporal before/after comparison
│   ├── phase2a_benchmark.py   # Rolling-origin sequence benchmark
│   ├── phase2b_benchmark.py   # Conditional diffusion gate benchmark
│   ├── phase2c_benchmark.py   # Split-head, multi-seed benchmark
│   └── phase2d_benchmark.py   # Validation path-energy selection
├── config_example.yaml        # Example configuration
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Configuration

Key parameters in `config_example.yaml`:

**Data:**
- `source`: "akshare" (real data) or "synthetic" (testing)
- `universe`: "hs300", "hs500", "hs1000", or "all"
- `start_date` / `end_date`: Date range

**Factors:**
- `n_vol_factors`: Number of PCA components
- `vol_window`: Causal realized-variance window before log-volatility PCA

**Diffusion:**
- `beta_min` / `beta_max`: VP-SDE noise schedule
- `hidden_dim`: Score network width (default: 256)
- `n_layers`: Score network depth (default: 6)
- `n_epochs`: Training iterations (default: 1000)
- `batch_size`: Training batch size (default: 128)

**Sampling:**
- `solver`: `"dpm_solver_pp"` (fast) or `"euler_maruyama"` (baseline)
- `n_steps`: Sampling steps (default: 20 for DPM-Solver++)
- `n_samples`: Number of samples to generate

**Temporal diffusion** (in `us_config.yaml`):
- `context_length`: Observed state-history length
- `horizon`: Generated future-path length
- `training_steps`: Maximum optimization budget before validation stopping
- `sampling_steps`: DDIM path-sampling steps
- `ensemble_paths`: Forecast paths per rolling origin
- `split_output_heads`: Separate mean-factor and volatility-factor outputs
- `mean_loss_weight` / `volatility_loss_weight`: Family-level denoising weights
- `validation_seeds`: Independent seeds aggregated without test-set selection
- `path_validation_paths` / `path_validation_steps`: Validation-only sampling budget

Paths such as `data_dir`, `model_dir`, and `output_dir` are top-level settings.
Deprecated configuration names are accepted with a warning; unsupported names also emit
a warning instead of being silently ignored.

## Usage Examples

### Training Only

```python
from diffusion_factor_model.pipeline import run_pipeline

run_pipeline(config_path="my_config.yaml", generate_samples=False)
```

### Load Pretrained Model & Generate

```python
from diffusion_factor_model.config import load_config
from diffusion_factor_model.diffusion.score_net import ScoreNetwork
from diffusion_factor_model.sampling.sampler import DPMSolverPlusPlus
import torch

config = load_config("my_config.yaml")
score_net = ScoreNetwork(config, latent_dim=11)
score_net.load_state_dict(torch.load("data/score_net_epoch_1000.pth"))

sampler = DPMSolverPlusPlus(config, score_net)
samples = sampler.sample(num_samples=5000, num_steps=20)
```

### Custom Factor Extraction

```python
from diffusion_factor_model.factors.extractor import FactorExtractor
from diffusion_factor_model.data.loader import DataPipeline
from diffusion_factor_model.config import load_config

config = load_config("my_config.yaml")
data_pipeline = DataPipeline(config)
returns, characteristics = data_pipeline.load_all_data()

extractor = FactorExtractor(config)
mean_factors = extractor.extract_mean_factors(returns, characteristics)
residuals = extractor.compute_residuals(returns, mean_factors)
vol_factors = extractor.extract_volatility_factors(residuals)
```

## Evaluation Metrics

The pipeline computes:

1. **Correlation Distance**: Frobenius norm between real and generated correlation matrices
2. **Volatility ACF**: Autocorrelation of absolute returns (volatility clustering)
3. **Tail Dependence**: Lower tail dependence coefficient
4. **Distribution Moments**: Mean, std, skewness, kurtosis

Plots are saved to `data/evaluation_plots.png`.

## Legacy synthetic-generation performance

Measured results, 68 HS300 constituents with full 2020-2024 history (1211 days / 596
latent vectors), 5000 epochs on CPU — reproduce with `prod_config.yaml`:

| Metric | Real | Generated |
|--------|------|-----------|
| Cross-sectional std | 0.0230 | 0.0273 |
| Mean return | 0.00088 | 0.00007 |
| Excess kurtosis | 3.33 | 9.22 |
| Tail dependence (5%) | 0.224 | 0.163 |
| Skewness | 0.682 | 0.085 |
| Correlation distance (Frobenius) | — | 5.72 |

**These numbers do not mean the model is good.** See `baselines.py`: benchmarked against
the empirical estimator on held-out data (the protocol of Chen et al. 2026, Sec. 6), the
diffusion model is *worse* than doing nothing, and worse than a plain multivariate
Gaussian fitted to the same latent. Results in `prod_output/baselines.json`.

Known gaps: cross-sectional skewness is under-reproduced, kurtosis now overshoots, and
volatility clustering is far below the real level (latents are sampled i.i.d.; the only
temporal structure comes from the sequential GARCH residual paths).

Training time: ~4 minutes on CPU (5000 epochs, single-threaded).

### Result provenance

`prod_output/results.json` is the latest production summary (68 stocks, 2020–2024).
`prod_log.txt` records an earlier 94-stock run with unstable generated returns and should
be treated as historical diagnostic output. The later benchmark files in `prod_output/`
are the authoritative model-comparison results; files whose names contain `stale` or
`undertrained` are retained only for comparison.

### Phase 1: factor/volatility integration

Phase 1 corrects the earlier interpretation of PCA residual-return components as
"volatility factors." The current pipeline instead:

- fits PCA to causal rolling log residual variance;
- fits loadings, PCA, and latent scaling only on the training window;
- treats volatility multiplicatively by scaling standardized residual innovations;
- includes a stable VAR(1) + GJR-GARCH dynamic factor baseline.

Matched US temporal benchmark: 87 assets, 1,300 training days, final 500 days held out,
2,000 generated paths, and 10,000 diffusion gradient steps:

| Method | Legacy RE4 | Phase 1 RE4 | Realized min-var vol | Vol-ACF MAE |
|---|---:|---:|---:|---:|
| Empirical | 1.643 | 1.643 | 0.00762 | 0.175 |
| Ledoit-Wolf | — | **1.615** | **0.00753** | — |
| Bootstrap | 2.060 | 1.629 | 0.01176 | 0.009 |
| Gaussian latent | 1.971 | 1.876 | 0.01176 | 0.012 |
| VAR-GARCH | — | 1.848 | 0.01196 | **0.009** |
| Diffusion | 4.841 | 4.595 | 0.01257 | 0.017 |

The redesign improves bootstrap RE4 by roughly 21% and gives VAR-GARCH substantially
better path diagnostics, but diffusion still does not earn its complexity on covariance
or portfolio risk. This is a negative but useful result: the next diffusion iteration
should generate conditional state paths rather than independent latent observations.

Reproduce the comparison from `diffusion_factor_model/`:

```bash
../.venv/bin/python phase1_benchmark.py
```

The machine-readable output, including the matched legacy reference, is in
`prod_output/phase1_benchmark.json`.

### Phase 2A: conditional-path forecasting protocol

Phase 2A establishes the protocol that a conditional diffusion model must beat:

- 60 observed state days condition each forecast;
- the target is the following 20-day state and return path;
- train, validation, and test windows are chronologically disjoint;
- all factor mappings and standardization are fitted only through June 2020;
- 25 non-overlapping rolling test origins cover December 2022 through December 2024;
- each method produces 20 ensemble paths per origin.

| Method | State energy ↓ | State RMSE ↓ | Log-vol RMSE ↓ | Daily-vol MAE ↓ | Tail error ↓ | Drawdown error ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Block bootstrap | 0.662 | 0.895 | 0.873 | 0.00316 | 0.00457 | **0.0158** |
| Gaussian VAR | **0.453** | **0.624** | **0.464** | **0.00310** | 0.00513 | 0.0173 |
| VAR-GARCH | **0.453** | **0.624** | **0.464** | 0.00325 | **0.00436** | 0.0172 |

No baseline dominates every metric. Gaussian VAR is the state-forecasting bar, block
bootstrap better preserves local path changes and drawdowns, and GARCH improves tail
calibration. Phase 2B conditional diffusion should be judged against the best baseline
for each metric, not against a single aggregate winner.

Reproduce from `diffusion_factor_model/`:

```bash
../.venv/bin/python phase2a_benchmark.py
```

The complete output is in `prod_output/phase2a_benchmark.json`.

### Phase 2B: VAR-anchored conditional diffusion

The first temporal model denoised complete future state paths directly. It approached VAR
but passed none of the best-baseline metric gates. The final Phase 2B model therefore uses
a tighter hybrid:

\[
z_{t+1:t+20}
= \widehat z^{VAR}_{t+1:t+20}
+ \text{DiffusionResidual}_{\theta}(z_{t-59:t})
\]

The VAR component supplies the strong linear conditional mean. Diffusion generates only
the normalized nonlinear residual path. Model selection uses 25 validation windows from
2020–2022; the 25 test origins from 2022–2024 are never used for checkpoint selection.
Return comparisons use the same GARCH innovation paths as the VAR-GARCH baseline.

| Metric ↓ | Best baseline | VAR-residual diffusion | Ratio | Gate |
|---|---:|---:|---:|:---:|
| State RMSE | 0.624 | 0.631 | 1.011 | — |
| Mean-factor RMSE | 0.745 | **0.738** | 0.991 | ✓ |
| Log-volatility RMSE | **0.464** | 0.492 | 1.060 | — |
| State energy score | **0.453** | 0.462 | 1.021 | — |
| State variogram | **0.0545** | 0.0579 | 1.063 | — |
| Return RMSE, scaled | **1.026** | 1.038 | 1.012 | — |
| Daily-volatility MAE | **0.00310** | 0.00314 | 1.011 | — |
| Tail-quantile error | 0.00442 | **0.00420** | 0.950 | ✓ |
| Drawdown error | 0.01575 | **0.01520** | 0.965 | ✓ |

The hybrid passes 3 of 9 gates. It is close on overall state, energy, return, and daily
volatility metrics, but its log-volatility path remains about 6% worse than Gaussian VAR.
This supports continued targeted work on volatility conditioning, but not a claim that
diffusion is generally superior.

Reproduce from `diffusion_factor_model/`:

```bash
../.venv/bin/python phase2b_benchmark.py
```

Artifacts:

- `prod_output/phase2b_benchmark.json`
- `prod_output/conditional_path_diffusion.pt`

### Phase 2C: volatility-aware split heads and seed robustness

Phase 2C retains the Phase 2B VAR-residual formulation but separates the final
mean-factor and log-volatility projections. Its denoising objective gives volatility
twice the weight of mean factors. Three independently initialized runs use seeds 42,
314, and 2718; their 75 origin-level scores are aggregated without selecting a seed
from test performance.

| Metric ↓ | Best baseline | Phase 2B | Phase 2C, 3-seed mean | Baseline gate |
|---|---:|---:|---:|:---:|
| State RMSE | **0.624** | 0.631 | 0.631 | — |
| Mean-factor RMSE | 0.745 | 0.738 | **0.736** | ✓ |
| Log-volatility RMSE | **0.464** | 0.492 | 0.495 | — |
| State energy score | **0.453** | 0.462 | 0.461 | — |
| State variogram | **0.0545** | 0.0579 | 0.0624 | — |
| Return RMSE, scaled | **1.026** | 1.038 | 1.036 | — |
| Daily-volatility MAE | **0.00310** | 0.00314 | 0.00316 | — |
| Tail-quantile error | **0.00446** | 0.00420 | 0.00557 | — |
| Drawdown error | **0.01575** | 0.01520 | 0.01834 | — |

The multi-seed result passes 1 of 9 best-baseline gates. Mean-factor RMSE improves
slightly, but the targeted log-volatility metric does not: its seed-level RMSE ranges
from 0.466 to 0.539. The seed with the lowest validation denoising loss also has the
worst test state and log-volatility RMSE. Therefore the split-head implementation works,
but weighted noise-prediction loss is not a sufficient model-selection objective for
forecast quality. Further work should first introduce validation-only path metrics or
calibration, rather than add model capacity.

Reproduce from `diffusion_factor_model/`:

```bash
../.venv/bin/python phase2c_benchmark.py
```

Artifacts:

- `prod_output/phase2c_benchmark.json`
- `prod_output/phase2c_diffusion_seed42.pt`
- `prod_output/phase2c_diffusion_seed314.pt`
- `prod_output/phase2c_diffusion_seed2718.pt`

### Phase 2D: validation path-energy checkpoint selection

Phase 2D changes no model layers, training data, or test protocol. Instead of selecting
checkpoints by fixed-noise denoising loss, it generates eight residual paths for every
validation context with a fixed seed and selects the lowest validation energy score.
Ten DDIM steps keep this selection calculation batched and inexpensive. Selected steps
are 1,750, 1,250, and 750 for seeds 42, 314, and 2718 respectively, compared with
2,250, 2,000, and 1,500 under Phase 2C's denoising-loss selector.

| Metric ↓ | Best baseline | Phase 2C | Phase 2D | Baseline gate |
|---|---:|---:|---:|:---:|
| State RMSE | 0.624 | 0.631 | **0.605** | ✓ |
| Mean-factor RMSE | 0.745 | 0.736 | **0.714** | ✓ |
| Log-volatility RMSE | 0.464 | 0.495 | **0.464** | ✓ |
| State energy score | 0.453 | 0.461 | **0.443** | ✓ |
| State variogram | **0.0545** | 0.0624 | 0.0587 | — |
| Return RMSE, scaled | **1.026** | 1.036 | 1.039 | — |
| Daily-volatility MAE | **0.00310** | 0.00316 | 0.00318 | — |
| Tail-quantile error | **0.00446** | 0.00557 | 0.00555 | — |
| Drawdown error | **0.01575** | 0.01834 | 0.02016 | — |

Phase 2D passes 4 of 9 gates and is the first diffusion variant to beat Gaussian VAR
on aggregate state RMSE and energy score. Relative to Phase 2C, state RMSE and energy
improve about 4%, log-volatility RMSE improves 6.3%, and state-RMSE dispersion across
seeds falls from 0.0229 to 0.0061. The remaining weakness is downstream return
calibration: improved latent forecasts do not yet produce better volatility, tails, or
drawdowns after reconstruction.

GARCH simulations now use explicitly seeded fitted Student-t/normal generators. Two
complete reevaluations produced exactly identical metric, gate, and seed-robustness
objects. Existing selected checkpoints can be reevaluated without retraining:

```bash
../.venv/bin/python phase2d_benchmark.py --reuse-checkpoints
```

Artifacts:

- `prod_output/phase2d_benchmark.json`
- `prod_output/phase2d_diffusion_seed42.pt`
- `prod_output/phase2d_diffusion_seed314.pt`
- `prod_output/phase2d_diffusion_seed2718.pt`

### Research market selection: S&P 500 primary

The retrospective paper workflow locks December 2023 constituent snapshots before
computational scoring of post-2023 observations. Universe files are checksum-pinned and accompanied
by provenance records in `research_data/universes/`. Eligibility and winsorization
are fit only on recorded pre-evaluation dates.

CSI 300 and S&P 500 were compared on a matched first-100 constituent screen
after applying the recorded coverage rules:
training ends in 2020, checkpoint validation uses 2021, and market selection uses
2022–2023. Each market uses the same architecture, 4,000-step budget, three seeds,
20 paths, and a recorded selection score: the geometric mean of the
diffusion/VAR-GARCH ratios for state energy and state RMSE.

| Development market | Eligible assets | Energy ratio | State-RMSE ratio | Selection score ↓ |
|---|---:|---:|---:|---:|
| CSI 300 | 64 | 1.1101 | 1.0952 | 1.1026 |
| **S&P 500** | 100 | **1.0196** | **1.0177** | **1.0187** |

S&P 500 is therefore the primary research market. This is a relative choice, not
evidence that diffusion is already superior: both scores exceed 1.0. Paired
origin bootstraps show that CSI underperformance is clear, while S&P differences
from VAR-GARCH remain statistically unresolved.

A validation-only residual-amplitude calibration was also tested. It improved
S&P state RMSE and return RMSE but worsened the combined score from 1.0187 to
1.0217, so it is rejected for the primary specification and retained as an
ablation.

After these decisions and the Phase 2F pool were locked, the 2024–current
holdout sample was loaded once. This lock occurred on 2026-07-30, after the
calendar period had occurred, and was not prospectively preregistered. The immutable
result is reported in the next section and must not be reused for model selection
or tuning.

Artifacts:

- `research_output/market_selection_decision.json`
- `research_output/csi300/market_selection_top100.json`
- `research_output/sp500/market_selection_top100.json`
- `research_sp500_confirmation.yaml`

### Phase 2F: retrospective locked-holdout result

The selected-market model was finalized as a validation-selected finite pool of
Phase 2D residual-diffusion paths and Gaussian VAR paths. The final models train
through 2022; 2023 selects checkpoints and pool weights. The frozen weights for
seeds 42, 314, and 2718 are 0.25, 0.25, and 0.50.

The one-time computational holdout covers 2024-01-02 through 2026-07-29: 645 trading days,
100 stocks, and 29 non-overlapping 20-day origins.

| Holdout state metric ↓ | Phase 2F | Gaussian VAR | Ratio | Paired 95% CI |
|---|---:|---:|---:|---:|
| State energy | **0.45257** | 0.46103 | **0.98165** | [-0.01529, -0.00181] |
| State RMSE | **0.61938** | 0.63295 | **0.97856** | [-0.02320, -0.00411] |

The locked composite ratio is **0.98010**, satisfying the recorded
holdout rule. The result persists at 5-, 10-, and 20-day prefixes. The original
artifact calls the state comparator VAR-GARCH, but its state paths are identical
to Gaussian VAR; GARCH affects only reconstructed stock returns.

The stronger Student-t VAR is essentially tied with Phase 2F: it has slightly
better energy (0.45202 versus 0.45257), while Phase 2F has slightly better
20-day state RMSE (0.61938 versus 0.62031). Phase 2F does not improve holdout
return-tail or drawdown metrics. The supported claim is improved latent
factor-state forecasting relative to Gaussian VAR/VAR-GARCH, not generally
superior return generation.

Post-hoc audits keep the models and validation-selected weights fixed. Nested
20/50/100-path scoring yields pool/Gaussian energy ratios of 0.9733, 0.9781, and
0.9776 and RMSE ratios of 0.9712, 0.9764, and 0.9751. Across five independent
20-path repetitions, every ratio favors the pool. At 100 paths, expanded
comparators include diagonal AR, training-selected ridge VAR, fixed-complexity
gradient-boosted AR, and persistence. The pool is dependence-robustly better than
those methods and Gaussian VAR. Its point estimates also beat Student-t VAR, but
that comparison remains inconclusive under block-bootstrap inference.

An equal-weight portfolio audit adds path-energy, volatility, 5% VaR pinball and
coverage, and covariance errors. It finds no consistent economic-risk advantage;
the pool's VaR pinball loss is reliably worse than Gaussian VAR. These consumed-
holdout analyses are sensitivity checks, not new confirmation evidence.

Artifacts:

- `Research_Paper_Draft.md`
- `research_output/sp500_frozen/frozen_protocol.json`
- `research_output/sp500_confirmation/confirmation.json`
- `research_output/sp500_confirmation/posthoc_monte_carlo_sensitivity.json`
- `research_output/sp500_confirmation/posthoc_expanded_baselines.json`
- `research_output/sp500_confirmation/posthoc_observable_risk_audit.json`

### Phase 3: post-primary-score development

Phase 3 experiments use only the pre-2024 development panel. They do not reopen,
overwrite, or tune against the locked S&P holdout result.

Phase 3A tested whether replacing independent GARCH innovations with joint
Student-t or moving-block innovations improved stock-return reconstruction.
The one-day joint block candidate was selected on 2021, but its untouched
2022–2023 five-metric return composite was 1.00510 relative to the incumbent.
It failed the development gate and was not applied.

The Phase 3A oracle diagnostic supplies the true future factor states while
retaining the existing GARCH innovation layer. Its return composite is 0.49650
relative to Phase 2F, indicating that state forecasting—not cross-stock
innovation dependence—is currently the larger bottleneck.

Phase 3B therefore replaced the Gaussian VAR portion of the finite pool with
either Student-t VAR or empirical-innovation VAR, with the classical component
and diffusion weights selected on 2021. Validation selected Student-t VAR and
a diffusion weight of 0.25 for all three seeds. On 2022–2023, all tracked point
estimates improved:

| Development result ↓ | Ratio to Phase 2F |
|---|---:|
| State energy | 0.99216 |
| State RMSE | 0.99242 |
| Return-distribution composite | 0.97853 |
| Tail-quantile error | 0.96311 |
| Drawdown error | 0.93900 |

The paired state-energy result was not statistically supported
(`p=0.1781`, 22 non-overlapping origins), so the recorded Phase 3B gate
rejected promotion. Phase 2F remains the primary model. The Student-t-base
pool was retained as a fixed candidate for a new cross-market holdout.

That one-time cross-market holdout has now been completed on CSI 300 data from
2024-01-02 through 2026-07-29. The refit uses 64 frozen securities, training
through 2022, 2023 checkpoint selection, and 28 non-overlapping 20-day origins.
The S&P-selected weights were not reselected for CSI.

| CSI external result ↓ | Phase 3B / Phase 2F ratio |
|---|---:|
| State energy | 0.99296 |
| State RMSE | 0.99405 |
| Return-distribution composite | 1.00702 |

Both state point estimates improved, and state energy improved at 67.9% of
origins. The paired state-energy interval nevertheless crossed zero
(`p=0.1952`), so the frozen external promotion rule failed. The CSI sample is
now consumed and cannot be used for further tuning.

Relative to pure Student-t VAR, the candidate had a state composite ratio of
0.99490 and a return composite ratio of 0.95030. Tail, drawdown, daily
volatility, return-variogram, and log-volatility improvements were individually
supported in paired exploratory comparisons. These secondary results are
encouraging but do not supersede the failed primary promotion rule.

Reproduce the development experiments without post-2023 holdout data:

```bash
PYTHONPATH=diffusion_factor_model .venv/bin/python \
  diffusion_factor_model/research/phase3a_reconstruction.py

PYTHONPATH=diffusion_factor_model .venv/bin/python \
  diffusion_factor_model/research/phase3b_state_pool.py
```

Artifacts:

- `research_output/sp500/phase3a_reconstruction.json`
- `research_output/sp500/phase3b_state_pool.json`
- `research_output/csi300_external/frozen_protocol.json`
- `research_output/csi300_external/replication.json`

### Phase 4A: regime-conditioned Student-t VAR

The first Phase 4 ablation tested whether the state bottleneck could be reduced
without retraining diffusion. A common stable VAR transition was retained, while
Student-t innovation covariance and tail thickness were estimated separately in
three training-only volatility regimes. The regime signal is the
cross-sectional average log variance reconstructed from current volatility
factors.

The fixed pre-2024 integration gate rejected the model:

| Development market | State composite vs Student-t VAR ↓ | Return composite ↓ | Energy p-value |
|---|---:|---:|---:|
| S&P 500 | 0.99103 | 1.03571 | 0.2023 |
| CSI 300 | 1.01654 | 1.04022 | 0.9367 |
| Cross-market pooled state | 1.00370 | — | — |

S&P state point estimates improved modestly, but return calibration worsened.
CSI state and return results worsened, including within every evaluated origin
regime. Regime conditioning is therefore retained as a negative ablation and is
not integrated into diffusion.

Artifact:

- `research_output/phase4a_regime_var.json`

### Phase 4B: shared cross-market residual diffusion

Phase 4B shares one residual-path denoiser across S&P 500 and CSI 300 while
retaining market-specific factor transforms, residual normalization, Student-t
VAR dynamics, and return reconstruction. A constant one-hot market token is
appended to every context step. The diffusion pool weight remains fixed at 0.25.

The pre-2024 development gate rejected a third-market freeze:

| Comparison with separate-market diffusion ↓ | S&P 500 | CSI 300 |
|---|---:|---:|
| State energy ratio | 1.00773 | **0.98446** |
| State RMSE ratio | 1.00667 | **0.98481** |
| Return composite ratio | 1.00682 | 1.00039 |
| Paired energy p-value | 0.9087 | **0.0002** |

The pooled state composite is 0.99585, a 0.41% improvement, below the required
1% materiality threshold. More importantly, state energy does not improve in
both markets. Shared learning significantly helps CSI but mildly harms S&P,
indicating negative transfer. The model is not frozen for a third market.

The next viable architectural experiment would require a shared trunk with
market-specific adapters or experts, rather than a completely shared denoiser.
That is a new development hypothesis and must retain the same unopened-market
boundary.

Artifacts:

- `research_output/phase4b_shared_diffusion.json`
- `research_output/phase4b_checkpoints/`

### Phase 4C: shared trunk with market-specific adapters

Phase 4C retains the Phase 4B shared residual denoiser trunk but adds two
explicit rank-8 residual adapters: one for S&P 500 and one for CSI 300. Adapter
output layers are initialized at zero, so training starts from the fully shared
architecture. The diffusion pool weight, three seeds, 4,000-step budget, paired
evaluation rows, and pre-2024 data boundary are unchanged.

The fixed gate again rejected a third-market freeze:

| Comparison ↓ | S&P 500 | CSI 300 | Pooled |
|---|---:|---:|---:|
| State composite vs separate-market diffusion | 1.00624 | **0.99068** | 0.99843 |
| State composite vs fully shared Phase 4B | **0.99905** | 1.00614 | 1.00259 |
| Return composite vs separate-market diffusion | 1.01209 | **0.99766** | — |

Adapters recover only 0.10% of S&P state performance relative to full sharing
and give back 0.61% on CSI, making pooled state performance 0.26% worse than
Phase 4B. Relative to separate-market diffusion, CSI still benefits while S&P
still experiences negative transfer. The pooled improvement is only 0.16%,
below the required 1%, and state energy does not improve in both markets.
Phase 4C is retained as a negative architectural ablation and is not used on an
unopened third market.

Artifacts:

- `research_output/phase4c_adapter_diffusion.json`
- `research_output/phase4c_checkpoints/`

## Notes & Considerations

### Data Requirements
- **AKShare**: Requires internet connection, may be slow for large universes
- **Caching**: Data is cached locally to avoid repeated downloads
- **Chinese Characters**: Factor names use Chinese (市场, 规模, 价值, etc.)

### Model Limitations
- Latent-state improvements do not yet transfer to return-tail calibration
- Path-energy checkpoint selection still uses a small 25-window validation set
- Factor loadings are static within each training window
- Regime-switching and macro conditioning are not yet implemented

### Extending the Model

**Add new factors:**
```python
# In factors/extractor.py
def extract_custom_factor(self, returns):
    # Your factor logic here
    return pd.Series(...)
```

**Custom score network:**
```python
# In diffusion/score_net.py
class CustomScoreNet(nn.Module):
    def __init__(self, config, latent_dim):
        # Your architecture
        pass
```

## References

- Chen, M., et al. (2026). "Diffusion Factor Models: Generating High-Dimensional Returns with Factor Structure"
- Song, Y., et al. (2021). "Score-Based Generative Modeling through Stochastic Differential Equations"
- Lu, C., et al. (2022). "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling"

## License

MIT License - See LICENSE file for details.
