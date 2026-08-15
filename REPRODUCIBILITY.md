# Reproducibility and research-status record

## Status of the evidence

This is a retrospective locked-holdout study. The December 2023 universe archive,
pre-2024 development data, model checkpoints, and scoring rule were locked on
2026-07-30 before post-2023 observations were loaded into the scoring pipeline.
The 2024–2026 calendar period had already occurred, and the protocol was not
externally preregistered. The saved timestamps and hashes establish execution
order inside this workspace; they do not establish a prospective research freeze.

The original immutable result is
`research_output/sp500_confirmation/confirmation.json`. The later
`posthoc_dependence_robustness.json` exactly reproduces its point estimates, stores
the origin-level losses, and adds dependence-robust inference. Later files prefixed
`posthoc_` add path-count, expanded-baseline, observable-risk, and
calibration/power checks. None redefines the original decision or creates a new
holdout sample.

A preregistered FTSE 100 untouched-market evaluation was run one time on
2026-08-12, after external notarization of tag `ftse100-preregistration-2026-08-11`
(commit `feff0986843ddd454b908e7084cb80da688b9918`). The protocol record is
`research_output/ftse100_frozen/ftse100_external.protocol.json`; the one-time
freeze and 2024–2026 confirmation are
`research_output/ftse100_frozen/frozen_protocol.json` and
`research_output/ftse100_confirmation/confirmation.json`. The FTSE sample is now
consumed for those two decisions and may only support post-hoc audits.

A preregistered HSI untouched-market evaluation was run one time on 2026-08-14
after public push of tag `hsi-preregistration-2026-08-12` (commit
`c6085db873be3d8d730e162c0231648c895b447a`). The frozen protocol and checkpoint
hashes were then published at commit `e694256` before outcome access. Both
prespecified decisions failed: the Gaussian-base pool composite was 1.016159
versus VAR-GARCH and the Student-t-base pool composite was 1.009953 versus
Student-t VAR. The immutable result is
`research_output/hsi_confirmation/confirmation.json`; the HSI sample is consumed
and may only support clearly labeled post-hoc audits.

## Audited environment

- Audit date: 2026-08-03
- Python: 3.11
- Exact Python environment: `requirements-lock.txt`
- Minimum supported dependencies: `requirements.txt`
- Random training seeds: 42, 314, and 2718
- The research runners force single-threaded PyTorch execution.

This directory did not have Git history when the audit began. Consequently, no
historical commit identifier is claimed. Git history now begins at an audited
baseline for future work; it cannot retroactively establish the July chronology.
Tag and disclose a commit before any new prospective evaluation.

## Verify immutable artifacts

From the project directory:

```bash
shasum -a 256 -c ARTIFACT_MANIFEST.sha256
```

The manifest covers the S&P, FTSE, and HSI universes and price caches, research
configurations, frozen checkpoints, original results, and post-hoc audit results.

## Install and test

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
```

The audited suite contains 116 tests.

## Reproduce the post-hoc audits

Use a new output path; the runner refuses to overwrite existing evidence:

```bash
MPLCONFIGDIR=/tmp/mplconfig \
PYTHONPATH=diffusion_factor_model \
.venv/bin/python diffusion_factor_model/research/confirm_phase2f.py \
  --output /tmp/posthoc_dependence_robustness.json \
  --report-role posthoc_robustness_audit
```

The rerun must report exact matches for the locked composite and co-primary ratios.
It also reports circular block-bootstrap intervals at block lengths 2–5 and
Newey–West inference at the automatic lag (three for 29 origins).

The following commands use new output paths because every runner refuses to
overwrite existing evidence:

```bash
MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=diffusion_factor_model .venv/bin/python \
  diffusion_factor_model/research/monte_carlo_sensitivity.py \
  --path-counts 20 50 100 --replicates 1 \
  --output /tmp/posthoc_monte_carlo_sensitivity.json

MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=diffusion_factor_model .venv/bin/python \
  diffusion_factor_model/research/expanded_baseline_audit.py \
  --paths 100 --output /tmp/posthoc_expanded_baselines.json

MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=diffusion_factor_model .venv/bin/python \
  diffusion_factor_model/research/confirm_phase2f.py \
  --output /tmp/posthoc_observable_risk_audit.json \
  --report-role posthoc_observable_risk_audit
```

The calibration and overlapping-origin power audit reproduces the locked
composite exactly, then reports rank-histogram calibration and an energy-score
decomposition on the locked origins and rescores the locked model at stride-5
overlapping origins with HAC and moving-block inference:

```bash
MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=diffusion_factor_model .venv/bin/python \
  diffusion_factor_model/research/posthoc_calibration_power_audit.py \
  --output /tmp/posthoc_calibration_power_audit.json --allow-overwrite
```

Its overlapping-origin comparison is post-hoc sensitivity evidence; it does not
create a new confirmation sample.

The matching FTSE audit reproduces both locked FTSE composites and reports the
same diagnostics for the dual pools:

```bash
MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=diffusion_factor_model .venv/bin/python \
  diffusion_factor_model/research/posthoc_ftse_calibration_power_audit.py \
  --output /tmp/posthoc_ftse_calibration_power_audit.json --allow-overwrite
```

The expanded nonlinear baseline is deliberately compute-intensive. For a future
unseen evaluation, create the protocol before outcomes are available:

```bash
PYTHONPATH=diffusion_factor_model .venv/bin/python \
  diffusion_factor_model/research/register_experiment.py \
  --experiment-id prospective-study \
  --hypothesis "The frozen candidate improves both co-primary losses" \
  --data-boundary "Only observations after the disclosed lock date" \
  --decision-rule "Both predeclared loss ratios must be below one" \
  --output prospective-study.protocol.json \
  --config research_sp500_confirmation.yaml
```

## Interpretation boundary

The defensible claim is an approximately 2% retrospective holdout improvement in
latent-state forecasts relative to Gaussian VAR. The hybrid is not conclusively
better than Student-t VAR and does not show a consistent observable portfolio-risk,
return-tail, or drawdown improvement. Neither the consumed S&P nor CSI 2024–2026
sample may be used for additional selection. A stronger confirmatory claim requires
a new market that has not influenced development or a future prospectively locked
evaluation period.

The stride-5 overlapping-origin audit reports that the pool separates from both
Gaussian VAR and Student-t VAR under HAC and moving-block inference. This is
post-hoc sensitivity evidence only: overlapping windows share target days, the
analysis was specified after the locked score was known, and it does not upgrade
the confirmatory status of the primary result. The energy-score decomposition
indicates the pool's Gaussian-relative gain comes from the ensemble-spread term
with the distance-to-observation term essentially unchanged.

The FTSE 100 evaluation ran one time on 2026-08-12 under the externally notarized
preregistration. Its defensible claim mirrors the S&P one: an approximately 2%
retrospective holdout improvement in latent-state forecasts relative to Gaussian
VAR (composite 0.98204, HAC p=0.0017 on both co-primary metrics), with the
Student-t-base pool statistically tied with Student-t VAR. No observable
portfolio-risk endpoint was confirmed on FTSE; the FTSE result addresses
latent-state generalization to an untouched market, not economic value. The
consumed FTSE 2024–2026 sample may only support post-hoc audits.

The HSI evaluation ran one time on 2026-08-14 under its publicly timestamped
preregistration and frozen protocol. It does not reproduce the S&P/FTSE pattern.
The Gaussian-base pool was effectively tied on energy but worsened state RMSE by
3.38%, giving composite 1.016159. The Student-t-base pool worsened both co-primary
losses, giving composite 1.009953. The HSI result therefore establishes genuine
cross-market heterogeneity and narrows any generalization claim: pooling helps in
S&P and FTSE under the recorded comparisons, but not reliably in HSI. The HSI
2024–2026 sample is consumed and may only support post-hoc audits.

The post-hoc cross-market synthesis uses only already locked score artifacts and
can be reproduced with:

```bash
MPLCONFIGDIR=/tmp/mplconfig .venv/bin/python \
  diffusion_factor_model/research/cross_market_meta_analysis.py \
  --allow-overwrite
```

It must exactly reproduce the S&P, FTSE, and HSI Gaussian-base composites before
continuing. It then resamples paired origins within each market, estimates an
equal-market mean log-composite with a Student-t interval across markets, reports
an external-only and DL/HKSJ sensitivity, and regenerates the forest and horizon
figures. Expected headline values are 0.992630 [0.943784, 1.044003] for all three
markets and 0.998953 [0.804095, 1.241032] for the two preregistered external
markets. CSI is excluded because its candidate does not share this estimand. This
analysis is explicitly post-hoc and does not load prices or rescore a holdout.

The reconstruction oracle-ceiling attribution is a pre-2024 diagnostic and can
be reproduced with:

```bash
MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=diffusion_factor_model .venv/bin/python \
  diffusion_factor_model/research/oracle_ceiling_attribution.py --overwrite
```

It evaluates all eight mean-state/volatility-state/innovation oracle coalitions
on the 22 S&P development origins and checks both previously archived Phase 3A
boundary results within `1e-12`. It reports paired-origin Shapley intervals and a
five-block cross-fitted loading-representation diagnostic. The script rejects any
configuration extending into 2024, and the artifact records that no confirmation
or consumed HSI observation was loaded.

The state-conditional innovation candidate uses the same split and can be
reproduced with:

```bash
MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=diffusion_factor_model .venv/bin/python \
  diffusion_factor_model/research/state_conditional_innovation_experiment.py \
  --overwrite
```

The script selects among `k=32,64,128,256` on 2021 and applies the unchanged
Phase 3A gate once on 2022–2023. It must reproduce selected `k=32`, composite
0.981565, and an accepted gate while recording that no post-2023 external sample
was loaded. Promotion still requires a new external confirmation protocol.

The publicly frozen fixed-candidate robustness audit is reproduced with:

```bash
MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=diffusion_factor_model .venv/bin/python \
  diffusion_factor_model/research/state_conditional_robustness_audit.py \
  --overwrite
```

The script first checks every frozen code, configuration, report, and candidate
hash from `state_conditional_robustness.protocol.json`. It must reproduce all five
improving 20-path composites, 50-path composite 0.971888 with energy ratio
1.000281, and 100-path composite 0.976445 with energy ratio 0.998635. The final
status is `robustness_failed`: the 50-path energy and low-volatility regime
requirements fail, so external promotion is prohibited. No post-2023 external
sample is loaded.

The pre-2024 mean-state diagnostic and frozen factorwise routing experiment are
reproduced with:

```bash
MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=diffusion_factor_model .venv/bin/python \
  diffusion_factor_model/research/mean_state_diagnostic.py --overwrite

MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=diffusion_factor_model .venv/bin/python \
  diffusion_factor_model/research/factorwise_mean_routing.py --overwrite
```

The diagnostic must identify market as the largest Phase2F mean-state MSE
contributor (share 0.30657). The frozen validation rule must route market and
illiquidity to Student-t VAR. The descriptive score must reproduce mean-factor
RMSE ratio 0.994558, state composite 0.996236, return composite 1.010342, and a
failed gate because paired state-energy p=0.1058 is not below 0.10. Neither script
loads post-2023 external observations.
