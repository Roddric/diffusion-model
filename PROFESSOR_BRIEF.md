# Professor brief: VAR–diffusion factor-state forecasting

## Bottom line

The project is ready to discuss as a technically complete **retrospective
locked-holdout study**. It should not be described as a prospective preregistered
confirmation. A validation-selected pool of Gaussian VAR and residual-path
diffusion improves S&P 500 latent-state energy by 1.83% and state RMSE by 2.14%
over Gaussian VAR on 29 non-overlapping 20-day origins. It is tied with a stronger
Student-t VAR in the immutable 20-path result. Post-hoc 100-path scoring gives the
pool better point estimates than all tested comparators, but remains inconclusive
against Student-t VAR under block-bootstrap inference. It does not establish a
portfolio-risk or return-tail improvement. A preregistered FTSE evaluation
reproduces the Gaussian-relative latent-state gain, but a subsequent preregistered
HSI evaluation fails both the Gaussian-base and Student-t-base rules. The evidence
therefore shows cross-market heterogeneity rather than universal transfer.

## External-market evidence

| Market and evidence class | Gaussian-base result | Student-t-base result |
|---|---|---|
| FTSE 100, preregistered analysis | PASS, composite 0.98204 | Rule-level PASS, composite 0.99631; statistically tied |
| HSI, preregistered analysis | **FAIL**, composite 1.01616 | **FAIL**, composite 1.00995 |

On HSI, Gaussian-relative energy is essentially tied (ratio 0.99886), but state
RMSE worsens 3.38%, driven by log-volatility-state error. Against Student-t VAR,
both co-primary losses worsen. These results were preserved without rerunning or
retuning the consumed sample.

## Workflow

1. Use an archived December 2023 S&P 500 membership snapshot and retain the first
   100 securities passing a training-era coverage screen.
2. Fit five return-summary states and five PCA log-volatility states using training
   data through 2022.
3. Fit a stable Gaussian VAR(1) and a conditional diffusion model for residual
   20-day state paths conditioned on 60 observed days.
4. Select checkpoints and discrete VAR/diffusion pool weights on 2023 only.
5. Lock hashes, checkpoints, and the decision rule, then score 2024–2026 once in
   the computational pipeline.
6. Reproduce the result in a separately labeled post-hoc audit and test sensitivity
   to serial dependence across origins.

## Primary evidence

| State loss | Pool | Gaussian VAR | Ratio | Original paired 95% interval |
|---|---:|---:|---:|---:|
| Energy | 0.45257 | 0.46103 | 0.98165 | [-0.01529, -0.00181] |
| RMSE | 0.61938 | 0.63295 | 0.97856 | [-0.02320, -0.00411] |

Circular block-bootstrap intervals remain below zero for block lengths 2–5.
Newey–West one-sided normal approximations are 0.00091 for energy and 0.00016 for
RMSE. These are post-hoc robustness checks, not a new confirmation.

Against Student-t VAR, the energy difference is +0.00055 and the RMSE difference
is -0.00093. All block-bootstrap intervals cross zero; HAC one-sided values are
0.539 and 0.450. The hybrid therefore does not dominate the strongest baseline.

## Fixes completed after the audit

- **Monte Carlo stability:** frozen weights were rescored with nested 20, 50, and
  100-path ensembles. Pool/Gaussian ratios remain between 0.971 and 0.978. Five
  independent 20-path repetitions give mean energy and RMSE ratios of 0.9766 and
  0.9747; every repetition favors the pool. These are post-hoc sensitivity results.
- **Expanded baselines:** at 100 paths the pool has energy 0.43110 and RMSE
  0.60160. It is significantly better under HAC and block-bootstrap checks than
  Gaussian VAR, ridge VAR, diagonal AR, gradient-boosted AR, and persistence.
  Student-t VAR scores 0.43698 and 0.61005, but its comparison remains
  inconclusive (HAC one-sided values 0.066 and 0.064; all block intervals cross
  zero).
- **Observable risk metrics:** equal-weight portfolio path energy, volatility
  error, 5% VaR pinball and coverage errors, and covariance-matrix error are now
  reported. No consistent risk advantage is supported. In particular, VaR
  pinball loss is worse than Gaussian VAR (0.001160 versus 0.001094), with all
  block-bootstrap intervals for the loss difference above zero.
- **Research governance:** an experiment ledger, immutable prospective protocol
  registrar, exact dependency lock, artifact checksums, and tests for the new
  metrics and baselines have been added. Git history begins only at the audited
  baseline and cannot prove the earlier chronology.

## Latest pre-2024 reconstruction development

A factorial oracle diagnostic now attributes 51.6% of attainable five-metric
return-loss reduction to mean-state error, 43.2% to innovations, and 5.1% to
volatility-state error. Cross-fitted loading updates show smaller headroom (mean
mapping ratio 0.982; volatility-PCA ratio 0.937).

Following that result, a training-only nearest-neighbor sampler conditions full
cross-sectional innovation vectors on each forecast log-volatility state. The
2021-selected `k=32` candidate achieves a 2022–2023 return composite of 0.981565
versus Independent-GARCH and passes the fixed development gate. Return energy
improves at 77.3% of origins (paired one-sided p=0.0064), while the return
variogram is effectively tied. This is the strongest reconstruction candidate so
far, but it is exploratory and requires a new untouched confirmation sample.

That candidate subsequently failed its publicly frozen all-requirements
robustness gate. Although all five 20-path repetitions improve and the 100-path
composite is 0.97644, 50-path energy is marginally worse (1.00028). A second
failure comes from the training-defined regime check: only one origin is labeled
low volatility, and its composite is 1.08874. Covariance and VaR safety bounds
pass, but the recorded rule forbids external promotion. The candidate remains a
useful exploratory mechanism rather than a new headline result.

A subsequent mean-state diagnostic finds that market returns contribute 30.7% of
mean-state MSE but are primarily a calibration problem; Phase 2F already performs
well on momentum and reversal. A frozen factorwise route to Student-t VAR for
market and illiquidity improves mean-factor RMSE (ratio 0.99456, paired p=0.0536)
and both full-state losses. It nevertheless fails its descriptive gate because
state-energy p=0.1058 misses the 0.10 threshold. Return reconstruction also
worsens (composite 1.01034; tail ratio 1.04274). This route is not promoted.

## Disclosures that must accompany the result

- The archive and protocol were locked on 2026-07-30, after the 2024–2026 period
  had occurred. The pipeline had not loaded those observations during development,
  but there was no prospective external preregistration.
- December 2023 membership creates survivorship conditioning when applied to
  2015–2022 training history.
- The feature internally called illiquidity is log mean absolute return, not a
  volume-based liquidity measure.
- Gaussian VAR and the artifact named VAR-GARCH have identical state forecasts;
  GARCH changes only reconstructed returns.
- The S&P, CSI, FTSE, and HSI post-2023 samples are consumed and cannot support
  more tuning.
- The original score used only 20 simulated paths. Larger post-hoc ensembles are
  reassuring for the Gaussian comparison, but do not retroactively change the
  locked result.

## Acceptable conclusion

> Conservative pooling adds modest latent-state forecast information beyond
> Gaussian VAR in the retrospective S&P holdout and a preregistered FTSE analysis,
> but fails to transfer to a preregistered HSI analysis. The hybrid does not
> conclusively dominate Student-t VAR and shows no consistent observable-risk or
> return-tail improvement. The supported result is therefore market-dependent,
> and a stronger claim requires a new protocol targeting the binding baseline and
> economic reconstruction layer.

This wording is honest about the design and should be suitable for an academic
progress discussion. The limitations do not make the experiment useless; they
determine the strength of the claim.
