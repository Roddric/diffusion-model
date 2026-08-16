# Prospective multi-market strong-baseline study — design draft

Status: **pre-outcome design draft; not yet frozen or preregistered**.

No post-2023 price observation from any candidate market may be downloaded until
the evaluation runner, tests, protocol hashes, trained checkpoints, and this
document are committed and publicly timestamped. Historical constituent revisions
and prices ending 2023 are permitted for deterministic eligibility, training, and
validation.

## Research question

Does the unchanged Student-t-VAR-base residual-diffusion pool improve probabilistic
factor-state forecasts over the better of Gaussian VAR and Student-t VAR across a
geographically diverse population of untouched equity markets, and does any gain
extend to realized covariance forecasting?

## Candidate markets and eligibility

The registry fixes DAX, ASX 200, CAC 40, IBEX 35, NIFTY 50, SMI, and TSX 60 before
price screening. DAX uses the archived December 2023 index-constituents file. The
other panels use the last English Wikipedia revision strictly before
2023-12-01T00:00:00Z, with revision ID and source hash recorded. Nikkei 225 was
considered but excluded before price access because its historical page revision
contained no constituent table.

A market remains in the primary study if, using data ending 2023 only:

1. the prespecified benchmark ticker resolves;
2. at least 15 securities pass the unchanged 95% coverage screen through
   2022-12-30;
3. at least one 2023 validation origin exists; and
4. the pre-2024 panel can be fingerprinted and reproduced.

The leading 100 eligible securities in immutable manifest order are retained.
These rules may exclude a market but may not substitute another after screening.

### Pre-2024 feasibility outcome

The eligibility screen was executed on 2026-08-16 using panels ending
2023-12-29; its machine-readable output is
`research_output/prospective_multimarket/pre2024_smoke.json`. DAX (35 assets),
ASX 200 (100), CAC 40 (38), NIFTY 50 (46), SMI (18), and TSX 60 (57) qualify.
IBEX 35 is excluded because 13 assets, fewer than the prespecified minimum of
15, pass the coverage rule. No replacement market is permitted. Thus `K = 6`
and the individual-market requirement below is fixed at five passes.

## Frozen model candidate

Architecture, factor construction, three seeds, 8,000-step budget, checkpoint
selection, path counts, and weight grid remain identical to Phase 2F. For every
market, 2023 validation selects the Student-t-VAR/diffusion pool weight separately
for each seed from `{0, 0.25, 0.50, 0.75, 1}`. No market-specific hyperparameter
or feature tuning is allowed. The Gaussian-base pool is retained as a labeled
secondary mechanism comparison but cannot satisfy the primary claim.

## Primary state estimand

For each market and each classical baseline, compute the geometric mean of the
pool/baseline state-energy and state-RMSE ratios. The market's strong-baseline
ratio is the worse (larger) of the ratios against Gaussian VAR and Student-t VAR.
Thus, a market passes only if the Student-t-base pool has composite below one
against both baselines and none of its four co-primary component ratios exceeds
one.

The population estimand is the equal-market mean log strong-baseline ratio. The
primary interval is a two-sided 95% Student-t interval across markets. The state
claim passes only if its upper endpoint is below zero on the log scale and at
least five of the six eligible markets pass individually. This count was fixed
as `ceil(0.70 * K)` after the pre-2024 screen produced `K = 6`. Had fewer than
five markets qualified, the study would have been underpowered and no population
claim would have been made.

## Economic-value endpoint

The key secondary estimand replaces the state composite with the scale-normalized
Frobenius error of the forecast covariance matrix relative to realized 20-day
covariance. For each origin and method, the forecast is the arithmetic mean of
the sample covariance matrices computed separately within each simulated 20-day
path; paths are not flattened together. The denominator is the Frobenius norm of
the realized 20-day covariance, floored at `1e-12`. The market ratio again uses
the worse comparison against Gaussian and Student-t VAR. It receives the same
two-part population rule—upper 95% interval endpoint below zero and at least
five of six individual market improvements—but is reported as a separate key
secondary claim. VaR pinball, portfolio-path energy, volatility error, coverage
error, tails, and drawdowns are supporting endpoints and cannot replace a failed
covariance result.

## Prespecified moderators

1. **Horizon:** repeat the state composite at cumulative 5-, 10-, and 20-day
   prefixes. The moderator statistic is each market's log-ratio change from 5 to
   20 days, summarized with a market-level Student-t interval.
2. **Volatility regime:** freeze each market's high/low threshold as the median
   trailing-20-day benchmark volatility across 2023 validation contexts. Evaluation
   origins are classified from context data only. Report high-minus-low log-ratio
   differences across markets, subject to at least five origins per regime.
3. **Eligible cross-section size:** exploratory meta-regression only; no selection
   or confirmatory claim.

## Multiplicity and claim boundary

The state effect is the sole primary claim. Covariance is the key secondary claim
and will be interpreted only at two-sided 95% confidence without using it to rescue
a failed state claim. Moderator and other economic endpoints are secondary and
clearly labeled. All markets and failures remain in the report. No market,
endpoint, horizon, regime, seed, or baseline can be removed after outcome access.

## Required lock sequence

1. Materialize and hash universes.
2. Run the pre-2024-only eligibility screen.
3. Complete and test the generic freeze and one-time evaluation runners.
4. Train through 2022 and select checkpoints/weights on 2023 only.
5. Record every manifest, config, panel, checkpoint, and code hash in one study
   protocol; commit, tag, and publicly push it.
6. Create and publicly push a notarization record containing the frozen protocol
   hash, public commit, and public URL. The one-time runner refuses to start
   without this record.
7. After the public timestamp, download post-2023 outcomes once, run all eligible
   markets, and preserve successes and operational failures atomically.
