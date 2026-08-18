# Prospective multi-market strong-baseline study — design draft

Status: **pre-outcome design draft; not yet frozen or preregistered**.
Amended 2026-08-18 (pre-lock): endpoint-hierarchy rationale, claim trichotomy
and stop rule, operational resume policy, post-2023 ticker attrition rule,
Europe-block sensitivity, raw-download pinning, and factor-quality diagnostics
were prespecified below and implemented in the runners before any outcome
access. All amendments precede the lock and are therefore design choices, not
post-outcome changes.

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

On 2026-08-18, pre-lock factor-quality diagnostics revealed that the recorded
TSX 60 panel fingerprint did not reproduce from its pinned pre-2024 price
cache, while all six other markets reproduced exactly. No TSX 60 checkpoint,
pool weight, decision, or post-2023 observation existed, so the screen was
re-executed on 2026-08-18 from the unchanged caches under the unchanged rules.
Only `screened_at_utc` and the TSX 60 fingerprint changed; eligibility,
columns, exclusions, and `K = 6` are unchanged. The superseded 2026-08-16
artifact copy is preserved for audit. This incident motivates the
raw-download pinning policy below.

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

### Endpoint hierarchy rationale

The prior review recommended covariance forecasting as the main observable
endpoint. This design deliberately keeps the latent-state composite as the sole
primary claim and treats covariance as the key secondary. Three reasons are
recorded before outcome access:

1. **Mechanism locality.** The candidate modifies the state-path distribution;
   the state composite measures exactly that mechanism. Covariance adds the
   reconstruction layer, which prior audits attribute most of its error to
   mean-state and innovation components that the pool does not claim to fix.
2. **Prior observable-risk evidence.** The locked S&P observable-risk audit
   found no consistent risk advantage and worse 5% VaR pinball loss than
   Gaussian VAR. Promoting the endpoint where prior evidence is weakest would
   risk selecting the primary claim on expected outcome direction.
3. **Claim honesty.** If the state claim passes and covariance fails, the
   supported conclusion is a mechanism gain without established economic value,
   which the follow-up paper must state as such. If state fails, covariance
   cannot rescue it. This ordering is fixed here and cannot be reversed later.

The deviation from the recommended hierarchy is disclosed in the paper's
limitations rather than silently absorbed.

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

### Decision trichotomy and program stop rule

Each population claim (primary state, key secondary covariance) is classified
by a prespecified trichotomy implemented in the shared helper:

- **pass** — the 95% interval's upper log endpoint is below zero and at least
  five of six markets pass individually;
- **fail** — the interval's lower log endpoint is above zero, so the population
  effect is confidently unfavorable;
- **inconclusive** — anything else, including a favorable interval that fails
  the individual-market count rule.

Whatever the classification, the one-time evaluation is terminal for this
candidate. No third-generation market sample, extended window, or replacement
market may be opened for the Student-t-base pool after the one-time run. A
failed or inconclusive outcome is published as a bounded result, not as a
mandate for further holdout consumption.

## Prespecified one-time evaluation policies

These policies are implemented in the one-time runner and recorded verbatim in
the frozen protocol. They may drop data deterministically; they may never add,
re-select, or rescore.

1. **Operational failure and resume.** Markets are scored in frozen protocol
   order. Each market's terminal result is written atomically to its own
   artifact file. A transient pipeline failure (network crash, code error)
   writes no artifact and may be retried by rerunning the runner with the same
   pinned evaluation end. A scored market or a terminal-attrition market is
   never rescored. Every invocation is appended to an attempt log recording the
   evaluation end and the per-market disposition. The evaluation end is pinned
   by the first attempt; later attempts with a different end are refused.
2. **Post-2023 ticker attrition.** A frozen asset survives the one-time
   download only if it resolves and has non-missing returns on at least 80% of
   post-2023 benchmark sessions. Survivors keep frozen manifest order; dropped
   assets are recorded and never replaced. A market needs at least 15
   survivors (the same floor as the eligibility screen); otherwise it is
   preserved as a terminal data-attrition outcome. Population inference
   additionally requires every eligible market to be scored; a terminal
   attrition market makes the population claim not evaluable, while all
   individual-market results remain in the report.
3. **Raw download pinning.** Vendor-adjusted closes can be recomputed silently
   after the fact. Both the frozen pre-2024 and the one-time current raw price
   downloads are therefore copied into the immutable artifact directory with
   content hashes and pin timestamps before any scoring.
4. **Europe-block sensitivity.** DAX, CAC 40, and SMI are integrated
   Alpine/Eurozone-adjacent markets, so treating all six markets as independent
   overstates the effective replication count. A prespecified sensitivity
   analysis collapses the three block markets into one unit (equal-weight mean
   log ratio), giving four units with a Student-t interval, for both the
   primary state claim and the key secondary covariance claim. The block is
   fixed by registry country, never by observed outcomes, and cannot replace
   the primary population decision.
5. **Pre-lock factor-quality diagnostics.** Before the lock, each market's
   pre-2024 panel is summarized descriptively: cross-sectional mean-factor OLS
   R² and log-volatility PCA explained variance on the training window only
   (`research_output/prospective_multimarket/factor_quality/`). These numbers
   contextualize heterogeneity (for example, SMI's 18-security cross-section)
   and cannot change any frozen decision.

## Required lock sequence

0. Merge the development branch into the main branch so the locked protocol
   sits on mainline history, and commit the factor-quality diagnostics.
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
