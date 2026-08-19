# Final research route — reconstruction-layer investigation

Status: **design frozen before any route execution (2026-08-19). Pre-2024 data
only. This is the final research route of the diffusion factor-model program.**

## Program context

The prospective six-market evaluation (tags `multimarket-freeze-2026-08-18`,
`multimarket-evaluation-2026-08-19`) closed the Student-t-base pool candidate:
primary state claim inconclusive (4/6 individual passes, population ratio
0.99385 [0.98085, 1.00703]), key secondary covariance inconclusive (3/6,
0.99073 [0.95016, 1.03304]). The prespecified stop rule engaged. All locked or
preregistered holdout samples (S&P 500, CSI 300, FTSE 100, HSI, and the six
prospective markets' post-2023 windows) are consumed.

The S&P 500 pre-2024 oracle-ceiling attribution
(`research_output/sp500/oracle_ceiling_attribution.json`) showed reconstruction
loss is dominated by mean-state (51.6%, CI [47.6, 55.3]) and innovation (43.2%,
CI [39.7, 47.3]) components, with volatility state contributing 5.1%. That is a
single-market result on consumed S&P development data. This route determines
whether that error structure is universal across the six prospective markets
and whether a reconstruction-layer improvement is worth a final preregistered
outcome claim — or whether the program ends.

## Hard scope guards

1. Only observations through 2023-12-29 may be loaded by any script in this
   route. Runners raise on any later observation.
2. Post-2023 outcomes already recorded in
   `research_output/prospective_multimarket/evaluation.json` may be read for
   descriptive labeling only and may never influence any design choice, gate,
   or candidate in this route.
3. No holdout market or window may be consumed by this route. Any eventual
   outcome claim requires a fresh preregistration on genuinely untouched data
   (markets never screened or scored, or a future locked period), never the
   consumed samples.
4. The closed Student-t-base pool candidate stays closed. Nothing in this
   route revives it or its variants (state-conditional resampling, factorwise
   routing), which already failed frozen gates.

## Phase A — descriptive "when does it work" diagnostics (hypothesis-generating)

Scope: the six eligible markets. Features are computed on the training window
(observations through 2022-12-30) only:

- eligible asset count;
- mean-factor OLS R² (mean and minimum across assets) and volatility-PCA
  cumulative explained variance from the pre-lock factor-quality diagnostics;
- annualized benchmark volatility;
- mean pairwise correlation of eligible-asset daily returns;
- mean cross-sectional daily return dispersion (annualized).

Correlates: the validation-selected Student-t-base pool weight (mean across
seeds, from the frozen protocol) and, labeled explicitly as post-hoc
descriptive, the post-2023 state strong-baseline ratio. With K = 6 the output
is raw values plus Spearman rank correlations only. No formal inference, no
claim. Artifact: `research_output/prospective_multimarket/when_does_it_work/`.

## Phase B — cross-market oracle-ceiling attribution (prespecified)

For each of the six markets, reproduce the S&P attribution mechanics on pre-2024
data with the frozen candidate itself:

- checkpoints: the market's frozen checkpoints; weights: its validation-selected
  Student-t-base pool weights per seed;
- windows: train through 2022-12-30 (identical to the frozen representation),
  validation buffer 2023-01, development/attribution origins February–December
  2023, non-overlapping with stride equal to the horizon;
- coalitions, return metrics, Shapley allocation, and paired-origin bootstrap
  (10,000 samples, seed 20260815) identical to the S&P diagnostic;
- representation-drift diagnostic (fixed vs cross-fit mappings) per market;
- fingerprint and asset-column checks against the pre-2024 smoke record before
  anything runs.

Population summaries treat the market as the replication unit: market-level
Student-t 95% intervals over per-market component shares.

**Prespecified proceed gate.** The route proceeds to Phase C only if all of:

1. pooled mean-state + innovation share: point estimate ≥ 0.80;
2. its 95% interval lower endpoint (arithmetic share scale) > 0.60;
3. per-market mean-state + innovation share > 0.50 in at least five of six
   markets.

Otherwise the route ends here. A heterogeneous or weak error structure means
reconstruction is not a universal bottleneck, and no candidate work is
justified.

## Phase C — conditional candidate development (only if the gate passes)

Rules, frozen now:

1. Candidates modify the reconstruction layer only (state-to-return mapping,
   innovation structure). State dynamics and the factor construction stay as
   frozen.
2. Development data: through 2022-12-30. Per-market gating data: the 2023
   validation window only. A candidate passes the development gate if it
   improves mean return-energy by at least 2% against the frozen candidate and
   does not worsen the state composite in at least four of six markets on the
   2023 validation window.
3. A candidate that passes the development gate earns exactly one fresh
   preregistration on untouched data, following the full freeze/notarize/
   one-time machinery of the prospective study. Untouched means markets never
   screened or scored by this program, or a future locked period; the consumed
   six markets' post-2023 windows are barred.
4. Whatever that final evaluation returns — pass, fail, or inconclusive — the
   program ends. No further route exists.

## Final-route stop rule

This route is terminal for the diffusion factor-model program. The only exit
that produces a new outcome claim is the single fresh preregistration in
Phase C, and only if the Phase B gate passes with the prespecified thresholds.
Every other outcome — gate failure, development-gate failure, or any final
evaluation result — closes the program with its existing artifacts as the
complete record.
