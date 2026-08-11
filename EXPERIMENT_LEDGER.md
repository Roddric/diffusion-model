# Experiment ledger

This ledger was reconstructed on 2026-08-03 from saved artifact metadata and local
filesystem timestamps. It was not maintained contemporaneously during the original
research, so it improves transparency but cannot retroactively prove researcher
blindness. Every future experiment should be entered here before execution and
linked to a version-control commit.

## Evidence classes

- **Legacy diagnostic:** early implementation work, not evidence for the active claim.
- **Development:** model or market choices using observations ending in 2023.
- **Primary retrospective holdout:** the one-time S&P post-2023 score.
- **Post-primary exploratory:** designed or run after the primary score was known.
- **Post-hoc audit:** sensitivity or inference applied to the locked specification;
  it cannot change the original decision.

## Reconstructed chronology

| Local time | Experiment | Evidence class | Data role | Decision/artifact |
|---|---|---|---|---|
| 2026-07-12–07-29 | Original generative model and Phases 1–2D | Legacy diagnostic | Mixed historical development/test splits | `prod_output/`; pure diffusion generally fails strong baselines |
| 2026-07-30 10:30 | Retrieve December 2023 S&P 500 and CSI 300 archives | Development input | Historical membership archive | `research_data/universes/`; retrieved after the later holdout period had occurred |
| 2026-07-30 10:59–12:14 | Matched CSI/S&P market comparison | Development | Train 2015–2020; select checkpoint 2021; compare 2022–2023 | S&P selected although diffusion remains worse than VAR in both markets |
| 2026-07-30 12:27–12:39 | Phase 2E innovation diffusion | Development | Pre-2024 only | Rejected; composite 1.0631 |
| 2026-07-30 12:42 | Phase 2F finite pool | Development | Pre-2024 only | Validation-selected Gaussian-VAR/diffusion pool retained |
| 2026-07-30 12:50–13:06 | Final training and protocol lock | Development/freeze | Train through 2022; validate 2023 | Three checkpoints, weights 0.25/0.25/0.50, hashes recorded |
| 2026-07-30 13:21 | S&P 2024–2026 one-time score | Primary retrospective holdout | 29 non-overlapping origins | Composite 0.98010 versus Gaussian VAR; original artifact made immutable |
| 2026-07-30 13:41–13:46 | Phase 3A/3B | Post-primary exploratory | Pre-2024 development reused after primary result known | Reconstruction and stronger-base candidates rejected by recorded gates |
| 2026-07-30 15:10–16:09 | CSI candidate lock and cross-market score | Post-primary cross-market holdout | CSI 2024–2026, same historical period as S&P | Promotion rule failed; sample consumed |
| 2026-07-30 16:28–17:49 | Phase 4A–4C | Post-primary exploratory | Pre-2024 only | Regime VAR, shared diffusion, and adapters rejected |
| 2026-08-03 | Dependence-robust audit | Post-hoc audit | Locked S&P origins | Exact point-estimate reproduction; block bootstrap and HAC support Gaussian-VAR comparison but not Student-t comparison |
| 2026-08-03 | Monte Carlo path-count audit | Post-hoc audit | Locked S&P origins | Frozen weights evaluated on nested 20/50/100-path ensembles; no model selection |
| 2026-08-03 | Expanded baseline audit | Post-hoc exploratory | Locked S&P origins | At 100 paths the pool beats Gaussian/ridge VAR, diagonal AR, boosted AR, and persistence; Student-t comparison remains inconclusive |
| 2026-08-03 | Observable portfolio-risk audit | Post-hoc audit | Locked S&P origins | No consistent risk advantage; 5% VaR pinball loss is worse than Gaussian VAR |

## Trial families disclosed

The repository contains at least the following materially distinct trial families:

1. i.i.d. latent diffusion and reconstruction variants;
2. factor/volatility integration;
3. conditional full-path diffusion;
4. VAR-residual path diffusion;
5. split mean/volatility heads and loss weighting;
6. path-energy checkpoint selection;
7. market choice between CSI 300 and S&P 500;
8. residual-amplitude calibration;
9. innovation-recursive diffusion;
10. finite VAR/diffusion pooling and five candidate weights;
11. joint innovation reconstruction;
12. Gaussian, Student-t, and empirical classical pool bases;
13. regime-conditioned Student-t VAR;
14. shared cross-market diffusion;
15. shared trunk with low-rank market adapters.

This list should be treated as a lower bound rather than a formal historical trial
count. It is why the 2024–2026 result is described as retrospective locked holdout
evidence rather than a prospectively preregistered test.

## Pre-registered experiments (entered before execution, 2026-08-11)

| Registered | Experiment | Evidence class | Data role | Prespecified protocol |
|---|---|---|---|---|
| 2026-08-11 | FTSE 100 untouched-market evaluation (dual pool) | Preregistered untouched-market holdout | FTSE training through 2022; 2023 checkpoint and weight selection; one-time 2024–2026 scoring | Universe: December 2023 FTSE 100 snapshot (sha256 13d55d86de45…ddcf), leading 100 manifest entries passing the 95% training-era coverage screen dated 2022-12-30. Market benchmark ISF.L. All architecture, budgets, seeds (42/314/2718), path counts (20), and selection rules transplanted unchanged from the locked S&P Phase 2F protocol; no FTSE-specific tuning. Primary endpoint: Gaussian-VAR-base pool versus VAR-GARCH under the locked S&P success rule. Prespecified secondary endpoint: Student-t-VAR-base pool versus Student-t VAR under the same rule. Git commit and tag recorded before any post-2023 FTSE observation is downloaded. |
| 2026-08-11 | S&P calibration and overlapping-origin power audit | Post-hoc audit | Locked S&P origins; locked window rescored at overlapping stride-5 origins | PIT/rank-histogram and energy-decomposition diagnostics on the locked 29 origins; stride-5 overlapping-origin scoring with HAC and moving-block inference. No model selection; the locked decision is unchanged. Artifact: `research_output/sp500_confirmation/posthoc_calibration_power_audit.json`. |

Known preregistration data constraint: as of 2026-08-10, seven constituents of the
December 2023 FTSE 100 snapshot (AHT.L, BDEV.L, BTA.L, DPH.L, HL.L, PHNX.L,
SMDS.L) no longer resolve on Yahoo Finance, and six further constituents fail the
95% training-era coverage screen. The eligible universe is therefore the
deterministic manifest-order screen intersection (87 securities passed in the
2026-08-10 pre-2024 smoke test), consistent with the CSI 300 external replication
precedent (64 of 100 eligible).

## Rule for future work

Before another external or prospective score is opened:

1. create a dated protocol containing universe, endpoints, baselines, training
   budget, seeds, path count, and decision rule;
2. commit and tag that protocol and the executable code;
3. publish the commit hash or send it to the supervisor before downloading outcomes;
4. run the evaluation once and preserve both success and failure artifacts;
5. never use the consumed sample for subsequent selection.

Create the immutable protocol record with
`diffusion_factor_model/research/register_experiment.py`; it records configuration
hashes and the current Git commit and refuses to overwrite an existing protocol.
