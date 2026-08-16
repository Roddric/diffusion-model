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
| 2026-08-11 | Calibration and overlapping-origin power audit | Post-hoc audit | Locked S&P origins; stride-5 overlapping origins | Locked composite reproduced exactly; rank histograms and energy-term decomposition show the pool gain over Gaussian VAR comes from the spread term; at 114 overlapping origins the pool improves on Gaussian VAR under HAC (p<0.00001) and also separates from Student-t VAR, but this is post-hoc sensitivity evidence only |
| 2026-08-12 | FTSE 100 untouched-market dual-pool freeze | External untouched-market holdout | FTSE training through 2022; 2023 checkpoint and weight selection | 87 eligible securities (deterministic screen); all three seeds select weight 0.25 for both bases; checkpoint steps 2250/2500/2000; post-2023 data not touched |
| 2026-08-12 | One-time FTSE 100 dual-pool scoring | External untouched-market holdout | One-time scoring of 2024-01-02 → 2026-08-10 (29 origins) after external notarization of commit feff0986 | Primary PASS: Gaussian-base pool composite 0.982038 versus VAR-GARCH (energy ratio 0.98243, RMSE ratio 0.98165), HAC p=0.0017 on both co-primary metrics, bootstrap intervals exclude zero. Secondary PASS by rule: Student-t-base pool composite 0.996315 versus Student-t VAR, but statistically tied (HAC p≈0.27-0.29, bootstrap intervals include zero). Horizon persistence of pool-over-own-base holds at 5/10/20 days; BH-adjusted secondary endpoints significant for variogram and log-vol factor RMSE only |
| 2026-08-12 | FTSE calibration and overlapping-origin power audit | Post-hoc audit | Locked FTSE origins; stride-5 overlapping origins | Both locked composites reproduced exactly; Gaussian-base pool edge-bin share equals the uniform value; at 116 overlapping origins the pool stays significant versus Gaussian VAR (HAC p≈0.03) and tied with Student-t VAR, with borderline Student-t-base separation on RMSE (HAC p=0.033) |
| 2026-08-14 | HSI untouched-market dual-pool freeze | Preregistered untouched-market holdout | HSI training through 2022; 2023 checkpoint and weight selection | 61 eligible securities; Gaussian-base weights 0.50/0.75/0.75 and Student-t-base weights 0.25/0.50/1.00 for seeds 42/314/2718; checkpoint and pre-2024 panel hashes published at commit `e694256` before outcome access |
| 2026-08-14 | One-time HSI dual-pool scoring | Preregistered untouched-market holdout | One-time 2024-01-02 → 2026-08-12 panel scoring (28 non-overlapping origins; targets 2024-03-28 → 2026-07-16) | Primary FAIL: Gaussian-base composite 1.016159 versus VAR-GARCH (energy ratio 0.99886, RMSE ratio 1.03376). Secondary FAIL: Student-t-base composite 1.009953 versus Student-t VAR (energy ratio 1.00584, RMSE ratio 1.01408). Both results preserved; HSI sample consumed |
| 2026-08-15 | Reconstruction oracle-ceiling attribution | Post-primary diagnostic | S&P pre-2024 development only; 22 non-overlapping 2022–2023 origins | Eight-coalition Shapley diagnostic attributes 51.6% of the five-metric oracle loss reduction to mean-state error, 43.2% to innovations, and 5.1% to volatility-state error. Five-block cross-fitted loading diagnostics show modest representation headroom (mean ratio 0.982; volatility-PCA ratio 0.937). No candidate selected; no consumed external sample loaded. |
| 2026-08-15 | State-conditional innovation reconstruction | Post-primary development | Train through 2020; select `k` on 2021; evaluate once on 22 pre-2024 2022–2023 origins | Validation selected a 32-neighbor log-volatility-state-conditioned full-vector bootstrap. Development composite 0.981565 versus Independent-GARCH; four of five metrics improve, energy ratio 0.99710 with paired one-sided p=0.0064, and the fixed Phase 3A gate passes. Candidate retained for a new external protocol; no post-2023 external sample loaded. |
| 2026-08-15 | State-conditional fixed-candidate robustness freeze — **executed; FAIL** | Post-selection robustness protocol | Pre-2024 S&P development origins only | Protocol and exact audit code publicly frozen at commit `d0d67c6` before scoring. All five 20-path repeats improve (median composite 0.97651); the 100-path composite is 0.97644 with energy ratio 0.99863 (paired one-sided p=0.0744), and the candidate beats unconditional full-vector resampling. The all-requirements gate nevertheless fails: 50-path energy ratio is 1.00028, and the training-defined low-volatility regime (only 1 of 22 origins) has composite 1.08874. Direct covariance and VaR safety bounds pass. External promotion forbidden; no retuning or external sample access. |
| 2026-08-15 | Mean-state factor/horizon diagnostic | Post-primary diagnostic | Validation 2021 and S&P 2022–2023; pre-2024 only | Market contributes 30.7% of Phase2F mean-state MSE but is slightly worse than Student-t VAR; momentum and reversal strongly favor Phase2F. Small market bias and weak coverage indicate calibration rather than a missing nonlinear conditional mean. No candidate scored in this diagnostic. |
| 2026-08-15 | Factorwise mean-state routing — **executed; descriptive gate FAIL** | Post-diagnostic development protocol | Route sources using 2021 validation only; describe on 2022–2023 | Frozen rule routes market and illiquidity to Student-t VAR while retaining Phase2F for momentum, reversal, dispersion, and all volatility states. Mean-factor RMSE improves (ratio 0.99456; paired p=0.0536), as do state energy (0.99660) and state RMSE (0.99587), but energy support narrowly misses the frozen p<0.10 requirement (p=0.1058). Return composite worsens to 1.01034, including tail-error ratio 1.04274. Gate fails; no cross-market replication or promotion. |
| 2026-08-15 | Cross-market Gaussian-base synthesis | Post-hoc descriptive meta-analysis | Locked S&P, preregistered FTSE, and preregistered HSI primary comparisons; no new model scoring | Equal-market mean log-composite corresponds to ratio 0.99263 with market-level Student-t 95% CI [0.94378, 1.04400]. Preregistered external-only ratio is 0.99895 [0.80409, 1.24103]. DL/HKSJ sensitivity is 0.98554 [0.95076, 1.02160], estimated I²=42.0%. CSI excluded because it scores a different candidate. Analysis declared post-hoc; market is the replication unit; no universal effect claimed. |
| 2026-08-16 | Prospective multi-market feasibility screen and design lock preparation | Prospective design/development input | Constituent revisions and prices ending 2023 only; no post-2023 candidate-market outcomes loaded | Seven markets were registered before price screening. DAX (35 assets), ASX 200 (100), CAC 40 (38), NIFTY 50 (46), SMI (18), and TSX 60 (57) pass the fixed eligibility rule; IBEX 35 is excluded at 13 assets with no replacement. This fixes `K=6` and the individual-market threshold at five. The state estimand is the Student-t-base pool versus the better of Gaussian and Student-t VAR; direct covariance is key secondary. Generic freeze/evaluation code and public-notarization guards were completed before checkpoint training. Artifact: `research_output/prospective_multimarket/pre2024_smoke.json`. |

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
16. factorial oracle attribution of mean-state, volatility-state, and innovation
    error, with cross-fitted loading-representation diagnostics.
17. log-volatility-state-conditioned nearest-neighbor innovation resampling.

This list should be treated as a lower bound rather than a formal historical trial
count. It is why the 2024–2026 result is described as retrospective locked holdout
evidence rather than a prospectively preregistered test.

## Pre-registered experiments (entered before execution, 2026-08-11)

| Registered | Experiment | Evidence class | Data role | Prespecified protocol |
|---|---|---|---|---|
| 2026-08-11 | FTSE 100 untouched-market evaluation (dual pool) — **executed 2026-08-12; primary PASS, secondary PASS by rule but statistically tied** | Preregistered untouched-market holdout | FTSE training through 2022; 2023 checkpoint and weight selection; one-time 2024–2026 scoring | Universe: December 2023 FTSE 100 snapshot (sha256 13d55d86de45…ddcf), leading 100 manifest entries passing the 95% training-era coverage screen dated 2022-12-30. Market benchmark ISF.L. All architecture, budgets, seeds (42/314/2718), path counts (20), and selection rules transplanted unchanged from the locked S&P Phase 2F protocol; no FTSE-specific tuning. Primary endpoint: Gaussian-VAR-base pool versus VAR-GARCH under the locked S&P success rule. Prespecified secondary endpoint: Student-t-VAR-base pool versus Student-t VAR under the same rule. Commit feff0986843ddd454b908e7084cb80da688b9918 (tag ftse100-preregistration-2026-08-11) was externally notarized before any post-2023 FTSE observation was downloaded. The consumed FTSE 2024–2026 sample cannot be reused for selection. |
| 2026-08-11 | S&P calibration and overlapping-origin power audit — **executed 2026-08-11** | Post-hoc audit | Locked S&P origins; locked window rescored at overlapping stride-5 origins | Rank-histogram and energy-decomposition diagnostics on the locked 29 origins; stride-5 overlapping-origin scoring with HAC and moving-block inference. No model selection; the locked decision is unchanged. Artifact: `research_output/sp500_confirmation/posthoc_calibration_power_audit.json`. |
| 2026-08-12 | FTSE calibration and overlapping-origin power audit — **executed 2026-08-12** | Post-hoc audit | Locked FTSE origins; locked window rescored at overlapping stride-5 origins | Both locked composites reproduced exactly (primary 0.982038, secondary 0.996315). Gaussian-base pool edge-bin share hits the uniform value (0.095); both VAR baselines are under-dispersed (0.112-0.114). The Gaussian-relative pool gain improves both energy terms (distance and spread). At 116 overlapping origins the Gaussian-base pool stays significant versus Gaussian VAR (HAC p≈0.03) and remains tied with Student-t VAR; the Student-t-base pool shows borderline separation from Student-t VAR (RMSE HAC p=0.033). Post-hoc sensitivity evidence only; locked decisions unchanged. Artifact: `research_output/ftse100_confirmation/posthoc_ftse_calibration_power_audit.json`. |
| 2026-08-12 | Hang Seng Index untouched-market evaluation (dual pool) — **executed 2026-08-14; primary FAIL, secondary FAIL** | Preregistered untouched-market holdout | HSI training through 2022; 2023 checkpoint and weight selection; one-time 2024–2026 scoring | Universe: December 2023 HSI snapshot (sha256 87307877e58f…de42), leading 100 manifest entries passing the 95% training-era coverage screen dated 2022-12-30. Market benchmark 2800.HK. All architecture, budgets, seeds (42/314/2718), path counts (20), and selection rules transplanted unchanged from the locked S&P Phase 2F protocol; no HSI-specific tuning. Primary endpoint: Gaussian-VAR-base pool versus VAR-GARCH under the locked S&P success rule. Prespecified secondary endpoint: Student-t-VAR-base pool versus Student-t VAR under the same rule. Commit c6085db873be3d8d730e162c0231648c895b447a (tag hsi-preregistration-2026-08-12) was publicly pushed before post-2023 HSI observations were downloaded; the frozen protocol and checkpoint hashes were additionally published at commit e694256 before outcome access. The primary Gaussian-base composite was 1.016159 (energy 0.99886, RMSE 1.03376); the secondary Student-t-base composite was 1.009953 (energy 1.00584, RMSE 1.01408). Both rules failed, the results were preserved, and the HSI sample is consumed. |

Known preregistration data constraint: as of 2026-08-10, seven constituents of the
December 2023 FTSE 100 snapshot (AHT.L, BDEV.L, BTA.L, DPH.L, HL.L, PHNX.L,
SMDS.L) no longer resolve on Yahoo Finance, and six further constituents fail the
95% training-era coverage screen. The eligible universe is therefore the
deterministic manifest-order screen intersection (87 securities passed in both the
2026-08-10 pre-2024 smoke test and the 2026-08-12 one-time run), consistent with
the CSI 300 external replication precedent (64 of 100 eligible).

The FTSE 100 2024–2026 sample is now consumed for the primary and secondary
decisions above. It may be used for post-hoc audits only, never for further model
or weight selection.

Known preregistration data constraint (HSI): as of 2026-08-12, one constituent of
the December 2023 HSI snapshot (0011.HK) no longer resolves on Yahoo Finance, and
eighteen further constituents fail the 95% training-era coverage screen. The
eligible universe is therefore the deterministic manifest-order screen intersection
(61 securities passed in the 2026-08-12 pre-2024 smoke test), consistent with the
CSI 300 (64 of 100) and FTSE 100 (87 of 100) precedents.

The HSI 2024–2026 sample is now consumed for both prespecified decisions. Its
failed primary and secondary results may be audited post hoc but may never be
used for model, weight, endpoint, or baseline selection.

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
