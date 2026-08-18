"""One-time post-2023 evaluation for the frozen prospective market panel."""

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from config.config import load_config
from data.yf_loader import YFinanceDataPipeline
from diffusion.conditional_path import VARResidualPathDiffusion
from phase2b_benchmark import reconstruct_paths
from phase2c_benchmark import _restore_checkpoint, _score_baselines
from research.freeze_ftse100_pools import POOL_BASES
from research.freeze_phase2f import _panel_fingerprint
from research.market_selection import _aggregate_seed_scores, _bootstrap_difference
from research.phase2f_pooling import _linear_pool
from research.prospective_multimarket import (
    CANDIDATE_NAME,
    DIRECT_COVARIANCE_METRIC,
    EUROPE_BLOCK_CODES,
    MIN_SURVIVING_ASSETS,
    POST2023_MIN_SESSION_COVERAGE,
    STATE_METRICS,
    STRONG_BASELINES,
    block_sensitivity_interval,
    load_json,
    market_config,
    market_t_interval,
    population_decision,
    sha256,
    strong_baseline_scalar_decision,
    strong_baseline_state_decision,
)
from research.robustness import dependence_robust_comparison
from research.state_conditional_robustness_audit import _direct_covariance_scores
from sequences.baselines import Phase2ABaselines
from sequences.dataset import FactorStateSequenceBuilder
from sequences.evaluation import PathForecastEvaluator


def _verify_notarization(path, frozen_path, frozen):
    record = load_json(path)
    checks = {
        "study_id_match": record.get("study_id") == frozen.get("study_id"),
        "protocol_hash_match": record.get("frozen_protocol_sha256")
        == sha256(frozen_path),
        "public_commit_recorded": len(str(record.get("public_commit", ""))) >= 7,
        "public_url_recorded": str(record.get("public_url", "")).startswith("https://"),
        "published_before_evaluation": record.get("post2023_evaluation_started") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"Public preregistration notarization failed: {checks}")
    return record, checks


def _verify_frozen_hashes(frozen, args):
    expected = frozen["frozen_hashes"]
    actual_paths = {
        "base_config_sha256": Path(args.base_config),
        "registry_sha256": Path(args.registry),
        "pre2024_smoke_sha256": Path(args.smoke),
        "design_sha256": Path("PROSPECTIVE_MULTIMARKET_DESIGN.md"),
        "freeze_runner_sha256": Path(
            "diffusion_factor_model/research/freeze_prospective_multimarket.py"
        ),
        "evaluation_runner_sha256": Path(__file__),
        "shared_helper_sha256": Path(
            "diffusion_factor_model/research/prospective_multimarket.py"
        ),
    }
    checks = {
        name: sha256(path) == expected.get(name)
        for name, path in actual_paths.items()
    }
    if not all(checks.values()):
        raise ValueError(f"Frozen study hash mismatch: {checks}")
    return checks


def _aggregate_for_indices(origin_scores, indices):
    return {
        metric: {"mean": float(np.mean(np.asarray(values)[indices]))}
        for metric, values in origin_scores.items()
    }


def _state_decision_for_indices(candidate, baselines, indices):
    metrics = {CANDIDATE_NAME: _aggregate_for_indices(candidate, indices)}
    metrics.update(
        {
            baseline: _aggregate_for_indices(baselines[baseline], indices)
            for baseline in STRONG_BASELINES
        }
    )
    return strong_baseline_state_decision(metrics)


def _regime_labels(market, windows, threshold):
    trailing = market.rolling(20, min_periods=20).std(ddof=1)
    labels = []
    signals = []
    for dates in windows.context_dates:
        signal = float(trailing.loc[pd.Timestamp(dates[-1])])
        signals.append(signal)
        labels.append("high" if signal > threshold else "low")
    return labels, signals


TERMINAL_MARKET_STATUSES = ("evaluated", "terminal_data_attrition")

ATTRACTION_RULE_TEXT = (
    "a frozen asset survives the one-time download only if it resolves and has "
    "non-missing returns on at least "
    f"{POST2023_MIN_SESSION_COVERAGE:.0%} of post-2023 benchmark sessions; "
    "survivors keep frozen manifest order; a market needs at least "
    f"{MIN_SURVIVING_ASSETS} survivors; dropped assets are recorded and never "
    "replaced"
)


class DataAttritionError(ValueError):
    """Terminal prespecified outcome: too few frozen assets survive post-2023."""

    def __init__(self, message, dropped=None, survivor_coverage=None, raw_pins=None):
        super().__init__(message)
        self.dropped = list(dropped or [])
        self.survivor_coverage = dict(survivor_coverage or {})
        self.raw_pins = dict(raw_pins or {})


def _write_atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_market_artifact(path):
    """Return an existing terminal market artifact, or None when absent.

    A scored or terminally attrited market is never rescored; a non-terminal
    artifact is treated as corruption rather than silently rerun.
    """
    path = Path(path)
    if not path.exists():
        return None
    payload = load_json(path)
    if payload.get("status") not in TERMINAL_MARKET_STATUSES:
        raise ValueError(f"Market artifact has a non-terminal status: {path}")
    return payload


def _append_attempt_log(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _pin_raw_downloads(frozen_pipeline, current_pipeline, code, artifact_dir):
    """Copy both raw price downloads into the immutable artifact directory.

    Vendor-adjusted closes can be recomputed silently after the fact, so the
    exact one-time download is content-hashed and preserved here.
    """
    pins = {}
    for label, pipeline in (
        ("frozen_pre2024", frozen_pipeline),
        ("current_post2023", current_pipeline),
    ):
        source = pipeline.cache_path()
        if not source.exists():
            raise ValueError(
                f"{code}: raw download cache missing for {label}: {source}"
            )
        target = Path(artifact_dir) / f"{code}_raw_{label}.parquet"
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".tmp")
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
        pins[label] = {
            "source_cache": str(source),
            "pinned_copy": str(target),
            "sha256": sha256(target),
            "pinned_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    return pins


def _post2023_survivor_coverage(frozen_columns, current_prices, market_ticker):
    """Apply the prespecified post-2023 ticker attrition screen."""
    if market_ticker not in current_prices.columns:
        raise ValueError(
            f"Benchmark {market_ticker} missing from the one-time download."
        )
    benchmark = current_prices[market_ticker]
    sessions = benchmark.index[
        (benchmark.index >= pd.Timestamp("2024-01-01")) & benchmark.notna()
    ]
    if len(sessions) == 0:
        raise ValueError("No post-2023 benchmark sessions in the one-time download.")
    survivors, dropped, coverage = [], [], {}
    for column in frozen_columns:
        if column not in current_prices.columns:
            coverage[column] = 0.0
            dropped.append(column)
            continue
        returns = current_prices[column].pct_change(fill_method=None)
        fraction = float(returns.reindex(sessions).notna().mean())
        coverage[column] = fraction
        if fraction >= POST2023_MIN_SESSION_COVERAGE:
            survivors.append(column)
        else:
            dropped.append(column)
    return survivors, dropped, coverage


def _load_history_with_prespecified_attrition(
    frozen_config, current_config, max_stocks, code, artifact_dir
):
    """Load immutable history plus the one-time current extension.

    The pre-2024 panel is returned in full frozen width so its fingerprint can
    be verified unchanged; scoring then uses only the survivors of the
    prespecified attrition screen. The screen may drop assets, never add or
    re-select them.
    """
    frozen_pipeline = YFinanceDataPipeline(frozen_config)
    current_pipeline = YFinanceDataPipeline(current_config)
    frozen_prices = frozen_pipeline._download()
    current_prices = current_pipeline._download()
    raw_pins = _pin_raw_downloads(
        frozen_pipeline, current_pipeline, code, artifact_dir
    )
    frozen_raw, frozen_market = frozen_pipeline.prepare_raw_returns(
        frozen_prices, max_stocks=max_stocks
    )
    survivors, dropped, survivor_coverage = _post2023_survivor_coverage(
        list(frozen_raw.columns), current_prices, current_pipeline.market_ticker
    )
    if len(survivors) < MIN_SURVIVING_ASSETS:
        raise DataAttritionError(
            f"{code}: only {len(survivors)} frozen assets satisfy the "
            "post-2023 session-coverage rule; the prespecified minimum is "
            f"{MIN_SURVIVING_ASSETS}.",
            dropped=dropped,
            survivor_coverage=survivor_coverage,
            raw_pins=raw_pins,
        )
    current_subset = current_prices[[*survivors, current_pipeline.market_ticker]]
    current_raw, current_market = current_pipeline.prepare_raw_returns(
        current_subset, max_stocks=max_stocks
    )
    missing = [column for column in survivors if column not in current_raw.columns]
    if missing:
        raise ValueError(
            f"{code}: one-time download no longer reproduces surviving frozen "
            f"columns: {missing}"
        )
    cutoff = pd.Timestamp("2024-01-01")
    fit_end = pd.to_datetime(frozen_config.data.preprocess_fit_end_date)
    reference = frozen_raw.loc[frozen_raw.index <= fit_end]
    frozen_returns = frozen_pipeline.winsorize(frozen_raw, reference)
    current_post = current_raw.loc[current_raw.index >= cutoff]
    current_post = frozen_pipeline.winsorize(current_post, reference)
    returns = pd.concat([frozen_returns, current_post]).sort_index()
    returns = returns.loc[~returns.index.duplicated(keep="first")]
    market = pd.concat(
        [
            frozen_market,
            current_market.loc[current_market.index >= cutoff],
        ]
    ).sort_index()
    market = market.loc[~market.index.duplicated(keep="first")]
    index = returns.index.intersection(market.index)
    return (
        returns.loc[index],
        market.loc[index],
        survivors,
        dropped,
        survivor_coverage,
        raw_pins,
    )


def _evaluate_market(
    base_config,
    registered_market,
    frozen_market,
    evaluation_end,
    artifact_dir,
):
    code = frozen_market["code"]
    frozen_config = market_config(base_config, registered_market)
    current_config = market_config(
        base_config,
        registered_market,
        current=True,
        evaluation_end=evaluation_end,
    )
    (
        returns,
        market,
        survivors,
        dropped,
        survivor_coverage,
        raw_pins,
    ) = _load_history_with_prespecified_attrition(
        frozen_config,
        current_config,
        registered_market["max_stocks"],
        code,
        artifact_dir,
    )
    pre2024 = returns.loc[returns.index < "2024-01-01"]
    pre2024_market = market.reindex(pre2024.index)
    if _panel_fingerprint(pre2024, pre2024_market) != frozen_market[
        "pre2024_panel_fingerprint"
    ]:
        raise ValueError(f"{code}: pre-2024 panel fingerprint changed.")
    if list(pre2024.columns) != frozen_market["asset_columns"]:
        raise ValueError(f"{code}: frozen asset columns changed.")
    manifest = Path(frozen_market["universe_manifest"])
    if sha256(manifest) != frozen_market["universe_manifest_sha256"]:
        raise ValueError(f"{code}: universe manifest hash mismatch.")
    scoring_returns = returns[survivors]

    splits = FactorStateSequenceBuilder(
        current_config,
        context_length=current_config.temporal.context_length,
        horizon=current_config.temporal.horizon,
        evaluation_stride=current_config.temporal.horizon,
        train_end_date=frozen_market["train_end"],
        validation_end_date=frozen_market["validation_end"],
    ).build(scoring_returns, market)
    if len(splits.test) == 0:
        raise ValueError(f"{code}: no complete post-2023 evaluation origin.")
    baselines = Phase2ABaselines(current_config, splits)
    evaluator = PathForecastEvaluator(splits.latent_metadata["n_mean_factors"])
    n_paths = current_config.temporal.ensemble_paths
    baseline_scores, rows, origins = _score_baselines(
        current_config,
        splits,
        baselines,
        evaluator,
        n_paths,
        len(splits.test),
    )
    for origin, row in enumerate(rows):
        for baseline in baselines.METHODS:
            baseline_scores[baseline][origin].update(
                _direct_covariance_scores(
                    row["baseline_return_paths"][baseline],
                    row["target_returns"],
                    np.full((1, len(baselines.stocks)), 1.0 / len(baselines.stocks)),
                )
            )

    horizons = [5, 10, current_config.temporal.horizon]
    baseline_horizon = {
        horizon: {baseline: [] for baseline in STRONG_BASELINES}
        for horizon in horizons
    }
    for row in rows:
        for horizon in horizons:
            for baseline in STRONG_BASELINES:
                baseline_horizon[horizon][baseline].append(
                    evaluator.score(
                        row["baseline_state_paths"][baseline][:, :horizon],
                        row["baseline_return_paths"][baseline][:, :horizon],
                        row["target_states"][:horizon],
                        row["target_returns"][:horizon],
                    )
                )

    seed_scores = []
    seed_horizon_scores = {horizon: [] for horizon in horizons}
    seed_reports = []
    for frozen_seed in frozen_market["seed_reports"]:
        seed = int(frozen_seed["seed"])
        checkpoint = Path(frozen_seed["checkpoint"])
        if sha256(checkpoint) != frozen_seed["checkpoint_sha256"]:
            raise ValueError(f"{code}: checkpoint hash mismatch for seed {seed}.")
        model = VARResidualPathDiffusion(
            current_config,
            state_dim=splits.train_states.shape[1],
            horizon=current_config.temporal.horizon,
            n_mean_factors=splits.latent_metadata["n_mean_factors"],
        )
        _restore_checkpoint(model, splits, checkpoint)
        weight = float(frozen_seed["pool_weights"]["student_t_base_pool"])
        scores = []
        horizon_scores = {horizon: [] for horizon in horizons}
        for origin, row in enumerate(rows):
            diffusion = model.sample(
                row["context"],
                n_paths=n_paths,
                seed=seed * 10000 + 80000 + origin * n_paths,
            )
            states = _linear_pool(
                diffusion,
                row["baseline_state_paths"][POOL_BASES["student_t_base_pool"]],
                weight,
            )
            generated = reconstruct_paths(
                splits, states, row["innovations"], baselines.stocks
            )
            score = evaluator.score(
                states,
                generated,
                row["target_states"],
                row["target_returns"],
            )
            score.update(
                _direct_covariance_scores(
                    generated,
                    row["target_returns"],
                    np.full((1, len(baselines.stocks)), 1.0 / len(baselines.stocks)),
                )
            )
            scores.append(score)
            for horizon in horizons:
                horizon_scores[horizon].append(
                    evaluator.score(
                        states[:, :horizon],
                        generated[:, :horizon],
                        row["target_states"][:horizon],
                        row["target_returns"][:horizon],
                    )
                )
        seed_scores.append(scores)
        for horizon in horizons:
            seed_horizon_scores[horizon].append(horizon_scores[horizon])
        seed_reports.append(
            {
                "seed": seed,
                "frozen_student_t_pool_weight": weight,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": frozen_seed["checkpoint_sha256"],
            }
        )

    all_scores = {
        **baseline_scores,
        CANDIDATE_NAME: [score for seed_rows in seed_scores for score in seed_rows],
    }
    metrics = evaluator.aggregate(all_scores)
    pooled_candidate = _aggregate_seed_scores(seed_scores)
    origin_baselines = {
        baseline: {
            metric: [row[metric] for row in baseline_scores[baseline]]
            for metric in baseline_scores[baseline][0]
        }
        for baseline in baselines.METHODS
    }
    state_decision = strong_baseline_state_decision(metrics)
    covariance_decision = strong_baseline_scalar_decision(
        metrics[CANDIDATE_NAME][DIRECT_COVARIANCE_METRIC]["mean"],
        {
            baseline: metrics[baseline][DIRECT_COVARIANCE_METRIC]["mean"]
            for baseline in STRONG_BASELINES
        },
    )

    paired = {
        baseline: {
            metric: _bootstrap_difference(
                pooled_candidate[metric],
                origin_baselines[baseline][metric],
                seed=20260816 + baseline_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(
                (*STATE_METRICS, DIRECT_COVARIANCE_METRIC)
            )
        }
        for baseline_index, baseline in enumerate(STRONG_BASELINES)
    }
    robust = {
        baseline: {
            metric: dependence_robust_comparison(
                pooled_candidate[metric],
                origin_baselines[baseline][metric],
                seed=20260817 + baseline_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(
                (*STATE_METRICS, DIRECT_COVARIANCE_METRIC)
            )
        }
        for baseline_index, baseline in enumerate(STRONG_BASELINES)
    }

    horizon_results = {}
    for horizon in horizons:
        candidate_rows = _aggregate_seed_scores(seed_horizon_scores[horizon])
        baseline_rows = {
            baseline: {
                metric: [row[metric] for row in baseline_horizon[horizon][baseline]]
                for metric in baseline_horizon[horizon][baseline][0]
            }
            for baseline in STRONG_BASELINES
        }
        horizon_results[str(horizon)] = _state_decision_for_indices(
            candidate_rows,
            baseline_rows,
            np.arange(len(rows)),
        )

    labels, signals = _regime_labels(
        market,
        splits.test,
        frozen_market["volatility_regime"]["threshold"],
    )
    regime_results = {}
    for label in ("low", "high"):
        indices = np.asarray([i for i, value in enumerate(labels) if value == label])
        regime_results[label] = {"n_origins": int(len(indices))}
        if len(indices) >= 5:
            regime_results[label]["state_decision"] = _state_decision_for_indices(
                pooled_candidate, origin_baselines, indices
            )
        else:
            regime_results[label]["state_decision"] = None
    high = regime_results["high"]["state_decision"]
    low = regime_results["low"]["state_decision"]
    high_minus_low = None
    if high is not None and low is not None:
        high_minus_low = float(
            np.log(high["strong_baseline_ratio"])
            - np.log(low["strong_baseline_ratio"])
        )

    return {
        "code": code,
        "name": frozen_market["name"],
        "country": frozen_market["country"],
        "status": "evaluated",
        "protocol_checks": {
            "pre2024_panel_fingerprint_match": True,
            "asset_columns_match": True,
            "manifest_hash_match": True,
            "checkpoint_hashes_match": True,
            "model_or_weight_retuning_after_2023": False,
            "post2023_attrition_rule_prespecified": True,
            "raw_downloads_pinned": True,
        },
        "evaluation_period": {
            "start": str(splits.test_returns.index.min().date()),
            "end": str(splits.test_returns.index.max().date()),
            "n_days": len(splits.test_returns),
            "n_nonoverlapping_origins": len(splits.test),
            "n_assets": len(baselines.stocks),
        },
        "scoring_asset_columns": list(survivors),
        "post2023_ticker_attrition": {
            "rule": ATTRACTION_RULE_TEXT,
            "n_frozen_assets": len(frozen_market["asset_columns"]),
            "n_survivors": len(survivors),
            "dropped_assets": dropped,
            "survivor_session_coverage": survivor_coverage,
        },
        "raw_download_pins": raw_pins,
        "state_decision": state_decision,
        "covariance_decision": covariance_decision,
        "metrics": metrics,
        "paired_origin_bootstrap": paired,
        "dependence_robust_comparisons": robust,
        "horizon_results": horizon_results,
        "horizon_log_ratio_change_20_minus_5": float(
            np.log(horizon_results[str(current_config.temporal.horizon)]["strong_baseline_ratio"])
            - np.log(horizon_results["5"]["strong_baseline_ratio"])
        ),
        "volatility_regime": {
            "frozen_threshold": frozen_market["volatility_regime"],
            "origin_signals": signals,
            "origin_labels": labels,
            "results": regime_results,
            "high_minus_low_log_strong_ratio": high_minus_low,
        },
        "origin_level_scores": {
            CANDIDATE_NAME: pooled_candidate,
            **origin_baselines,
        },
        "seed_reports": seed_reports,
        "origins": origins,
    }


def _population_synthesis(markets, required_passes):
    state = population_decision(
        [row["state_decision"] for row in markets], required_passes
    )
    covariance = population_decision(
        [row["covariance_decision"] for row in markets], required_passes
    )
    state_by_code = {row["code"]: row["state_decision"] for row in markets}
    covariance_by_code = {row["code"]: row["covariance_decision"] for row in markets}
    europe_block = {
        "block_definition": list(EUROPE_BLOCK_CODES),
        "role": (
            "prespecified sensitivity analysis treating the integrated Alpine / "
            "Eurozone-adjacent markets as one replication unit; it cannot "
            "replace the primary population decision"
        ),
        "primary_state": block_sensitivity_interval(state_by_code),
        "key_secondary_covariance": block_sensitivity_interval(covariance_by_code),
    }
    horizon_changes = [row["horizon_log_ratio_change_20_minus_5"] for row in markets]
    horizon_interval = market_t_interval(np.exp(horizon_changes))
    regime_changes = [
        row["volatility_regime"]["high_minus_low_log_strong_ratio"]
        for row in markets
        if row["volatility_regime"]["high_minus_low_log_strong_ratio"] is not None
    ]
    regime_interval = (
        market_t_interval(np.exp(regime_changes)) if len(regime_changes) >= 2 else None
    )
    sizes = np.asarray([row["evaluation_period"]["n_assets"] for row in markets])
    effects = np.log(
        [row["state_decision"]["strong_baseline_ratio"] for row in markets]
    )
    regression = stats.linregress(sizes, effects)
    return {
        "primary_state": state,
        "key_secondary_covariance": covariance,
        "claim_boundary": (
            "covariance cannot rescue a failed primary state decision"
        ),
        "claim_trichotomy": (
            "each population claim is classified by the prespecified "
            "pass / fail / inconclusive rule; the one-time outcome is terminal "
            "for this candidate regardless of classification"
        ),
        "europe_block_sensitivity": europe_block,
        "horizon_20_minus_5": horizon_interval,
        "volatility_regime_high_minus_low": regime_interval,
        "exploratory_cross_section_meta_regression": {
            "slope_log_ratio_per_asset": float(regression.slope),
            "intercept": float(regression.intercept),
            "two_sided_p": float(regression.pvalue),
            "r_value": float(regression.rvalue),
        },
    }


def run(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(
            f"One-time study output already exists: {output}; refusing overwrite."
        )
    if pd.to_datetime(args.evaluation_end) < pd.Timestamp("2024-01-02"):
        raise ValueError("Evaluation end must include post-2023 observations.")
    frozen_path = Path(args.frozen_protocol)
    frozen = load_json(frozen_path)
    if frozen.get("status") != "frozen_before_post2023_evaluation":
        raise ValueError("Study protocol is not frozen for one-time evaluation.")
    hash_checks = _verify_frozen_hashes(frozen, args)
    notarization, notarization_checks = _verify_notarization(
        args.notarization, frozen_path, frozen
    )
    registry = load_json(args.registry)
    registered = {row["code"]: row for row in registry["primary_markets"]}
    base = load_config(args.base_config)

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    state_path = artifact_dir / "one_time_state.json"
    if state_path.exists():
        state = load_json(state_path)
        if state.get("evaluation_end_exclusive") != args.evaluation_end:
            raise ValueError(
                "evaluation_end is pinned by the first one-time attempt: "
                f"expected {state.get('evaluation_end_exclusive')!r}, "
                f"got {args.evaluation_end!r}."
            )
    else:
        _write_atomic_json(
            state_path,
            {
                "study_id": frozen["study_id"],
                "evaluation_end_exclusive": args.evaluation_end,
                "first_attempt_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    attempt = {
        "attempt_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_end_exclusive": args.evaluation_end,
        "markets": {},
    }
    market_reports = []
    for frozen_market in frozen["markets"]:
        code = frozen_market["code"]
        artifact_path = artifact_dir / f"{code}.json"
        existing = _read_market_artifact(artifact_path)
        if existing is not None:
            market_reports.append(existing)
            attempt["markets"][code] = f"reused:{existing['status']}"
            continue
        try:
            report = _evaluate_market(
                base,
                registered[code],
                frozen_market,
                args.evaluation_end,
                artifact_dir,
            )
            _write_atomic_json(artifact_path, report)
            market_reports.append(report)
            attempt["markets"][code] = "scored"
        except DataAttritionError as exc:
            terminal = {
                "code": code,
                "name": frozen_market["name"],
                "country": frozen_market.get("country"),
                "status": "terminal_data_attrition",
                "error_type": "DataAttritionError",
                "error": str(exc),
                "post2023_ticker_attrition": {
                    "rule": ATTRACTION_RULE_TEXT,
                    "dropped_assets": exc.dropped,
                    "survivor_session_coverage": exc.survivor_coverage,
                },
                "raw_download_pins": exc.raw_pins,
            }
            _write_atomic_json(artifact_path, terminal)
            market_reports.append(terminal)
            attempt["markets"][code] = "terminal_data_attrition"
        except Exception as exc:
            attempt["markets"][code] = f"transient_failure:{type(exc).__name__}"
            print(
                f"{code}: transient operational failure preserved in the attempt "
                f"log ({type(exc).__name__}: {exc})"
            )

    _append_attempt_log(artifact_dir / "attempt_log.jsonl", attempt)

    evaluated = [row for row in market_reports if row["status"] == "evaluated"]
    terminal = [
        row for row in market_reports if row["status"] in TERMINAL_MARKET_STATUSES
    ]
    all_terminal = len(terminal) == len(frozen["markets"])
    population = None
    population_note = None
    if all_terminal:
        if len(evaluated) == len(frozen["markets"]):
            population = _population_synthesis(
                evaluated,
                frozen["protocol"]["required_individual_market_passes"],
            )
        else:
            population_note = (
                "Population claims are not evaluable: the prespecified rule "
                "requires every eligible market to be scored. Terminal "
                "attrition markets are preserved without substitution."
            )
    else:
        missing = [
            frozen_market["code"]
            for frozen_market in frozen["markets"]
            if not (artifact_dir / f"{frozen_market['code']}.json").exists()
        ]
        print(
            "One-time evaluation is operationally incomplete; rerun with the "
            f"same pinned evaluation end to retry: {missing}"
        )
        return None

    report = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "one_time_evaluation_complete"
            if len(evaluated) == len(frozen["markets"])
            else "one_time_evaluation_terminal_attrition"
        ),
        "study_id": frozen["study_id"],
        "frozen_protocol": str(frozen_path),
        "frozen_protocol_sha256": sha256(frozen_path),
        "public_notarization": notarization,
        "global_protocol_checks": {
            **hash_checks,
            **notarization_checks,
            "output_was_absent_before_run": True,
            "all_registered_markets_preserved": True,
            "post_outcome_retuning": False,
            "scored_markets_never_rescored": True,
            "evaluation_end_pinned_on_first_attempt": True,
        },
        "evaluation_end_exclusive": args.evaluation_end,
        "one_time_artifacts": {
            "directory": str(artifact_dir),
            "state": str(state_path),
            "attempt_log": str(artifact_dir / "attempt_log.jsonl"),
        },
        "population_synthesis": population,
        "population_note": population_note,
        "markets": market_reports,
    }
    _write_atomic_json(output, report)
    print(f"Immutable one-time multi-market report saved to {output}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="research_multimarket_base.yaml")
    parser.add_argument("--registry", default="research_multimarket_markets.json")
    parser.add_argument(
        "--smoke",
        default="research_output/prospective_multimarket/pre2024_smoke.json",
    )
    parser.add_argument(
        "--frozen-protocol",
        default="research_output/prospective_multimarket/frozen_protocol.json",
    )
    parser.add_argument("--notarization", required=True)
    parser.add_argument(
        "--evaluation-end",
        required=True,
        help="Exclusive YYYYMMDD end date fixed at the public one-time run.",
    )
    parser.add_argument(
        "--artifact-dir",
        default="research_output/prospective_multimarket/one_time",
        help="Per-market immutable artifacts; scored markets are never rescored.",
    )
    parser.add_argument(
        "--output",
        default="research_output/prospective_multimarket/evaluation.json",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
