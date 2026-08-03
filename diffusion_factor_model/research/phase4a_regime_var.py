"""Phase 4A: pre-2024 multi-market regime-conditioned Student-t VAR."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import load_config
from data.loader import DataPipeline
from data.yf_loader import YFinanceDataPipeline
from dynamics.var import RegimeStudentTLatentVAR
from phase2b_benchmark import reconstruct_paths
from research.market_selection import _bootstrap_difference
from research.phase3a_reconstruction import (
    RETURN_SELECTION_METRICS,
    _geometric_ratio,
    _score,
)
from sequences.baselines import Phase2ABaselines
from sequences.dataset import FactorStateSequenceBuilder
from sequences.evaluation import PathForecastEvaluator


BASELINE = "Student-t-VAR"
CANDIDATE = "Regime-Student-t-VAR"
STATE_METRICS = ("state_energy_score", "state_rmse")


def _pipeline(config):
    if config.data.source == "yfinance":
        return YFinanceDataPipeline(config)
    if config.data.source == "akshare":
        return DataPipeline(config)
    raise ValueError("Phase 4A requires yfinance or AKShare data.")


def _regime_spec(splits):
    """Map standardized latent states to average reconstructed log variance."""
    n_mean = splits.latent_metadata["n_mean_factors"]
    vol_loadings = splits.reconstructor.vol_loadings.loc[
        splits.reconstructor.stocks
    ].mean(axis=0).values
    loc = np.asarray(splits.latent_metadata["loc"], dtype=float)
    scale = np.asarray(splits.latent_metadata["scale"], dtype=float)
    weights = np.zeros_like(loc)
    weights[n_mean:n_mean + len(vol_loadings)] = (
        scale[n_mean:n_mean + len(vol_loadings)] * vol_loadings
    )
    intercept = float(
        loc[n_mean:n_mean + len(vol_loadings)] @ vol_loadings
        + splits.reconstructor.log_variance_mean.loc[
            splits.reconstructor.stocks
        ].mean()
    )
    signal = splits.train_states.values @ weights + intercept
    return signal, weights, intercept


def _accept_integration(market_reports):
    market_names = list(market_reports)
    state_ratios = {
        metric: float(
            np.exp(
                np.mean(
                    [
                        np.log(
                            market_reports[name]["state_ratios_vs_student_t"][
                                metric
                            ]
                        )
                        for name in market_names
                    ]
                )
            )
        )
        for metric in STATE_METRICS
    }
    pooled_state_composite = float(
        np.exp(np.mean(np.log(list(state_ratios.values()))))
    )
    market_state_composites = {
        name: market_reports[name]["state_composite_vs_student_t"]
        for name in market_names
    }
    energy_p_values = {
        name: market_reports[name]["paired_candidate_vs_student_t"][
            "state_energy_score"
        ]["one_sided_p_diffusion_not_better"]
        for name in market_names
    }
    return_composites = {
        name: market_reports[name]["return_composite_vs_student_t"]
        for name in market_names
    }
    checks = {
        "pooled_state_improvement_at_least_one_percent": (
            pooled_state_composite < 0.99
        ),
        "both_pooled_state_primaries_improve": all(
            value < 1.0 for value in state_ratios.values()
        ),
        "state_energy_improves_in_every_market": all(
            market_reports[name]["state_ratios_vs_student_t"][
                "state_energy_score"
            ]
            < 1.0
            for name in market_names
        ),
        "at_least_one_market_energy_p_below_0_10": any(
            value < 0.10 for value in energy_p_values.values()
        ),
        "no_market_state_composite_worse_by_one_percent": all(
            value <= 1.01 for value in market_state_composites.values()
        ),
        "no_market_return_composite_worse_by_two_percent": all(
            value <= 1.02 for value in return_composites.values()
        ),
    }
    return {
        "accepted_for_diffusion_integration": bool(all(checks.values())),
        "checks": checks,
        "pooled_state_composite": pooled_state_composite,
        "pooled_state_ratios": state_ratios,
        "market_state_composites": market_state_composites,
        "market_return_composites": return_composites,
        "market_energy_p_values": energy_p_values,
    }


def _run_market(name, config_path, args):
    config = load_config(config_path)
    if pd.to_datetime(config.data.end_date) >= pd.Timestamp("2024-01-02"):
        raise ValueError(
            f"{name} Phase 4A config must end before 2024."
        )
    returns, market = _pipeline(config).load_all_data(
        max_stocks=args.max_stocks
    )
    splits = FactorStateSequenceBuilder(
        config,
        context_length=config.temporal.context_length,
        horizon=config.temporal.horizon,
        evaluation_stride=config.temporal.horizon,
        train_end_date=args.train_end,
        validation_end_date=args.validation_end,
    ).build(returns, market)
    baselines = Phase2ABaselines(config, splits)
    evaluator = PathForecastEvaluator(
        splits.latent_metadata["n_mean_factors"]
    )
    signal, weights, intercept = _regime_spec(splits)
    model = RegimeStudentTLatentVAR(
        n_regimes=args.n_regimes,
        min_regime_obs=args.min_regime_obs,
    ).fit(
        splits.train_states.values,
        signal,
        weights,
        regime_intercept=intercept,
    )
    n_paths = args.paths or config.temporal.ensemble_paths
    return_scale = (
        splits.train_returns[baselines.stocks]
        .std(ddof=0)
        .clip(lower=1e-6)
        .values
    )
    scores = {BASELINE: [], CANDIDATE: []}
    origins = []
    by_regime = {
        regime: {BASELINE: [], CANDIDATE: []}
        for regime in range(args.n_regimes)
    }
    for origin in range(len(splits.test)):
        context = splits.test.context[origin]
        forecast = baselines.forecast(
            context,
            config.temporal.horizon,
            n_paths=n_paths,
            seed=1400000 + origin * n_paths,
        )
        candidate_states = np.stack(
            [
                model.simulate(
                    config.temporal.horizon,
                    seed=1500000 + origin * n_paths + path,
                    initial_state=context[-1],
                )
                for path in range(n_paths)
            ]
        )
        candidate_returns = reconstruct_paths(
            splits,
            candidate_states,
            forecast.innovations["VAR-GARCH"],
            baselines.stocks,
        )
        target_dates = splits.test.target_dates[origin]
        target_returns = splits.test_returns.reindex(target_dates)[
            baselines.stocks
        ].values
        target_states = splits.test.target[origin]
        baseline_score = _score(
            evaluator,
            forecast.states[BASELINE],
            forecast.returns[BASELINE],
            target_states,
            target_returns,
            return_scale,
        )
        candidate_score = _score(
            evaluator,
            candidate_states,
            candidate_returns,
            target_states,
            target_returns,
            return_scale,
        )
        scores[BASELINE].append(baseline_score)
        scores[CANDIDATE].append(candidate_score)
        regime = model.classify(context[-1])
        by_regime[regime][BASELINE].append(baseline_score)
        by_regime[regime][CANDIDATE].append(candidate_score)
        origins.append(
            {
                "context_end": str(
                    np.datetime_as_string(
                        splits.test.context_dates[origin, -1], unit="D"
                    )
                ),
                "target_start": str(
                    np.datetime_as_string(target_dates[0], unit="D")
                ),
                "target_end": str(
                    np.datetime_as_string(target_dates[-1], unit="D")
                ),
                "origin_regime": regime,
            }
        )

    metrics = evaluator.aggregate(scores)
    state_composite, state_ratios = _geometric_ratio(
        metrics[CANDIDATE], metrics[BASELINE], STATE_METRICS
    )
    return_composite, return_ratios = _geometric_ratio(
        metrics[CANDIDATE],
        metrics[BASELINE],
        RETURN_SELECTION_METRICS,
    )
    paired = {
        metric: _bootstrap_difference(
            [row[metric] for row in scores[CANDIDATE]],
            [row[metric] for row in scores[BASELINE]],
            seed=20261010 + index,
        )
        for index, metric in enumerate(scores[CANDIDATE][0])
    }
    regime_metrics = {}
    for regime, regime_scores in by_regime.items():
        if not regime_scores[CANDIDATE]:
            continue
        aggregated = evaluator.aggregate(regime_scores)
        regime_state_composite, regime_state_ratios = _geometric_ratio(
            aggregated[CANDIDATE],
            aggregated[BASELINE],
            STATE_METRICS,
        )
        regime_metrics[str(regime)] = {
            "n_origins": len(regime_scores[CANDIDATE]),
            "state_composite_vs_student_t": regime_state_composite,
            "state_ratios_vs_student_t": regime_state_ratios,
            "metrics": aggregated,
        }
    return {
        "config": config_path,
        "data_start": str(returns.index.min().date()),
        "data_end": str(returns.index.max().date()),
        "n_assets": len(baselines.stocks),
        "n_origins": len(splits.test),
        "n_paths": n_paths,
        "state_composite_vs_student_t": state_composite,
        "state_ratios_vs_student_t": state_ratios,
        "return_composite_vs_student_t": return_composite,
        "return_ratios_vs_student_t": return_ratios,
        "paired_candidate_vs_student_t": paired,
        "metrics": metrics,
        "regime_model": {
            "signal": (
                "training-fitted cross-sectional average reconstructed "
                "log variance"
            ),
            "thresholds": model.thresholds_.tolist(),
            "training_counts": model.regime_counts_.tolist(),
            "student_df_by_regime": model.regime_dfs_.tolist(),
            "spectral_radius": model.spectral_radius_,
        },
        "evaluation_by_origin_regime": regime_metrics,
        "origins": origins,
    }


def run(args):
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Phase 4A output already exists: {output}.")
    market_reports = {
        "sp500": _run_market("sp500", args.sp500_config, args),
        "csi300": _run_market("csi300", args.csi300_config, args),
    }
    decision = _accept_integration(market_reports)
    report = {
        "created_at": datetime.now().isoformat(),
        "status": (
            "regime_baseline_accepted_for_diffusion_integration"
            if decision["accepted_for_diffusion_integration"]
            else "regime_baseline_rejected"
        ),
        "protocol": {
            "phase": "4A",
            "purpose": (
                "pre-2024 multi-market regime-baseline development; "
                "not confirmation"
            ),
            "confirmation_data_loaded": False,
            "train_end": args.train_end,
            "validation_end": args.validation_end,
            "development_period": "2022 through 2023",
            "n_regimes": args.n_regimes,
            "regime_quantiles": [
                value / args.n_regimes
                for value in range(1, args.n_regimes)
            ],
            "min_regime_obs": args.min_regime_obs,
            "markets": list(market_reports),
        },
        "integration_decision": decision,
        "markets": market_reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Phase 4A pooled state composite="
        f"{decision['pooled_state_composite']:.6f}; "
        f"accepted={decision['accepted_for_diffusion_integration']}"
    )
    print(f"Saved {output}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sp500-config", default="research_sp500_selection.yaml"
    )
    parser.add_argument(
        "--csi300-config", default="research_csi300_selection.yaml"
    )
    parser.add_argument(
        "--output",
        default="research_output/phase4a_regime_var.json",
    )
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--validation-end", default="2021-12-31")
    parser.add_argument("--paths", type=int, default=None)
    parser.add_argument("--max-stocks", type=int, default=100)
    parser.add_argument("--n-regimes", type=int, default=3)
    parser.add_argument("--min-regime-obs", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

