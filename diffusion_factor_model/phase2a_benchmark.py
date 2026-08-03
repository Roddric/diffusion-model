"""Rolling-origin benchmark for the Phase 2A sequence forecasting protocol."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

from config.config import load_config
from data.yf_loader import YFinanceDataPipeline
from sequences.baselines import Phase2ABaselines
from sequences.dataset import FactorStateSequenceBuilder
from sequences.evaluation import PathForecastEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--paths", type=int, default=20)
    parser.add_argument("--max-origins", type=int, default=None)
    parser.add_argument(
        "--output", default="../prod_output/phase2a_benchmark.json"
    )
    args = parser.parse_args()

    config = load_config("../us_config.yaml")
    returns, market = YFinanceDataPipeline(config).load_all_data()
    builder = FactorStateSequenceBuilder(
        config,
        context_length=args.context,
        horizon=args.horizon,
        evaluation_stride=args.horizon,
    )
    splits = builder.build(returns, market)
    baselines = Phase2ABaselines(config, splits)
    evaluator = PathForecastEvaluator(
        splits.latent_metadata["n_mean_factors"]
    )

    n_origins = len(splits.test)
    if args.max_origins is not None:
        n_origins = min(n_origins, args.max_origins)
    if n_origins == 0:
        raise RuntimeError("The test split contains no complete forecast windows.")

    scores = {method: [] for method in baselines.METHODS}
    origins = []
    for origin in range(n_origins):
        forecast = baselines.forecast(
            splits.test.context[origin],
            args.horizon,
            n_paths=args.paths,
            seed=1000 + origin * args.paths,
        )
        target_dates = splits.test.target_dates[origin]
        target_returns = (
            splits.test_returns.reindex(target_dates)[baselines.stocks].values
        )
        target_states = splits.test.target[origin]
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
            }
        )
        for method in baselines.METHODS:
            scores[method].append(
                evaluator.score(
                    forecast.states[method],
                    forecast.returns[method],
                    target_states,
                    target_returns,
                )
            )
        print(
            f"origin {origin + 1}/{n_origins}: "
            f"{origins[-1]['target_start']} -> {origins[-1]['target_end']}"
        )

    report = {
        "timestamp": datetime.now().isoformat(),
        "protocol": {
            "context_length": args.context,
            "horizon": args.horizon,
            "ensemble_paths": args.paths,
            "evaluation_stride": args.horizon,
            "n_origins": n_origins,
            "n_assets": len(baselines.stocks),
            "state_dim": splits.train_states.shape[1],
            "train_start": str(splits.train_returns.index.min().date()),
            "train_end": str(splits.train_returns.index.max().date()),
            "validation_start": str(
                splits.validation_returns.index.min().date()
            ),
            "validation_end": str(
                splits.validation_returns.index.max().date()
            ),
            "test_start": str(splits.test_returns.index.min().date()),
            "test_end": str(splits.test_returns.index.max().date()),
            "split_contained_windows": True,
            "train_only_transformations": True,
        },
        "metrics": evaluator.aggregate(scores),
        "origins": origins,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nPHASE 2A ROLLING PATH BENCHMARK")
    print(
        f"{'method':<18}{'energy':>10}{'state RMSE':>13}"
        f"{'vol MAE':>11}{'drawdown':>11}"
    )
    for method, metrics in report["metrics"].items():
        print(
            f"{method:<18}"
            f"{metrics['state_energy_score']['mean']:>10.3f}"
            f"{metrics['state_rmse']['mean']:>13.3f}"
            f"{metrics['daily_volatility_mae']['mean']:>11.5f}"
            f"{metrics['max_drawdown_error']['mean']:>11.4f}"
        )
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
