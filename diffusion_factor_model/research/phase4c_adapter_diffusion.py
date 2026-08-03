"""Phase 4C: shared residual trunk with S&P and CSI adapter versions."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

from diffusion.shared_market import SharedMarketResidualDiffusion
from dynamics.var import LatentVAR
from phase2b_benchmark import reconstruct_paths
from research.market_selection import (
    _aggregate_seed_scores,
    _bootstrap_difference,
)
from research.phase2f_pooling import _linear_pool
from research.phase3a_reconstruction import (
    RETURN_SELECTION_METRICS,
    _geometric_ratio,
    _score,
)
from research.phase4b_shared_diffusion import (
    SEPARATE,
    STUDENT,
    STATE_METRICS,
    _build_market,
    _checkpoint_payload,
    _evaluation_rows,
    _integration_decision,
    _load_separate_models,
    _shared_training_data,
)


ADAPTER = "Phase4C-Market-Adapter-Diffusion-Pool"
FULLY_SHARED = "Phase4B-Fully-Shared-Diffusion-Pool"


def _restore_phase4b_models(markets, checkpoint_pattern, seeds):
    models = {}
    names = list(markets)
    state_dim = markets[names[0]]["splits"].train_states.shape[1]
    n_mean = markets[names[0]]["splits"].latent_metadata["n_mean_factors"]
    horizon = markets[names[0]]["config"].temporal.horizon
    for seed in seeds:
        checkpoint = Path(checkpoint_pattern.format(seed=seed))
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        model = SharedMarketResidualDiffusion(
            markets[names[0]]["config"],
            market_names=names,
            state_dim=state_dim,
            horizon=horizon,
            n_mean_factors=n_mean,
        )
        if payload["market_names"] != names:
            raise ValueError("Phase 4B checkpoint market order changed.")
        model.network.load_state_dict(payload["state_dict"])
        for market in names:
            parameters = payload["market_parameters"][market]
            var = LatentVAR()
            var.intercept_ = parameters["var_intercept"]
            var.transition_ = parameters["var_transition"]
            var.innovation_cov_ = parameters["var_innovation_cov"]
            var.last_state_ = markets[market][
                "splits"
            ].train_states.values[-1].copy()
            model.vars[market] = var
            model.residual_locations[market] = parameters[
                "residual_location"
            ]
            model.residual_scales[market] = parameters["residual_scale"]
        models[seed] = model
    return models


def _paired(first_seed_scores, second_scores, seed_offset):
    first = _aggregate_seed_scores(first_seed_scores)
    if isinstance(second_scores[0], list):
        second = _aggregate_seed_scores(second_scores)
    else:
        second = {
            metric: [row[metric] for row in second_scores]
            for metric in second_scores[0]
        }
    return {
        metric: _bootstrap_difference(
            first[metric],
            second[metric],
            seed=seed_offset + index,
        )
        for index, metric in enumerate(first)
    }


def _comparison(metrics, first, second, paired):
    state_composite, state_ratios = _geometric_ratio(
        metrics[first], metrics[second], STATE_METRICS
    )
    return_composite, return_ratios = _geometric_ratio(
        metrics[first], metrics[second], RETURN_SELECTION_METRICS
    )
    return {
        "state_composite": state_composite,
        "state_ratios": state_ratios,
        "return_composite": return_composite,
        "return_ratios": return_ratios,
        "paired": paired,
    }


def _score_market(
    name,
    values,
    rows,
    adapter_models,
    phase4b_models,
    separate_models,
    weight,
    n_paths,
):
    evaluator = values["evaluator"]
    splits = values["splits"]
    stocks = values["baselines"].stocks
    return_scale = values["return_scale"]
    student_scores = [
        _score(
            evaluator,
            row["student_states"],
            row["student_returns"],
            row["target_states"],
            row["target_returns"],
            return_scale,
        )
        for row in rows
    ]
    adapter_seed_scores = []
    shared_seed_scores = []
    separate_seed_scores = []
    for seed, adapter in adapter_models.items():
        adapter_rows = []
        shared_rows = []
        separate_rows = []
        phase4b = phase4b_models[seed]
        separate = separate_models[seed]
        for origin, row in enumerate(rows):
            adapter_diffusion = adapter.sample(
                name,
                row["context"],
                n_paths=n_paths,
                seed=seed * 10000 + 2100000 + origin * n_paths,
            )
            shared_diffusion = phase4b.sample(
                name,
                row["context"],
                n_paths=n_paths,
                seed=seed * 10000 + 1600000 + origin * n_paths,
            )
            separate_diffusion = separate.sample(
                row["context"],
                n_paths=n_paths,
                seed=seed * 10000 + 1700000 + origin * n_paths,
            )
            state_sets = {
                ADAPTER: _linear_pool(
                    adapter_diffusion, row["student_states"], weight
                ),
                FULLY_SHARED: _linear_pool(
                    shared_diffusion, row["student_states"], weight
                ),
                SEPARATE: _linear_pool(
                    separate_diffusion, row["student_states"], weight
                ),
            }
            score_sets = {}
            for method, states in state_sets.items():
                generated_returns = reconstruct_paths(
                    splits, states, row["innovations"], stocks
                )
                score_sets[method] = _score(
                    evaluator,
                    states,
                    generated_returns,
                    row["target_states"],
                    row["target_returns"],
                    return_scale,
                )
            adapter_rows.append(score_sets[ADAPTER])
            shared_rows.append(score_sets[FULLY_SHARED])
            separate_rows.append(score_sets[SEPARATE])
        adapter_seed_scores.append(adapter_rows)
        shared_seed_scores.append(shared_rows)
        separate_seed_scores.append(separate_rows)

    metrics = evaluator.aggregate(
        {
            ADAPTER: [
                row for seed_rows in adapter_seed_scores for row in seed_rows
            ],
            FULLY_SHARED: [
                row for seed_rows in shared_seed_scores for row in seed_rows
            ],
            SEPARATE: [
                row
                for seed_rows in separate_seed_scores
                for row in seed_rows
            ],
            STUDENT: student_scores,
        }
    )
    comparisons = {
        SEPARATE: _comparison(
            metrics,
            ADAPTER,
            SEPARATE,
            _paired(adapter_seed_scores, separate_seed_scores, 20261210),
        ),
        FULLY_SHARED: _comparison(
            metrics,
            ADAPTER,
            FULLY_SHARED,
            _paired(adapter_seed_scores, shared_seed_scores, 20261310),
        ),
        STUDENT: _comparison(
            metrics,
            ADAPTER,
            STUDENT,
            _paired(adapter_seed_scores, student_scores, 20261410),
        ),
    }
    return {
        "metrics": metrics,
        "comparisons": comparisons,
        "n_origins": len(rows),
    }


def _adapter_decision(market_reports):
    base = _integration_decision(market_reports)
    names = list(market_reports)
    versus_shared = {
        name: market_reports[name]["comparisons"][FULLY_SHARED]
        for name in names
    }
    pooled_shared_ratios = {
        metric: float(
            np.exp(
                np.mean(
                    [
                        np.log(versus_shared[name]["state_ratios"][metric])
                        for name in names
                    ]
                )
            )
        )
        for metric in STATE_METRICS
    }
    pooled_shared_composite = float(
        np.exp(np.mean(np.log(list(pooled_shared_ratios.values()))))
    )
    adapter_checks = {
        "adapters_improve_pooled_state_vs_fully_shared": (
            pooled_shared_composite < 1.0
        ),
        "sp500_adapter_protects_state_vs_fully_shared": (
            versus_shared["sp500"]["state_composite"] < 1.0
        ),
        "csi_adapter_not_worse_than_fully_shared_by_one_percent": (
            versus_shared["csi300"]["state_composite"] <= 1.01
        ),
        "no_adapter_return_composite_worse_by_two_percent": all(
            versus_shared[name]["return_composite"] <= 1.02
            for name in names
        ),
    }
    return {
        "accepted_for_third_market_freeze": bool(
            base["accepted_for_third_market_freeze"]
            and all(adapter_checks.values())
        ),
        "base_cross_market_gate": base,
        "adapter_checks": adapter_checks,
        "pooled_state_composite_vs_fully_shared": pooled_shared_composite,
        "pooled_state_ratios_vs_fully_shared": pooled_shared_ratios,
    }


def run(args):
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Phase 4C output already exists: {output}.")
    markets = {
        "sp500": _build_market(args.sp500_config, args),
        "csi300": _build_market(args.csi300_config, args),
    }
    seeds = args.seeds or markets["sp500"]["config"].temporal.validation_seeds
    training_data = _shared_training_data(markets)
    adapter_models = {}
    seed_reports = []
    checkpoint_dir = output.parent / "phase4c_checkpoints"
    for seed in seeds:
        print(f"Training Phase 4C adapter seed {seed}")
        model = SharedMarketResidualDiffusion(
            markets["sp500"]["config"],
            market_names=list(markets),
            state_dim=markets["sp500"]["splits"].train_states.shape[1],
            horizon=markets["sp500"]["config"].temporal.horizon,
            n_mean_factors=markets["sp500"]["splits"].latent_metadata[
                "n_mean_factors"
            ],
            adapter_rank=args.adapter_rank,
        )
        history = model.fit(
            training_data,
            seed=seed,
            training_steps=args.steps,
            selection_metric="sampled_path_energy",
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_dir / f"adapter_seed{seed}.pt"
        payload = _checkpoint_payload(model, seed)
        payload["adapter_rank"] = args.adapter_rank
        payload["market_adapter_versions"] = {
            market: index
            for market, index in model.market_index.items()
        }
        torch.save(payload, checkpoint)
        adapter_models[seed] = model
        seed_reports.append(
            {
                "seed": seed,
                "best_step": model.best_step_,
                "best_validation_loss": model.best_validation_loss_,
                "best_validation_state_energy": (
                    model.best_selection_score_
                ),
                "checkpoint": str(checkpoint),
                "training_history": history,
            }
        )

    phase4b_models = _restore_phase4b_models(
        markets, args.phase4b_checkpoint_pattern, seeds
    )
    separate_models = _load_separate_models(
        markets,
        {
            "sp500": args.sp500_separate_report,
            "csi300": args.csi300_separate_report,
        },
        seeds,
    )
    market_reports = {}
    origins = {}
    for index, (name, values) in enumerate(markets.items()):
        n_paths = args.paths or values["config"].temporal.ensemble_paths
        rows, market_origins = _evaluation_rows(
            values,
            n_paths,
            seed_offset=1800000 + index * 100000,
        )
        market_reports[name] = _score_market(
            name,
            values,
            rows,
            adapter_models,
            phase4b_models,
            separate_models[name],
            args.diffusion_weight,
            n_paths,
        )
        market_reports[name].update(
            {
                "n_assets": len(values["baselines"].stocks),
                "data_start": str(values["returns"].index.min().date()),
                "data_end": str(values["returns"].index.max().date()),
            }
        )
        origins[name] = market_origins

    decision = _adapter_decision(market_reports)
    report = {
        "created_at": datetime.now().isoformat(),
        "status": (
            "adapter_model_accepted_for_third_market_freeze"
            if decision["accepted_for_third_market_freeze"]
            else "adapter_model_rejected"
        ),
        "protocol": {
            "phase": "4C",
            "purpose": (
                "pre-2024 two-version market-adapter development; "
                "not confirmation"
            ),
            "confirmation_data_loaded": False,
            "shared_component": "conditional residual-path denoiser trunk",
            "market_specific_versions": {
                "sp500": "rank-constrained residual adapter 0",
                "csi300": "rank-constrained residual adapter 1",
            },
            "adapter_rank": args.adapter_rank,
            "adapter_initialization": (
                "zero output; exact fully shared model at initialization"
            ),
            "train_end": args.train_end,
            "validation_end": args.validation_end,
            "development_period": "2022 through 2023",
            "training_steps_requested": args.steps,
            "seeds": seeds,
            "diffusion_pool_weight": args.diffusion_weight,
            "weight_reselected": False,
        },
        "integration_decision": decision,
        "markets": market_reports,
        "seed_reports": seed_reports,
        "origins": origins,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Phase 4C adapter vs shared pooled state="
        f"{decision['pooled_state_composite_vs_fully_shared']:.6f}; "
        f"accepted={decision['accepted_for_third_market_freeze']}"
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
        "--sp500-separate-report",
        default="research_output/sp500/phase2f_development.json",
    )
    parser.add_argument(
        "--csi300-separate-report",
        default="research_output/csi300/market_selection_top100.json",
    )
    parser.add_argument(
        "--phase4b-checkpoint-pattern",
        default="research_output/phase4b_checkpoints/shared_seed{seed}.pt",
    )
    parser.add_argument(
        "--output",
        default="research_output/phase4c_adapter_diffusion.json",
    )
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--validation-end", default="2021-12-31")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--paths", type=int, default=None)
    parser.add_argument("--max-stocks", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--diffusion-weight", type=float, default=0.25)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

