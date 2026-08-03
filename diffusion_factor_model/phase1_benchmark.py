"""Strict temporal benchmark for the Phase 1 factor/diffusion integration.

Uses the same US data, final 500-day test block, and preceding 1,300-day training
window as the legacy temporal sweep. All transformations are fit only on training data.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf

torch.set_num_threads(1)

from config.config import load_config
from data.yf_loader import YFinanceDataPipeline
from diffusion.score_net import ScoreNetwork
from diffusion.sde import VPSDE
from dynamics.var import DynamicFactorVARGARCH
from evaluation.metrics import Evaluator
from sampling.sampler import DPMSolverPlusPlus


TRAIN_DAYS = 1300
TEST_DAYS = 500
SEED = 0


def relative_mean_error(estimate, target):
    return float(
        np.linalg.norm(estimate - target)
        / max(np.linalg.norm(target), 1e-12)
    )


def relative_covariance_error(estimate, target):
    return float(
        np.linalg.norm(estimate - target, "fro")
        / max(np.linalg.norm(target, "fro"), 1e-12)
    )


def correlation_distance(generated, target):
    varying = (generated.std(0) > 1e-12) & (target.std(0) > 1e-12)
    return float(
        np.linalg.norm(
            np.corrcoef(generated[:, varying], rowvar=False)
            - np.corrcoef(target[:, varying], rowvar=False),
            "fro",
        )
    )


def minimum_variance_volatility(covariance, test):
    d = covariance.shape[0]
    inverse = np.linalg.pinv(covariance + 1e-10 * np.eye(d))
    weights = inverse @ np.ones(d)
    weights /= weights.sum()
    return float((test @ weights).std())


def train_diffusion(config, latent, steps, n_gen, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    sde = VPSDE(
        config.diffusion.beta_min,
        config.diffusion.beta_max,
        config.diffusion.T,
    )
    network = ScoreNetwork(
        input_dim=latent.shape[1],
        hidden_dim=config.diffusion.hidden_dim,
        num_layers=config.diffusion.n_layers,
        dropout=config.diffusion.dropout,
        sde=sde,
    )
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=config.diffusion.learning_rate,
        weight_decay=config.diffusion.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps
    )
    tensor = torch.from_numpy(latent).float()
    batch_size = min(config.diffusion.batch_size, len(tensor))
    network.train()
    for _ in range(steps):
        x0 = tensor[torch.randint(0, len(tensor), (batch_size,))]
        t = 1e-3 + torch.rand(batch_size) * (sde.T - 1e-3)
        mean, std = sde.marginal_params(x0, t)
        noise = torch.randn_like(x0)
        loss = torch.mean(
            (network.eps(mean + std * noise, t) - noise) ** 2
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    generated = DPMSolverPlusPlus(None, sde, network).sample(
        n_gen, latent.shape[1], num_steps=config.sampling.n_steps
    )
    return generated, float(generated.std(axis=0).mean())


def load_legacy_reference(path):
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("split") == "temporal" and row.get("train_days") == TRAIN_DAYS:
            return row
    return None


def score_sample(values, test_values, stocks):
    covariance = np.cov(values, rowvar=False)
    target_covariance = np.cov(test_values, rowvar=False)
    real = pd.DataFrame(test_values, columns=stocks)
    generated = pd.DataFrame(values, columns=stocks)
    evaluator = Evaluator(None)
    real_acf = evaluator.volatility_acf(real)
    generated_acf = evaluator.volatility_acf(generated)
    return {
        "RE3": relative_mean_error(values.mean(0), test_values.mean(0)),
        "RE4": relative_covariance_error(covariance, target_covariance),
        "correlation_distance": correlation_distance(values, test_values),
        "min_variance_vol": minimum_variance_volatility(
            covariance, test_values
        ),
        "volatility_acf_mae": float(
            np.nanmean(np.abs(real_acf - generated_acf))
        ),
        "tail_dependence_error": float(
            abs(
                evaluator.tail_dependence(real)
                - evaluator.tail_dependence(generated)
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--n-gen", type=int, default=2000)
    parser.add_argument(
        "--output", default="../prod_output/phase1_benchmark.json"
    )
    args = parser.parse_args()

    config = load_config("../us_config.yaml")
    returns, market = YFinanceDataPipeline(config).load_all_data()
    test_returns = returns.iloc[-TEST_DAYS:]
    pool_returns = returns.iloc[:-TEST_DAYS]
    pool_market = market.iloc[:-TEST_DAYS]
    train_returns = pool_returns.iloc[-TRAIN_DAYS:]
    train_market = pool_market.iloc[-TRAIN_DAYS:]

    model = DynamicFactorVARGARCH(config).fit(train_returns, train_market)
    stocks = model.reconstructor.stocks
    train_values = train_returns[stocks].values
    test_values = test_returns[stocks].values
    latent = model.latent_

    np.random.seed(SEED)
    innovations = model.garch.sample_all(args.n_gen)
    rng = np.random.default_rng(SEED)
    latent_samples = {
        "Bootstrap": latent[
            rng.integers(0, len(latent), size=args.n_gen)
        ],
        "Gaussian": rng.multivariate_normal(
            latent.mean(0), np.cov(latent, rowvar=False), size=args.n_gen
        ),
        "VAR-GARCH": model.var.simulate(args.n_gen, seed=SEED),
    }
    diffusion, latent_std = train_diffusion(
        config, latent, args.steps, args.n_gen, SEED
    )
    latent_samples["Diffusion"] = diffusion

    target_covariance = np.cov(test_values, rowvar=False)
    scores = {
        "Empirical": score_sample(train_values, test_values, stocks),
    }
    lw_covariance = LedoitWolf().fit(train_values).covariance_
    scores["Ledoit-Wolf"] = {
        "RE4": relative_covariance_error(lw_covariance, target_covariance),
        "min_variance_vol": minimum_variance_volatility(
            lw_covariance, test_values
        ),
    }
    for name, states in latent_samples.items():
        generated = model.reconstructor.reconstruct(states, innovations)
        scores[name] = score_sample(
            generated[stocks].values, test_values, stocks
        )
    scores["Diffusion"]["latent_std"] = latent_std

    legacy_path = Path("../prod_output/sweep_us.jsonl")
    output = {
        "timestamp": datetime.now().isoformat(),
        "protocol": {
            "split": "strict_temporal",
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "train_start": str(train_returns.index.min().date()),
            "train_end": str(train_returns.index.max().date()),
            "test_start": str(test_returns.index.min().date()),
            "test_end": str(test_returns.index.max().date()),
            "n_assets": len(stocks),
            "n_latent": len(latent),
            "n_generated": args.n_gen,
            "diffusion_steps": args.steps,
            "seed": SEED,
            "train_only_transformations": True,
        },
        "scores": scores,
        "legacy_reference": load_legacy_reference(legacy_path),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\nPHASE 1 TEMPORAL BENCHMARK")
    print(f"{'method':<14}{'RE4':>10}{'MV vol':>12}{'ACF MAE':>11}")
    for name, values in scores.items():
        print(
            f"{name:<14}{values['RE4']:>10.3f}"
            f"{values['min_variance_vol']:>12.5f}"
            f"{values.get('volatility_acf_mae', float('nan')):>11.3f}"
        )
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
