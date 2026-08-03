"""Is the diffusion model earning its keep?

Protocol follows Chen, Xu, Xu & Zhang (2026), Section 6: compare the moments implied by
diffusion-generated data ("Diff") against the naive empirical estimate computed on exactly
the same training data ("Emp"), using their relative error definitions:

    RE3 = ||mu_hat  - mu_true||_2 / ||mu_true||_2        (eq. 32)
    RE4 = ||Sig_hat - Sig_true||_F / ||Sig_true||_F      (eq. 33)

The paper's ground truth is a known synthetic DGP. On real data there is none, so we use an
out-of-sample split: fit everything on the training window, and treat the held-out window's
sample moments as the target. Every method sees exactly the same training data.

Methods compared. All generative methods share the same factors, loadings, GARCH residuals
and reconstruction, so the ONLY thing that varies is how the latent is generated:

    Emp        no generative model at all -- just the training sample moments
    Bootstrap  resample historical latent rows with replacement
    Gauss      multivariate normal fitted to the latent (mean + full covariance)
    Diff       the diffusion model

If Diff cannot beat Gauss, the diffusion machinery is decoration on a linear factor model.
If Diff cannot beat Emp, the generative pipeline is not adding anything at all.
"""

import sys, os
sys.path.insert(0, os.getcwd())

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# The score net is tiny (128 wide, 10-dim input). With torch's default thread count the
# per-op coordination overhead dominates the actual math: 1545 ms/epoch at 16 threads vs
# 51 ms/epoch at 1 -- a 30x speedup from doing less parallelism, not more.
torch.set_num_threads(1)

from config.config import load_config
from data.loader import DataPipeline
from factors.extractor import FactorExtractor
from latent.parametrizer import LatentParametrizer
from diffusion.sde import VPSDE
from diffusion.score_net import ScoreNetwork
from diffusion.trainer import DiffusionTrainer
from residuals.garch import GARCHModeler
from sampling.sampler import DPMSolverPlusPlus
from reconstruction.reconstructor import ReturnReconstructor

N_GEN = 5000
SEEDS = [0, 1, 2]
TRAIN_FRAC = 0.70


def rel_mean_err(mu_hat, mu_true):
    return np.linalg.norm(mu_hat - mu_true) / np.linalg.norm(mu_true)


def rel_cov_err(cov_hat, cov_true):
    return np.linalg.norm(cov_hat - cov_true, "fro") / np.linalg.norm(cov_true, "fro")


def corr_dist(a, b):
    # Suspended stocks are filled with 0.0 by the loader, so a column can be constant over
    # a window and its correlation undefined. Compare only columns that vary in both.
    ok = (a.std(0) > 1e-12) & (b.std(0) > 1e-12)
    a, b = a[:, ok], b[:, ok]
    return np.linalg.norm(np.corrcoef(a, rowvar=False) - np.corrcoef(b, rowvar=False), "fro")


def build_fit(config, train_returns, train_market):
    """Everything that is fitted on the training window, shared by all methods."""
    fe = FactorExtractor(config)
    mfr = fe.extract_mean_factors(train_returns, train_market)
    resid = fe.compute_residuals(train_returns, mfr)
    vfr = fe.extract_volatility_factors(resid)

    lp = LatentParametrizer(config)
    latent, meta = lp.parametrize(mfr, vfr)

    idio = fe.compute_idiosyncratic(resid, vfr)
    gm = GARCHModeler(config)
    gm.fit_all(idio)

    rec = ReturnReconstructor(config, mfr, vfr, parametrizer=lp, latent_metadata=meta)
    return latent, meta, gm, rec


def reconstruct(rec, gm, latent_samples, seed):
    np.random.seed(seed)
    resid = gm.sample_all(len(latent_samples))
    return rec.reconstruct(latent_samples, resid)


def gen_diffusion(config, latent, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    sde = VPSDE(config.diffusion.beta_min, config.diffusion.beta_max, config.diffusion.T)
    net = ScoreNetwork(
        input_dim=latent.shape[1],
        hidden_dim=config.diffusion.hidden_dim,
        num_layers=config.diffusion.n_layers,
        dropout=config.diffusion.dropout,
        sde=sde,
    )
    trainer = DiffusionTrainer(config, sde, net)
    dl = DataLoader(
        TensorDataset(torch.from_numpy(latent).float()),
        batch_size=config.diffusion.batch_size,
        shuffle=True,
    )
    trainer.train(dl, num_epochs=config.diffusion.n_epochs)
    return DPMSolverPlusPlus(config, sde, net).sample(N_GEN, latent.shape[1], num_steps=20)


def gen_gaussian(latent, seed):
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(latent.mean(axis=0), np.cov(latent, rowvar=False), size=N_GEN)


def gen_bootstrap(latent, seed):
    rng = np.random.default_rng(seed)
    return latent[rng.integers(0, len(latent), size=N_GEN)]


def main():
    config = load_config("../prod_config.yaml")
    dp = DataPipeline(config)
    returns, market = dp.load_all_data(max_stocks=100)

    split = int(len(returns) * TRAIN_FRAC)
    train_r, test_r = returns.iloc[:split], returns.iloc[split:]
    train_m = market.iloc[:split]
    print(f"\nTrain: {train_r.shape}   Test: {test_r.shape}")

    latent, meta, gm, rec = build_fit(config, train_r, train_m)
    stocks = rec.stocks
    print(f"Latent rows N={len(latent)}, assets d={len(stocks)}, N/d={len(latent)/len(stocks):.2f}")

    train_x = train_r[stocks].values
    test_x = test_r[stocks].values

    # Targets: held-out sample moments (out-of-sample) and training moments (in-sample fidelity)
    targets = {
        "out-of-sample (held-out window)": (test_x.mean(0), np.cov(test_x, rowvar=False), test_x),
        "in-sample (training window)": (train_x.mean(0), np.cov(train_x, rowvar=False), train_x),
    }

    # --- noise floor: how far apart are two halves of the REAL data? ---
    half = len(train_x) // 2
    a, b = train_x[:half], train_x[half:]
    print("\n=== Noise floor (real data vs itself) ===")
    print(f"  two halves of train: RE4_cov={rel_cov_err(np.cov(a, rowvar=False), np.cov(b, rowvar=False)):.3f}"
          f"  corr_dist={corr_dist(a, b):.2f}")
    print(f"  train vs test      : RE4_cov={rel_cov_err(np.cov(train_x, rowvar=False), np.cov(test_x, rowvar=False)):.3f}"
          f"  corr_dist={corr_dist(train_x, test_x):.2f}")

    # --- generate ---
    samples = {"Emp": [train_x]}  # Emp = the training data itself, no generative model
    for name, fn in [
        ("Bootstrap", lambda s: gen_bootstrap(latent, s)),
        ("Gauss", lambda s: gen_gaussian(latent, s)),
        ("Diff", lambda s: gen_diffusion(config, latent, s)),
    ]:
        runs = []
        for seed in SEEDS:
            print(f"\n--- {name} seed={seed}")
            runs.append(reconstruct(rec, gm, fn(seed), seed)[stocks].values)
        samples[name] = runs

    # --- score ---
    results = {}
    for tname, (mu_t, cov_t, raw_t) in targets.items():
        print(f"\n{'=' * 70}\nTARGET: {tname}\n{'=' * 70}")
        print(f"{'method':<12}{'RE3 mean':>12}{'RE4 cov':>12}{'corr dist':>12}")
        results[tname] = {}
        for name, runs in samples.items():
            re3 = [rel_mean_err(r.mean(0), mu_t) for r in runs]
            re4 = [rel_cov_err(np.cov(r, rowvar=False), cov_t) for r in runs]
            cd = [corr_dist(r, raw_t) for r in runs]
            results[tname][name] = {
                "RE3": float(np.mean(re3)), "RE3_sd": float(np.std(re3)),
                "RE4": float(np.mean(re4)), "RE4_sd": float(np.std(re4)),
                "corr_dist": float(np.mean(cd)),
            }
            tag = "  (no model)" if name == "Emp" else ""
            print(f"{name:<12}{np.mean(re3):>12.3f}{np.mean(re4):>12.3f}{np.mean(cd):>12.2f}{tag}")

        emp = results[tname]["Emp"]
        if emp["RE4"] == 0:
            # In-sample, Emp *is* the target, so its error is 0 and the ratio is undefined.
            print("\n  (ratios vs Emp are undefined in-sample: Emp is the target)")
            continue
        print("\n  ratios vs Emp (below 1.0 = generative model helps):")
        for name in ("Bootstrap", "Gauss", "Diff"):
            r = results[tname][name]
            print(f"    {name:<10} RE3 {r['RE3']/emp['RE3']:.3f}   RE4 {r['RE4']/emp['RE4']:.3f}")

    out = Path("../prod_output/baselines.json")
    out.write_text(json.dumps({
        "n_latent": len(latent), "n_assets": len(stocks),
        "n_train_days": len(train_x), "n_test_days": len(test_x),
        "n_gen": N_GEN, "seeds": SEEDS, "results": results,
    }, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
