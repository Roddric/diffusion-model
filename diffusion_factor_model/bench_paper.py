"""Strategy A+D: the paper's factor-structured score network, against real baselines.

A: diffuse the FULL d-dimensional return vector with the factor structure baked into the
   score (Chen et al. eq. 16), instead of projecting to a 10-dim factor latent and
   reconstructing linearly (our earlier design, which the sweep showed adds nothing over a
   Gaussian -- because a 10-dim latent density essentially IS Gaussian).

D: benchmark against estimators a practitioner would actually use, not just the naive
   sample covariance:
     Emp    sample covariance                     (what the paper calls "Emp Method")
     LW     Ledoit-Wolf shrinkage                 (the real bar to clear)
     POET   factor covariance beta Sig_f beta' + diag(sigma^2)   (same structure the
            diffusion model is handed -- isolates what the DIFFUSION adds on top of it)
     Diff   the paper's diffusion factor model

Metrics:
    RE4  ||Sig_hat - Sig_test||_F / ||Sig_test||_F        (paper eq. 33)
    MV   realized out-of-sample volatility of the minimum-variance portfolio
         w = Sig_hat^-1 1 / (1' Sig_hat^-1 1). This is the decision-relevant test: a better
         covariance estimate produces a portfolio that is actually less volatile out of
         sample. Frobenius norms can flatter an estimator that gets the big eigenvalues
         right and the inverse wrong.
"""

import sys, os
sys.path.insert(0, os.getcwd())

import json
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

from sklearn.covariance import LedoitWolf

from config.config import load_config
from data.yf_loader import YFinanceDataPipeline
from diffusion.sde import VPSDE
from diffusion.factor_score_net import FactorScoreNetwork, estimate_factor_structure
from sampling.sampler import DPMSolverPlusPlus

TEST_DAYS = 500
TRAIN_N = [60, 90, 150, 300, 600, 1200]   # d = 87  -> N/d = 0.69 .. 13.8
K = 5
STEPS = 10000
BATCH = 64
N_GEN = 5000
SEEDS = [0, 1]
CKPT = Path("../prod_output/bench_paper.jsonl")


def re4(cov_hat, cov_true):
    return np.linalg.norm(cov_hat - cov_true, "fro") / np.linalg.norm(cov_true, "fro")


def min_var_vol(cov_hat, test_x):
    """Realized OOS volatility of the min-variance portfolio implied by cov_hat."""
    d = cov_hat.shape[0]
    # Ridge only to keep the inverse defined at N < d; identical for every estimator.
    inv = np.linalg.pinv(cov_hat + 1e-10 * np.eye(d))
    w = inv @ np.ones(d)
    w /= w.sum()
    return float((test_x @ w).std())


def poet_cov(X, k):
    beta, sigma2 = estimate_factor_structure(_z(X)[0], k)
    Xs, mu, sd = _z(X)
    f = Xs @ beta
    cov_s = beta @ np.cov(f, rowvar=False) @ beta.T + np.diag(sigma2)
    D = np.diag(sd)
    return D @ cov_s @ D


def _z(X):
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd, mu, sd


def train_diffusion(X, k, seed, sde):
    """Train the factor-structured score net on standardized returns."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xs, mu, sd = _z(X)
    beta, sigma2 = estimate_factor_structure(Xs, k)

    net = FactorScoreNetwork(beta, sigma2, sde, hidden=128, layers=3, dropout=0.0)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    T = torch.tensor(Xs, dtype=torch.float32)
    net.train()
    for _ in range(STEPS):
        x0 = T[torch.randint(0, len(T), (min(BATCH, len(T)),))]
        t = 1e-3 + torch.rand(len(x0)) * (sde.T - 1e-3)
        mean, std = sde.marginal_params(x0, t)
        z = torch.randn_like(x0)
        xt = mean + std * z
        # h-weighted score matching == eps-prediction: target stays O(1) for every t.
        loss = torch.mean((std * net(xt, t) + z) ** 2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sch.step()

    gen = DPMSolverPlusPlus(None, sde, net).sample(N_GEN, X.shape[1], num_steps=50)
    return gen * sd + mu, float(gen.std(axis=0).mean())   # unstandardize; convergence check


def run_one(returns, test_x, n, sde):
    train_x = returns[-n:]
    d = train_x.shape[1]
    cov_t = np.cov(test_x, rowvar=False)

    ests = {
        "Emp": np.cov(train_x, rowvar=False),
        "LW": LedoitWolf().fit(train_x).covariance_,
        "POET": poet_cov(train_x, K),
    }

    gen_std = []
    diff_covs = []
    for seed in SEEDS:
        g, gs = train_diffusion(train_x, K, seed, sde)
        diff_covs.append(np.cov(g, rowvar=False))
        gen_std.append(gs)
    ests["Diff"] = np.mean(diff_covs, axis=0)

    row = {"N": int(n), "d": int(d), "N_over_d": n / d,
           "gen_std": float(np.mean(gen_std)), "scores": {}}
    for name, cov in ests.items():
        row["scores"][name] = {"RE4": float(re4(cov, cov_t)),
                               "MV_vol": min_var_vol(cov, test_x)}

    print(f"\n>>> N={n} d={d}  N/d={n/d:.2f}   (gen std={np.mean(gen_std):.2f}, target 1.0)")
    print(f"    {'method':<8}{'RE4':>9}{'MV vol':>10}")
    for name, s in row["scores"].items():
        print(f"    {name:<8}{s['RE4']:>9.3f}{s['MV_vol']:>10.5f}")
    r = row["scores"]["Diff"]["RE4"] / row["scores"]["Emp"]["RE4"]
    rl = row["scores"]["Diff"]["RE4"] / row["scores"]["LW"]["RE4"]
    print(f"    Diff/Emp={r:.3f}   Diff/LW={rl:.3f}")
    return row


def main():
    config = load_config("../us_config.yaml")
    returns, _ = YFinanceDataPipeline(config).load_all_data()
    X = returns.values

    rng = np.random.default_rng(42)
    mask = np.zeros(len(X), bool)
    mask[rng.choice(len(X), size=TEST_DAYS, replace=False)] = True
    test_x, pool = X[mask], X[~mask]
    print(f"pool {pool.shape}  test {test_x.shape}")

    sde = VPSDE(0.1, 20.0, 1.0)
    done = set()
    if CKPT.exists():
        done = {json.loads(l)["N"] for l in CKPT.read_text().splitlines() if l.strip()}

    for n in TRAIN_N:
        if n in done:
            print(f"skip N={n}")
            continue
        row = run_one(pool, test_x, n, sde)
        with CKPT.open("a") as f:
            f.write(json.dumps(row) + "\n")

    rows = sorted([json.loads(l) for l in CKPT.read_text().splitlines() if l.strip()],
                  key=lambda r: r["N"])
    print("\n" + "=" * 78)
    print(f"{'N/d':>6}{'N':>6}{'Emp':>8}{'LW':>8}{'POET':>8}{'Diff':>8}"
          f"{'Diff/Emp':>10}{'Diff/LW':>9}{'genstd':>8}")
    print("=" * 78)
    for r in rows:
        s = r["scores"]
        print(f"{r['N_over_d']:>6.2f}{r['N']:>6}"
              f"{s['Emp']['RE4']:>8.3f}{s['LW']['RE4']:>8.3f}"
              f"{s['POET']['RE4']:>8.3f}{s['Diff']['RE4']:>8.3f}"
              f"{s['Diff']['RE4']/s['Emp']['RE4']:>10.3f}"
              f"{s['Diff']['RE4']/s['LW']['RE4']:>9.3f}{r['gen_std']:>8.2f}")
    print("\nMin-variance portfolio realized OOS vol (lower = better covariance):")
    print(f"{'N/d':>6}{'Emp':>10}{'LW':>10}{'POET':>10}{'Diff':>10}")
    for r in rows:
        s = r["scores"]
        print(f"{r['N_over_d']:>6.2f}{s['Emp']['MV_vol']:>10.5f}{s['LW']['MV_vol']:>10.5f}"
              f"{s['POET']['MV_vol']:>10.5f}{s['Diff']['MV_vol']:>10.5f}")


if __name__ == "__main__":
    main()
