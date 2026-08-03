"""Does the diffusion model beat the empirical estimator where the theory says it should?

Chen et al. (2026) claim the diffusion factor model beats the naive empirical estimator only
in the small-data regime N <= d (their Table 2: the advantage vanishes once N is a few times
d). Our A-share test sat at N/d = 8.8 -- far outside that regime -- so "Diff loses to Emp"
there confirmed the theory rather than testing it.

Here we sweep N/d from below 1 up to ~25 on US large caps (2010-2024, so N can actually
vary), holding the held-out test set and the asset universe fixed. If the paper is right,
the Diff/Emp covariance-error ratio should sit BELOW 1 at small N and rise above 1 as N grows.

Two splits:
  random   -- interleaved held-out days. Train and test estimate the SAME population, so the
              error measures estimation quality. This is the honest analogue of the paper's
              i.i.d. setup, and of what this model assumes (it samples days i.i.d.).
  temporal -- held-out final block. Realistic, but train/test span different regimes, so the
              error is dominated by non-stationarity rather than by estimator quality.

Training budget is fixed in GRADIENT STEPS, not epochs, so larger N does not silently buy
more optimization. Results are checkpointed per (split, window) and the run resumes.
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

from torch.utils.data import DataLoader, TensorDataset

from config.config import load_config
from data.yf_loader import YFinanceDataPipeline
from baselines import gen_gaussian, gen_bootstrap, rel_cov_err, rel_mean_err
from factors.extractor import FactorExtractor
from latent.parametrizer import LatentParametrizer
from reconstruction.reconstructor import ReturnReconstructor
from diffusion.sde import VPSDE
from diffusion.score_net import ScoreNetwork
from diffusion.trainer import DiffusionTrainer
from sampling.sampler import DPMSolverPlusPlus

TEST_DAYS = 500
TRAIN_DAYS = [320, 550, 1300, 2300]   # -> latent N ~ 69 .. 2048, i.e. N/d 0.8 .. 23
TARGET_STEPS = 10000
BATCH = 64
N_GEN = 5000
SEEDS = [0]
SPLITS = ["random", "temporal"]
CKPT = Path("../prod_output/sweep_us.jsonl")


def build_fit_fast(config, train_returns, train_market):
    """Same pipeline as baselines.build_fit, but idiosyncratic residuals are drawn by
    resampling historical idio rows instead of refitting ~90 GARCH models per window (which
    costs ~7 min and would dominate the sweep). Every generative method shares the identical
    residual draws, so the comparison BETWEEN latent generators is unaffected."""
    fe = FactorExtractor(config)
    mfr = fe.extract_mean_factors(train_returns, train_market)
    resid = fe.compute_residuals(train_returns, mfr)
    vfr = fe.extract_volatility_factors(resid)
    lp = LatentParametrizer(config)
    latent, meta = lp.parametrize(mfr, vfr)
    idio = fe.compute_idiosyncratic(resid, vfr)
    rec = ReturnReconstructor(config, mfr, vfr, parametrizer=lp, latent_metadata=meta)
    return latent, idio, rec


def draw_resid(idio, stocks, n, seed):
    rng = np.random.default_rng(1000 + seed)
    return idio[stocks].iloc[rng.integers(0, len(idio), size=n)].reset_index(drop=True)


def gen_diffusion(config, latent, seed):
    """Train to convergence and sample. Uses the same objective as DiffusionTrainer, but
    steps on random minibatches drawn directly from a tensor: at small N the DataLoader and
    tqdm overhead dominate (38 ms/step vs ~2 ms), and an undertrained score net produces a
    wildly over-dispersed latent, which would show up as a bad RE4 that says nothing about
    the method."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    sde = VPSDE(config.diffusion.beta_min, config.diffusion.beta_max, config.diffusion.T)
    net = ScoreNetwork(
        input_dim=latent.shape[1], hidden_dim=config.diffusion.hidden_dim,
        num_layers=config.diffusion.n_layers, dropout=config.diffusion.dropout, sde=sde,
    )
    X = torch.from_numpy(latent).float()
    opt = torch.optim.AdamW(net.parameters(), lr=config.diffusion.learning_rate,
                            weight_decay=config.diffusion.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TARGET_STEPS)
    net.train()
    n, eps_t = len(X), 1e-3
    for _ in range(TARGET_STEPS):
        x0 = X[torch.randint(0, n, (min(BATCH, n),))]
        t = eps_t + torch.rand(len(x0)) * (sde.T - eps_t)
        mean, std = sde.marginal_params(x0, t)
        eps = torch.randn_like(x0)
        loss = torch.mean((net.eps(mean + std * eps, t) - eps) ** 2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()

    return DPMSolverPlusPlus(config, sde, net).sample(N_GEN, latent.shape[1], num_steps=20)


def split_data(returns, market, mode):
    if mode == "temporal":
        return returns.iloc[-TEST_DAYS:], returns.iloc[:-TEST_DAYS], market.iloc[:-TEST_DAYS]
    rng = np.random.default_rng(42)
    mask = np.zeros(len(returns), bool)
    mask[rng.choice(len(returns), size=TEST_DAYS, replace=False)] = True
    return returns[mask], returns[~mask], market[~mask]


def run_one(config, returns, market, mode, n_days):
    test_r, pool, pool_m = split_data(returns, market, mode)
    if n_days > len(pool):
        return None

    train_r, train_m = pool.iloc[-n_days:], pool_m.iloc[-n_days:]
    latent, idio, rec = build_fit_fast(config, train_r, train_m)
    stocks = [s for s in rec.stocks if s in idio.columns]
    N, d = len(latent), len(stocks)

    test_x = test_r[stocks].values
    train_x = train_r[stocks].values
    mu_t, cov_t = test_x.mean(0), np.cov(test_x, rowvar=False)

    scores = {"Emp": {"RE3": rel_mean_err(train_x.mean(0), mu_t),
                      "RE4": rel_cov_err(np.cov(train_x, rowvar=False), cov_t)}}

    for name in ("Bootstrap", "Gauss", "Diff"):
        re3s, re4s, lat_std = [], [], []
        for seed in SEEDS:
            if name == "Diff":
                lat = gen_diffusion(config, latent, seed)
                lat_std.append(float(lat.std(axis=0).mean()))
            elif name == "Gauss":
                lat = gen_gaussian(latent, seed)
            else:
                lat = gen_bootstrap(latent, seed)
            g = rec.reconstruct(lat, draw_resid(idio, stocks, len(lat), seed))[stocks].values
            re3s.append(rel_mean_err(g.mean(0), mu_t))
            re4s.append(rel_cov_err(np.cov(g, rowvar=False), cov_t))
        scores[name] = {"RE3": float(np.mean(re3s)), "RE4": float(np.mean(re4s)),
                        "RE4_sd": float(np.std(re4s))}
        if lat_std:
            # 1.0 = the sampler reproduced the standardized latent's scale (converged).
            scores[name]["latent_std"] = float(np.mean(lat_std))

    ratio = scores["Diff"]["RE4"] / scores["Emp"]["RE4"]
    print(f"\n>>> split={mode} days={n_days}  N={N} d={d}  N/d={N/d:.2f}")
    for k, v in scores.items():
        print(f"    {k:<10}{v['RE4']:>10.3f}")
    ls = scores["Diff"].get("latent_std", float("nan"))
    flag = "" if abs(ls - 1) < 0.35 else "   <-- NOT CONVERGED (latent std should be ~1.0)"
    print(f"    diffusion latent std = {ls:.2f}{flag}")
    print(f"    Diff/Emp = {ratio:.3f}  ({'Diff WINS' if ratio < 1 else 'Emp wins'})")

    return {"split": mode, "train_days": n_days, "N": N, "d": d, "N_over_d": N / d,
            "scores": scores, "diff_over_emp_RE4": ratio}


def main():
    config = load_config("../us_config.yaml")
    returns, market = YFinanceDataPipeline(config).load_all_data()

    done = set()
    if CKPT.exists():
        done = {(json.loads(l)["split"], json.loads(l)["train_days"])
                for l in CKPT.read_text().splitlines() if l.strip()}

    for mode in SPLITS:
        for n_days in TRAIN_DAYS:
            if (mode, n_days) in done:
                print(f"skip {mode}/{n_days} (done)")
                continue
            row = run_one(config, returns, market, mode, n_days)
            if row:
                with CKPT.open("a") as f:
                    f.write(json.dumps(row) + "\n")

    rows = [json.loads(l) for l in CKPT.read_text().splitlines() if l.strip()]
    for mode in SPLITS:
        rs = sorted([r for r in rows if r["split"] == mode], key=lambda r: r["N_over_d"])
        if not rs:
            continue
        print(f"\n{'=' * 68}\nSPLIT = {mode}")
        print(f"{'N/d':>7}{'N':>7}{'d':>5}{'Emp':>9}{'Boot':>9}{'Gauss':>9}{'Diff':>9}{'Diff/Emp':>10}")
        for r in rs:
            s = r["scores"]
            print(f"{r['N_over_d']:>7.2f}{r['N']:>7}{r['d']:>5}"
                  f"{s['Emp']['RE4']:>9.3f}{s['Bootstrap']['RE4']:>9.3f}"
                  f"{s['Gauss']['RE4']:>9.3f}{s['Diff']['RE4']:>9.3f}"
                  f"{r['diff_over_emp_RE4']:>10.3f}")
    print("\nPaper's claim: Diff/Emp < 1 when N <= d, rising toward 1.0 as N grows.")


if __name__ == "__main__":
    main()
