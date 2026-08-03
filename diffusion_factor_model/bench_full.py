"""Full evaluation: does the diffusion factor model beat structured baselines on things a
covariance matrix CANNOT capture -- tails, higher moments -- as well as on portfolios?

RE4/min-variance are pure second-moment tests, and a factor covariance is already optimal
for second moments by construction, so they cannot reveal what a generative model adds. Here
we grade every method on the FULL distribution it implies.

Methods (all produce SAMPLES, so tail metrics are well defined):
    Boot       resample historical training days. The empirical distribution: fat tails and
               true tail dependence for free. The honest bar for any tail claim.
    G-Emp      N(mu, Sigma_sample)
    G-LW       N(mu, Sigma_LedoitWolf)
    G-POET     N(mu, beta Sig_f beta' + diag(sigma^2))   -- the factor covariance, Gaussian
    Diff       the paper's factor-structured diffusion model

The Gaussian rows are there to prove the point: they have zero excess kurtosis and near-zero
tail dependence by construction. If Diff cannot beat THEM on tails, it has learned nothing
non-Gaussian. If it cannot beat Boot, it has learned nothing the raw data didn't already say.

Metrics
    RE4         ||Sig_hat - Sig_test||_F / ||Sig_test||_F
    MinVar vol  realized OOS vol of w = Sig^-1 1 / (1'Sig^-1 1)         (lower better)
    MV Sharpe   realized OOS Sharpe of w propto Sig^-1 mu, annualized   (higher better)
    VaR cover   realized exceedance rate of the model's 1%/5% VaR on held-out days
                (target 1% / 5%; >target = model underestimates risk)
    ES ratio    realized expected shortfall / model's predicted ES at 5% (1.0 = calibrated)
    Kurt        excess kurtosis of the equal-weight portfolio's returns
    TailDep     mean lower 5% tail dependence over asset pairs
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

from scipy import stats
from sklearn.covariance import LedoitWolf

from config.config import load_config
from data.yf_loader import YFinanceDataPipeline
from diffusion.sde import VPSDE
from diffusion.factor_score_net import estimate_factor_structure
from bench_paper import train_diffusion, poet_cov, re4, min_var_vol, _z, K, N_GEN

TEST_DAYS = 500
TRAIN_N = [60, 150, 300, 600]
SEEDS = [0, 1]
CKPT = Path("../prod_output/bench_full.jsonl")
ANN = np.sqrt(252)


def mv_sharpe(mu_hat, cov_hat, test_x):
    """Realized OOS Sharpe of the mean-variance portfolio implied by (mu_hat, cov_hat)."""
    d = len(mu_hat)
    w = np.linalg.pinv(cov_hat + 1e-10 * np.eye(d)) @ mu_hat
    n = np.abs(w).sum()
    if n < 1e-12:
        return 0.0
    w = w / n                      # unit gross exposure, so Sharpe is scale-free
    r = test_x @ w
    return float(r.mean() / (r.std() + 1e-12) * ANN)


def var_metrics(samples, test_x):
    """VaR/ES calibration of the equal-weight portfolio."""
    d = samples.shape[1]
    w = np.ones(d) / d
    sim = samples @ w
    real = test_x @ w

    out = {}
    for q in (0.01, 0.05):
        var = np.quantile(sim, q)                       # model's VaR (a negative number)
        out[f"var{int(q*100)}_cover"] = float((real < var).mean())
    q = 0.05
    var5 = np.quantile(sim, q)
    pred_es = sim[sim <= var5].mean()
    tail = real[real <= var5]
    real_es = tail.mean() if len(tail) else var5
    out["es5_ratio"] = float(real_es / pred_es) if pred_es < 0 else float("nan")
    out["port_kurt"] = float(stats.kurtosis(sim))
    return out


def tail_dep(X, q=0.05, max_assets=30):
    X = X[:, :max_assets]
    thr = np.quantile(X, q, axis=0)
    below = X < thr
    n = X.shape[1]
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            mi = below[:, i].sum()
            if mi:
                vals.append((below[:, i] & below[:, j]).sum() / mi)
    return float(np.mean(vals))


def run_one(pool, test_x, n, sde):
    train_x = pool[-n:]
    d = train_x.shape[1]
    cov_t = np.cov(test_x, rowvar=False)
    mu_t = test_x.mean(0)
    mu_hat = train_x.mean(0)

    covs = {
        "G-Emp": np.cov(train_x, rowvar=False),
        "G-LW": LedoitWolf().fit(train_x).covariance_,
        "G-POET": poet_cov(train_x, K),
    }

    rng = np.random.default_rng(0)
    samples = {"Boot": train_x[rng.integers(0, n, size=N_GEN)]}
    for name, c in covs.items():
        samples[name] = rng.multivariate_normal(mu_hat, c, size=N_GEN)

    diff_s, gstd = [], []
    for seed in SEEDS:
        g, gs = train_diffusion(train_x, K, seed, sde)
        diff_s.append(g)
        gstd.append(gs)
    samples["Diff"] = np.vstack(diff_s)

    row = {"N": int(n), "d": int(d), "N_over_d": n / d,
           "gen_std": float(np.mean(gstd)), "scores": {}}

    for name, S in samples.items():
        cov = np.cov(S, rowvar=False)
        mu = S.mean(0)
        m = {
            "RE4": float(re4(cov, cov_t)),
            "minvar_vol": min_var_vol(cov, test_x),
            "mv_sharpe": mv_sharpe(mu_hat, cov, test_x),   # mu from data for all, cov varies
            "taildep": tail_dep(S),
        }
        m.update(var_metrics(S, test_x))
        row["scores"][name] = m

    # what the real held-out data actually looks like
    row["realized"] = {"taildep": tail_dep(test_x),
                       "port_kurt": float(stats.kurtosis(test_x @ (np.ones(d) / d)))}

    print(f"\n>>> N={n} N/d={n/d:.2f}  (gen std={np.mean(gstd):.2f})")
    hdr = f"{'method':<8}{'RE4':>8}{'MinVarVol':>11}{'Sharpe':>8}{'VaR1%':>7}{'VaR5%':>7}{'ES5':>7}{'Kurt':>7}{'TailDep':>9}"
    print(hdr)
    for name, s in row["scores"].items():
        print(f"{name:<8}{s['RE4']:>8.3f}{s['minvar_vol']:>11.5f}{s['mv_sharpe']:>8.2f}"
              f"{s['var1_cover']*100:>6.1f}%{s['var5_cover']*100:>6.1f}%"
              f"{s['es5_ratio']:>7.2f}{s['port_kurt']:>7.2f}{s['taildep']:>9.3f}")
    print(f"{'REALIZED':<8}{'-':>8}{'-':>11}{'-':>8}{'  1.0%':>7}{'  5.0%':>7}{'1.00':>7}"
          f"{row['realized']['port_kurt']:>7.2f}{row['realized']['taildep']:>9.3f}")
    return row


def main():
    config = load_config("../us_config.yaml")
    returns, _ = YFinanceDataPipeline(config).load_all_data()
    X = returns.values

    rng = np.random.default_rng(42)
    mask = np.zeros(len(X), bool)
    mask[rng.choice(len(X), size=TEST_DAYS, replace=False)] = True
    test_x, pool = X[mask], X[~mask]

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

    print("\n" + "=" * 80)
    print("SUMMARY (averaged over N)")
    rows = [json.loads(l) for l in CKPT.read_text().splitlines() if l.strip()]
    methods = ["Boot", "G-Emp", "G-LW", "G-POET", "Diff"]
    print(f"{'method':<8}{'RE4':>8}{'MinVarVol':>11}{'Sharpe':>8}{'VaR1%':>8}{'VaR5%':>8}{'Kurt':>8}{'TailDep':>9}")
    for m in methods:
        g = lambda key: np.mean([r["scores"][m][key] for r in rows])
        print(f"{m:<8}{g('RE4'):>8.3f}{g('minvar_vol'):>11.5f}{g('mv_sharpe'):>8.2f}"
              f"{g('var1_cover')*100:>7.1f}%{g('var5_cover')*100:>7.1f}%"
              f"{g('port_kurt'):>8.2f}{g('taildep'):>9.3f}")
    rk = np.mean([r["realized"]["port_kurt"] for r in rows])
    rt = np.mean([r["realized"]["taildep"] for r in rows])
    print(f"{'REALIZED':<8}{'-':>8}{'-':>11}{'-':>8}{'1.0%':>8}{'5.0%':>8}{rk:>8.2f}{rt:>9.3f}")


if __name__ == "__main__":
    main()
