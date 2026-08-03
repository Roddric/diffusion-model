"""Relax the paper's Gaussian idiosyncratic assumption (their Assumption 1(iii)).

The score decomposition is exact only for eps ~ N(0, diag(sigma^2)) -- that Gaussianity is
what makes the complement score closed-form. But it also caps the model's tails: our full
benchmark reached excess kurtosis 1.74 against a realized 8.45, and the gap lives largely in
the idiosyncratic part.

The model is r = beta F + eps. The diffusion learns the FACTOR density p_fac nonparametrically
(that is what xi is for) and we keep it. We only replace the Gaussian eps with a per-asset
Student-t fitted to the training idiosyncratic residuals:

    r_gen = beta (beta^T r_diff)  +  eps_t,     eps_t ~ t(nu_i) * scale_i

The systematic part is untouched, so this cannot damage the covariance structure that the
factor architecture buys us -- it only fattens the idiosyncratic tails.

Compared here: Boot, G-POET (best Gaussian), Diff (paper, Gaussian eps), Diff-t (this).
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

from config.config import load_config
from data.yf_loader import YFinanceDataPipeline
from diffusion.sde import VPSDE
from diffusion.score_net import SinusoidalPositionalEmbedding  # noqa: F401
from diffusion.factor_score_net import FactorScoreNetwork, estimate_factor_structure
from sampling.sampler import DPMSolverPlusPlus
from bench_paper import poet_cov, re4, min_var_vol, _z, K, N_GEN, STEPS, BATCH
from bench_full import var_metrics, tail_dep, mv_sharpe

TEST_DAYS = 500
TRAIN_N = [60, 150, 300, 600]
SEEDS = [0, 1]
CKPT = Path("../prod_output/bench_t.jsonl")


def fit_t_idio(Xs, beta):
    """Per-asset Student-t fit to the idiosyncratic residual (I - beta beta^T) Xs."""
    resid = Xs - (Xs @ beta) @ beta.T
    params = []
    for i in range(resid.shape[1]):
        nu, loc, scale = stats.t.fit(resid[:, i], floc=0.0)
        params.append((max(nu, 2.5), scale))   # nu > 2 so the variance exists
    return params


def sample_t_idio(params, n, rng):
    cols = [stats.t.rvs(nu, loc=0.0, scale=sc, size=n, random_state=rng)
            for nu, sc in params]
    return np.column_stack(cols)


def train_and_generate(X, k, seed, sde):
    """Train the factor score net; return BOTH the Gaussian-eps samples (paper) and the
    Student-t-eps samples, from the same trained model."""
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
        loss = torch.mean((std * net(mean + std * z, t) + z) ** 2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sch.step()

    gen_s = DPMSolverPlusPlus(None, sde, net).sample(N_GEN, X.shape[1], num_steps=50)

    # Diff-t: keep the learned systematic part, swap Gaussian eps for Student-t.
    rng = np.random.default_rng(seed)
    tparams = fit_t_idio(Xs, beta)
    systematic = (gen_s @ beta) @ beta.T
    gen_t_s = systematic + sample_t_idio(tparams, len(gen_s), rng)

    nu_med = float(np.median([p[0] for p in tparams]))
    return gen_s * sd + mu, gen_t_s * sd + mu, float(gen_s.std(0).mean()), nu_med


def run_one(pool, test_x, n, sde):
    # Draw the training window RANDOMLY from the pool, not as the last N days. The last N
    # days were a recent, calm sample (portfolio kurtosis 1.3-2.5) while the test set spans
    # 2010-2024 including COVID (kurtosis 8.5), so every model looked like it 'understated
    # the tails' when it was faithfully reproducing the data it was shown.
    train_x = pool[np.random.default_rng(7).choice(len(pool), size=n, replace=False)]
    d = train_x.shape[1]
    cov_t = np.cov(test_x, rowvar=False)
    mu_hat = train_x.mean(0)

    rng = np.random.default_rng(0)
    samples = {
        "Boot": train_x[rng.integers(0, n, size=N_GEN)],
        "G-POET": rng.multivariate_normal(mu_hat, poet_cov(train_x, K), size=N_GEN),
    }

    gd, gt, gs, nus = [], [], [], []
    for seed in SEEDS:
        a, b, s, nu = train_and_generate(train_x, K, seed, sde)
        gd.append(a)
        gt.append(b)
        gs.append(s)
        nus.append(nu)
    samples["Diff"] = np.vstack(gd)
    samples["Diff-t"] = np.vstack(gt)

    row = {"N": int(n), "d": int(d), "N_over_d": n / d, "gen_std": float(np.mean(gs)),
           "median_nu": float(np.mean(nus)), "scores": {}}

    for name, S in samples.items():
        cov = np.cov(S, rowvar=False)
        m = {"RE4": float(re4(cov, cov_t)),
             "minvar_vol": min_var_vol(cov, test_x),
             "mv_sharpe": mv_sharpe(mu_hat, cov, test_x),
             "taildep": tail_dep(S),
             # Idiosyncratic risk is only 0.2% of an 87-stock equal-weight portfolio's
             # variance, so portfolio kurtosis is blind to it. Per-asset kurtosis is where
             # fat idiosyncratic tails actually show up.
             "asset_kurt": float(np.mean(stats.kurtosis(S, axis=0)))}
        m.update(var_metrics(S, test_x))
        row["scores"][name] = m

    row["realized"] = {
        "taildep": tail_dep(test_x),
        "port_kurt": float(stats.kurtosis(test_x @ (np.ones(d) / d))),
        "asset_kurt": float(np.mean(stats.kurtosis(test_x, axis=0))),
        "train_port_kurt": float(stats.kurtosis(train_x @ (np.ones(d) / d))),
        "train_asset_kurt": float(np.mean(stats.kurtosis(train_x, axis=0))),
    }

    print(f"\n>>> N={n} N/d={n/d:.2f}  (gen std={np.mean(gs):.2f}, median t dof={np.mean(nus):.1f})")
    print(f"{'method':<9}{'RE4':>8}{'MinVarVol':>11}{'Sharpe':>8}{'VaR1%':>7}{'VaR5%':>7}{'ES5':>7}{'PKurt':>7}{'AKurt':>7}{'TailDep':>9}")
    for name, s in row["scores"].items():
        print(f"{name:<9}{s['RE4']:>8.3f}{s['minvar_vol']:>11.5f}{s['mv_sharpe']:>8.2f}"
              f"{s['var1_cover']*100:>6.1f}%{s['var5_cover']*100:>6.1f}%"
              f"{s['es5_ratio']:>7.2f}{s['port_kurt']:>7.2f}{s['asset_kurt']:>7.2f}{s['taildep']:>9.3f}")
    rz = row["realized"]
    print(f"{'REALIZED':<9}{'-':>8}{'-':>11}{'-':>8}{'  1.0%':>7}{'  5.0%':>7}{'1.00':>7}"
          f"{rz['port_kurt']:>7.2f}{rz['asset_kurt']:>7.2f}{rz['taildep']:>9.3f}")
    print(f"    (TRAIN window kurtosis: portfolio {rz['train_port_kurt']:.2f}, "
          f"per-asset {rz['train_asset_kurt']:.2f} -- what the models can actually learn)")
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
        with CKPT.open("a") as f:
            f.write(json.dumps(run_one(pool, test_x, n, sde)) + "\n")

    rows = [json.loads(l) for l in CKPT.read_text().splitlines() if l.strip()]
    print("\n" + "=" * 82)
    print("SUMMARY (averaged over N)")
    print(f"{'method':<9}{'RE4':>8}{'MinVarVol':>11}{'Sharpe':>8}{'VaR1%':>8}{'VaR5%':>8}{'ES5':>7}{'PKurt':>8}{'AKurt':>8}{'TailDep':>9}")
    for m in ["Boot", "G-POET", "Diff", "Diff-t"]:
        g = lambda key: np.mean([r["scores"][m][key] for r in rows])
        print(f"{m:<9}{g('RE4'):>8.3f}{g('minvar_vol'):>11.5f}{g('mv_sharpe'):>8.2f}"
              f"{g('var1_cover')*100:>7.1f}%{g('var5_cover')*100:>7.1f}%{g('es5_ratio'):>7.2f}"
              f"{g('port_kurt'):>8.2f}{g('asset_kurt'):>8.2f}{g('taildep'):>9.3f}")
    rk = np.mean([r["realized"]["port_kurt"] for r in rows])
    ra = np.mean([r["realized"]["asset_kurt"] for r in rows])
    rt = np.mean([r["realized"]["taildep"] for r in rows])
    tk = np.mean([r["realized"]["train_port_kurt"] for r in rows])
    ta = np.mean([r["realized"]["train_asset_kurt"] for r in rows])
    print(f"{'REALIZED':<9}{'-':>8}{'-':>11}{'-':>8}{'1.0%':>8}{'5.0%':>8}{'1.00':>7}{rk:>8.2f}{ra:>8.2f}{rt:>9.3f}")
    print(f"{'(TRAIN)':<9}{'-':>8}{'-':>11}{'-':>8}{'-':>8}{'-':>8}{'-':>7}{tk:>8.2f}{ta:>8.2f}{'-':>9}")


if __name__ == "__main__":
    main()
