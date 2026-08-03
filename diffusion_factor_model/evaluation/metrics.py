import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict


class Evaluator:
    def __init__(self, config):
        self.config = config

    def correlation_distance(self, real_returns, gen_returns):
        corr_real = real_returns.corr()
        corr_gen = gen_returns.corr()
        # Use common columns
        common = corr_real.index.intersection(corr_gen.index)
        if len(common) == 0:
            return float('nan')
        cr = corr_real.loc[common, common]
        cg = corr_gen.loc[common, common]
        return np.linalg.norm(cr.values - cg.values, 'fro')

    def volatility_acf(self, returns, lags=20):
        abs_returns = returns.abs()
        acf = []
        for lag in range(1, lags + 1):
            corr = abs_returns.corrwith(abs_returns.shift(lag))
            acf.append(corr.mean())
        return np.array(acf)

    def tail_dependence(self, returns, quantile=0.05):
        cols = list(returns.columns[:20])
        tail_deps = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                q_i = returns[cols[i]].quantile(quantile)
                q_j = returns[cols[j]].quantile(quantile)
                joint = ((returns[cols[i]] < q_i) & (returns[cols[j]] < q_j)).sum()
                marginal = (returns[cols[i]] < q_i).sum()
                if marginal > 0:
                    tail_deps.append(joint / marginal)
        return np.mean(tail_deps) if tail_deps else 0

    def compute_all_metrics(self, real_returns, gen_returns):
        metrics = {
            'correlation_distance': self.correlation_distance(real_returns, gen_returns),
            'vol_acf_real': self.volatility_acf(real_returns),
            'vol_acf_gen': self.volatility_acf(gen_returns),
            'tail_dep_real': self.tail_dependence(real_returns),
            'tail_dep_gen': self.tail_dependence(gen_returns),
            'mean_real': real_returns.mean(axis=1).mean(),
            'mean_gen': gen_returns.mean(axis=1).mean(),
            'std_real': real_returns.std(axis=1).mean(),
            'std_gen': gen_returns.std(axis=1).mean(),
            'skew_real': real_returns.skew(axis=1).mean(),
            'skew_gen': gen_returns.skew(axis=1).mean(),
            'kurt_real': real_returns.kurtosis(axis=1).mean(),
            'kurt_gen': gen_returns.kurtosis(axis=1).mean(),
        }
        return metrics

    def plot_comparison(self, real_returns, gen_returns, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].hist(real_returns.values.flatten(), bins=100, alpha=0.5, label='Real', density=True)
        axes[0, 0].hist(gen_returns.values.flatten(), bins=100, alpha=0.5, label='Generated', density=True)
        axes[0, 0].set_title('Return Distribution')
        axes[0, 0].legend()
        axes[0, 0].set_xlabel('Returns')

        vol_acf_real = self.volatility_acf(real_returns)
        vol_acf_gen = self.volatility_acf(gen_returns)
        axes[0, 1].plot(vol_acf_real, label='Real', marker='o')
        axes[0, 1].plot(vol_acf_gen, label='Generated', marker='s')
        axes[0, 1].set_title('Volatility ACF')
        axes[0, 1].set_xlabel('Lag')
        axes[0, 1].set_ylabel('ACF')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Subsample columns for correlation heatmap
        real_cols = list(real_returns.columns[:50])
        gen_cols = list(gen_returns.columns[:50])
        common = [c for c in real_cols if c in gen_returns.columns][:50]
        if len(common) > 1:
            corr_real = real_returns[common].corr()
            corr_gen = gen_returns[common].corr()
            sns.heatmap(corr_real, ax=axes[1, 0], cmap='coolwarm', center=0, xticklabels=False, yticklabels=False)
            axes[1, 0].set_title('Correlation Matrix (Real)')
            sns.heatmap(corr_gen, ax=axes[1, 1], cmap='coolwarm', center=0, xticklabels=False, yticklabels=False)
            axes[1, 1].set_title('Correlation Matrix (Generated)')
        else:
            axes[1, 0].text(0.5, 0.5, 'No common columns', ha='center', va='center')
            axes[1, 1].text(0.5, 0.5, 'No common columns', ha='center', va='center')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'Saved evaluation plot to {save_path}')
        plt.close()
