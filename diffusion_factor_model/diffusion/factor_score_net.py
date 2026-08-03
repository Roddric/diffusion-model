"""Factor-structured score network -- Chen, Xu, Xu & Zhang (2026), eq. (12)/(16).

The key result (their Lemma 1): under a factor model R = beta F + eps with orthonormal beta
and diagonal idiosyncratic covariance diag(sigma^2), the score of the diffused returns
decomposes exactly as

    grad log p_t(r) = alpha_t Lam_t^-1 beta . xi(beta^T Lam_t^-1 r, t)  -  Lam_t^-1 r
                      \_________ subspace score _________/               \_ complement _/

    Lam_t = diag(h_t + sigma_i^2 alpha_t^2),   alpha_t = VP mean coeff,  h_t = 1 - alpha_t^2

The ONLY nonlinear, learned object is xi: R^k -> R^k. The complement score is linear and
closed-form. So the network never faces more than k dimensions even though the model
generates the full d-dimensional return vector -- which is precisely the sample-efficiency
claim, and precisely what our earlier "project to 10 factors, then diffuse" design threw
away (it truncated the data instead of structuring the score).
"""

import numpy as np
import torch
import torch.nn as nn

from .score_net import SinusoidalPositionalEmbedding


class XiNet(nn.Module):
    """The k-dimensional nonlinear core, xi(z, t): R^k x [0,T] -> R^k."""

    def __init__(self, k, hidden=128, layers=3, dropout=0.0):
        super().__init__()
        self.time_embed = SinusoidalPositionalEmbedding(hidden)
        self.time_proj = nn.Linear(hidden, hidden)
        self.inp = nn.Linear(k, hidden)
        self.blocks = nn.ModuleList(
            nn.ModuleDict({
                "norm": nn.LayerNorm(hidden),
                "fc1": nn.Linear(hidden, hidden),
                "fc2": nn.Linear(hidden, hidden),
                "drop": nn.Dropout(dropout),
            })
            for _ in range(layers)
        )
        self.out = nn.Linear(hidden, k)

    def forward(self, z, t):
        h = self.inp(z)
        temb = self.time_proj(self.time_embed(t))
        for b in self.blocks:
            res = h
            h = b["norm"](h) + temb
            h = torch.nn.functional.silu(h)
            h = b["fc2"](torch.nn.functional.silu(b["fc1"](h)))
            h = b["drop"](h) + res
        return self.out(h)


class FactorScoreNetwork(nn.Module):
    """Score network s(r, t) for the full d-dimensional return vector.

    Args:
        beta:   (d, k) factor loadings with ORTHONORMAL columns (Assumption 1(i)).
        sigma2: (d,)   idiosyncratic variances.
    """

    def __init__(self, beta, sigma2, sde, hidden=128, layers=3, dropout=0.0):
        super().__init__()
        beta = np.asarray(beta, dtype=np.float32)
        sigma2 = np.asarray(sigma2, dtype=np.float32)

        # Assumption 1(i) is load-bearing: the decomposition is only exact for orthonormal
        # beta. Enforce it rather than trust the caller.
        gram = beta.T @ beta
        if not np.allclose(gram, np.eye(beta.shape[1]), atol=1e-4):
            raise ValueError("beta must have orthonormal columns (beta^T beta = I)")

        self.register_buffer("beta", torch.from_numpy(beta))           # (d, k)
        self.register_buffer("sigma2", torch.from_numpy(sigma2))       # (d,)
        self.sde = sde
        self.d, self.k = beta.shape
        self.xi = XiNet(self.k, hidden, layers, dropout)

    def _lam(self, t):
        """Lam_t = h_t + sigma_i^2 alpha_t^2, shape (batch, d)."""
        log_a = -0.25 * t**2 * (self.sde.beta_max - self.sde.beta_min) - 0.5 * t * self.sde.beta_min
        alpha = torch.exp(log_a).unsqueeze(-1)          # (batch, 1)
        h = (1.0 - alpha**2).clamp(min=1e-8)            # (batch, 1)
        lam = h + self.sigma2.unsqueeze(0) * alpha**2   # (batch, d)
        return alpha, lam

    def forward(self, r, t):
        alpha, lam = self._lam(t)
        r_over_lam = r / lam                            # Lam^-1 r
        z = r_over_lam @ self.beta                      # beta^T Lam^-1 r  -> (batch, k)
        xi = self.xi(z, t)                              # (batch, k)
        # score = (alpha * beta xi - r) / lam
        return (alpha * (xi @ self.beta.T) - r) / lam


def estimate_factor_structure(X, k):
    """PCA on standardized returns -> orthonormal beta (d,k) and idiosyncratic vars (d,).

    X: (n, d) standardized returns (each column zero-mean, unit-variance).
    """
    cov = np.cov(X, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1][:k]
    beta = vecs[:, idx]                                  # orthonormal by construction
    resid = X - (X @ beta) @ beta.T                      # (I - beta beta^T) X
    sigma2 = resid.var(axis=0)
    return beta.astype(np.float32), np.maximum(sigma2, 1e-6).astype(np.float32)
