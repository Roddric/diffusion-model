"""Diffusion core: SDE, score parametrization, time embedding, samplers."""

import numpy as np
import torch

from diffusion.sde import VPSDE
from diffusion.score_net import ScoreNetwork, SinusoidalPositionalEmbedding
from sampling.sampler import DPMSolverPlusPlus, EulerMaruyamaSampler


def test_marginal_std_matches_marginal_params():
    sde = VPSDE()
    x0 = torch.randn(16, 4)
    t = torch.rand(16)
    _, std = sde.marginal_params(x0, t)
    assert torch.allclose(sde.marginal_std(t), std, atol=1e-6)


def test_score_is_eps_over_minus_sigma():
    """The network must expose score = -eps/sigma, which the samplers rely on to invert."""
    sde = VPSDE()
    net = ScoreNetwork(input_dim=4, hidden_dim=16, num_layers=2, dropout=0.0, sde=sde).eval()
    x, t = torch.randn(8, 4), torch.rand(8) * 0.9 + 0.05

    with torch.no_grad():
        score = net(x, t)
        eps = net.eps(x, t)

    assert torch.allclose(score, -eps / sde.marginal_std(t), atol=1e-6)


def test_score_target_stays_bounded_near_zero():
    """Regression: the old objective regressed on the raw score -eps/std, whose target
    diverges as t -> 0. The eps target must stay O(1) at every t."""
    sde = VPSDE()
    net = ScoreNetwork(input_dim=4, hidden_dim=16, num_layers=2, dropout=0.0, sde=sde).eval()
    t_tiny = torch.full((32,), 1e-3)
    x = torch.randn(32, 4)

    with torch.no_grad():
        eps_pred = net.eps(x, t_tiny)
        raw_score = net(x, t_tiny)

    assert eps_pred.abs().max() < 100, "eps target should be O(1)"
    # The raw score really is huge here -- that is exactly why we do not regress on it.
    assert raw_score.abs().max() > eps_pred.abs().max()


def test_time_embedding_resolves_distinct_noise_levels():
    """Regression: with t in [0,1] and unscaled frequencies, every sin(t*f) sits in the
    small-angle regime and the embedding collapses to ~linear in t (numerical rank ~2),
    leaving the network unable to tell noise levels apart."""
    emb = SinusoidalPositionalEmbedding(64)
    t = torch.linspace(1e-3, 1.0, 50)
    e = emb(t).detach().numpy()

    rank = np.linalg.matrix_rank(e, tol=1e-3)
    assert rank >= 8, f"time embedding is degenerate (rank {rank})"

    # Nearby noise levels must be distinguishable.
    lo, hi = emb(torch.tensor([0.01])), emb(torch.tensor([0.05]))
    assert torch.norm(lo - hi) > 0.1


class _AnalyticGaussianScore(torch.nn.Module):
    """For x0 ~ N(0, I) the VP marginal is N(0, I) at every t, so the true score is -x."""

    def __init__(self, sde):
        super().__init__()
        self.sde = sde

    def forward(self, x, t):
        return -x

    def eps(self, x, t):
        return x * self.sde.marginal_std(t)


def test_dpm_sampler_recovers_standard_normal():
    sde = VPSDE()
    s = DPMSolverPlusPlus(None, sde, _AnalyticGaussianScore(sde)).sample(4000, 5, num_steps=50)
    assert abs(s.mean()) < 0.1
    assert abs(s.std() - 1.0) < 0.15


def test_euler_sampler_recovers_standard_normal():
    sde = VPSDE()
    s = EulerMaruyamaSampler(None, sde, _AnalyticGaussianScore(sde)).sample(4000, 5, num_steps=500)
    assert abs(s.mean()) < 0.1
    assert abs(s.std() - 1.0) < 0.15
