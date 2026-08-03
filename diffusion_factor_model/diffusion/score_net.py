"""Score network architectures for diffusion models."""

import torch
import torch.nn as nn
import math


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal time embedding."""

    def __init__(self, embedding_dim, scale=1000.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        # t is continuous in [0, 1], but the sinusoid frequencies span 1 .. 1e-4.
        # Without rescaling, every sin(t * f) sits in the small-angle regime and the
        # embedding degenerates to something almost linear in t, so the network cannot
        # resolve the noise level. Rescaling to the usual 0..1000 timestep range fixes it.
        self.scale = scale

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = (t * self.scale).unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResidualMLP(nn.Module):
    """Residual MLP with FiLM conditioning."""

    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_embed = SinusoidalPositionalEmbedding(hidden_dim)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)

        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(nn.ModuleDict({
                "norm": nn.LayerNorm(hidden_dim),
                "fc1": nn.Linear(hidden_dim, hidden_dim),
                "fc2": nn.Linear(hidden_dim, hidden_dim),
                "dropout": nn.Dropout(dropout)
            }))

        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) latent state
            t: (batch,) time in [0, 1]
        """
        h = self.input_proj(x)
        t_emb = self.time_embed(t)
        t_emb = self.time_proj(t_emb)

        for block in self.blocks:
            residual = h
            h = block["norm"](h)
            h = h + t_emb  # FiLM conditioning
            h = torch.nn.functional.silu(h)
            h = block["fc1"](h)
            h = torch.nn.functional.silu(h)
            h = block["fc2"](h)
            h = block["dropout"](h)
            h = h + residual

        return self.output_proj(h)


class ScoreNetwork(nn.Module):
    """Score network s_theta(x, t) = -eps_theta(x, t) / sigma_t.

    The MLP predicts the noise eps, which is O(1) at every t, and the 1/sigma_t
    blow-up is applied analytically. Regressing the raw score directly would ask
    the MLP to span ~1 (at t=T) to ~100 (at t=1e-3) in output magnitude, which it
    does not fit well and which shows up as heavily over-dispersed samples.
    """

    def __init__(self, input_dim, hidden_dim=256, num_layers=6, dropout=0.1, sde=None):
        super().__init__()
        self.mlp = ResidualMLP(input_dim, hidden_dim, num_layers, dropout)
        if sde is None:
            from .sde import VPSDE
            sde = VPSDE()
        self.sde = sde

    def eps(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the noise added at time t."""
        return self.mlp(x, t)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute score function estimate."""
        return -self.eps(x, t) / self.sde.marginal_std(t)