"""Training loop for diffusion models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


class DiffusionTrainer:
    """Train score network for diffusion model."""

    def __init__(self, config, sde, score_net: nn.Module):
        self.config = config
        self.sde = sde
        self.score_net = score_net
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.score_net.to(self.device)
        self.eps_t = 1e-3

        self.optimizer = torch.optim.AdamW(
            score_net.parameters(),
            lr=config.diffusion.learning_rate,
            weight_decay=config.diffusion.weight_decay
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.score_net.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            x0 = batch[0].to(self.device)
            batch_size = x0.shape[0]

            # Sample random time. Floor at eps_t: as t -> 0 the marginal std -> 0 and
            # the score target -eps/std diverges.
            t = self.eps_t + torch.rand(batch_size, device=self.device) * (self.sde.T - self.eps_t)

            # Sample from marginal p(x_t | x_0) using the same noise we score against
            mean, std = self.sde.marginal_params(x0, t)
            eps = torch.randn_like(x0)
            x_t = mean + std * eps

            # Likelihood-weighted score matching. Weighting ||s - s_true||^2 by std^2
            # reduces exactly to epsilon-prediction, whose target is O(1) at every t,
            # unlike the raw score target -eps/std which diverges as t -> 0.
            eps_pred = self.score_net.eps(x_t, t)
            loss = torch.mean((eps_pred - eps)**2)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.score_net.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def train(self, dataloader: DataLoader, num_epochs: int, save_dir: str = None):
        """Full training loop."""
        history = []
        for epoch in tqdm(range(num_epochs), desc="Training"):
            epoch_loss = self.train_epoch(dataloader)
            history.append(epoch_loss)

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")

        if save_dir:
            from pathlib import Path
            save_path = Path(save_dir) / "score_net.pt"
            torch.save(self.score_net.state_dict(), save_path)
            print(f"Saved model to {save_path}")

        return history