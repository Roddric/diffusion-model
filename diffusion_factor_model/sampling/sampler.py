import torch
import numpy as np
from tqdm import tqdm


class EulerMaruyamaSampler:
    def __init__(self, config, sde, score_net):
        self.config = config
        self.sde = sde
        self.score_net = score_net
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.score_net.to(self.device).eval()

    def sample(self, num_samples, latent_dim, num_steps=1000):
        with torch.no_grad():
            x = torch.randn(num_samples, latent_dim, device=self.device)
            dt = self.sde.T / num_steps
            timesteps = torch.linspace(self.sde.T, 0, num_steps + 1, device=self.device)

            for i in tqdm(range(num_steps), desc='Sampling'):
                t = timesteps[i].expand(num_samples)
                drift = self.sde.drift(x, t)
                score = self.score_net(x, t)
                diffusion = self.sde.diffusion(x, t)
                z = torch.randn_like(x) if i < num_steps - 1 else 0
                x = x - (drift - diffusion**2 * score) * dt + diffusion * np.sqrt(dt) * z

        return x.cpu().numpy()


class DPMSolverPlusPlus:
    def __init__(self, config, sde, score_net):
        self.config = config
        self.sde = sde
        self.score_net = score_net
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.score_net.to(self.device).eval()

    def sample(self, num_samples, latent_dim, num_steps=20):
        with torch.no_grad():
            x = torch.randn(num_samples, latent_dim, device=self.device)
            timesteps = torch.linspace(self.sde.T, 1e-3, num_steps + 1, device=self.device)

            for i in tqdm(range(num_steps), desc='DPM-Solver++'):
                t_curr = timesteps[i].expand(num_samples)
                t_next = timesteps[i + 1].expand(num_samples)

                # VP-SDE coefficients
                log_alpha_curr = -0.25 * t_curr**2 * (self.sde.beta_max - self.sde.beta_min) - 0.5 * t_curr * self.sde.beta_min
                log_alpha_next = -0.25 * t_next**2 * (self.sde.beta_max - self.sde.beta_min) - 0.5 * t_next * self.sde.beta_min

                alpha_curr = torch.exp(log_alpha_curr.unsqueeze(-1))
                alpha_next = torch.exp(log_alpha_next.unsqueeze(-1))
                
                sigma_curr = torch.sqrt(torch.clamp(1 - alpha_curr**2, min=1e-8))
                sigma_next = torch.sqrt(torch.clamp(1 - alpha_next**2, min=1e-8))

                # Score network predicts score = -eps / sigma
                # Convert to epsilon prediction
                score = self.score_net(x, t_curr)
                eps_pred = -score * sigma_curr
                
                # Clamp for stability
                eps_pred = torch.clamp(eps_pred, -10, 10)

                # Predict x0 from (x, eps): x = alpha * x0 + sigma * eps => x0 = (x - sigma * eps) / alpha
                x0_pred = (x - sigma_curr * eps_pred) / (alpha_curr + 1e-8)
                
                # DDIM update: x_next = alpha_next * x0_pred + sigma_next * eps_pred
                x = alpha_next * x0_pred + sigma_next * eps_pred

        return x.cpu().numpy()
