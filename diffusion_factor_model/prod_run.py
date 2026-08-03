import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import torch

# Tiny score net: torch's default thread count costs ~30x in coordination overhead.
torch.set_num_threads(1)
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json
from datetime import datetime

from config.config import load_config
from data.loader import DataPipeline
from factors.extractor import FactorExtractor
from latent.parametrizer import LatentParametrizer
from diffusion.sde import VPSDE
from diffusion.score_net import ScoreNetwork
from diffusion.trainer import DiffusionTrainer
from residuals.garch import GARCHModeler
from sampling.sampler import DPMSolverPlusPlus
from reconstruction.reconstructor import ReturnReconstructor
from evaluation.metrics import Evaluator

print('=' * 70)
print('DIFFUSION FACTOR MODEL - PRODUCTION RUN (HS300 subset)')
print('=' * 70)
print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

config_path = '../prod_config.yaml'
config = load_config(config_path)

# Phase 1: Load data (100 stocks for efficiency)
print('\n[1/7] Loading data...')
dp = DataPipeline(config)
returns, mkt = dp.load_all_data(max_stocks=100)
print(f'Returns: {returns.shape}')

# Phase 2: Extract factors
print('\n[2/7] Extracting factors...')
fe = FactorExtractor(config)
mfr = fe.extract_mean_factors(returns, mkt)
resid = fe.compute_residuals(returns, mfr)
vfr = fe.extract_volatility_factors(resid)
print(f'Mean factors: {mfr["factors"].columns.tolist()}')
print(f'Vol factors: {vfr["factors"].shape[1]}')

# Phase 3: Parametrize latent
print('\n[3/7] Parametrizing latent...')
lp = LatentParametrizer(config)
latent, meta = lp.parametrize(mfr, vfr)
print(f'Latent: {latent.shape}')

# Phase 4: Train diffusion (500 epochs)
print('\n[4/7] Training diffusion (500 epochs)...')
sde = VPSDE(beta_min=config.diffusion.beta_min, beta_max=config.diffusion.beta_max, T=config.diffusion.T)
sn = ScoreNetwork(input_dim=latent.shape[1], hidden_dim=config.diffusion.hidden_dim, num_layers=config.diffusion.n_layers, dropout=config.diffusion.dropout, sde=sde)
trainer = DiffusionTrainer(config, sde, sn)
ds = TensorDataset(torch.from_numpy(latent).float())
dl = DataLoader(ds, batch_size=config.diffusion.batch_size, shuffle=True)
hist = trainer.train(dl, num_epochs=config.diffusion.n_epochs)
print(f'Final loss: {hist[-1]:.6f}')

# Phase 5: GARCH on the idiosyncratic part only (vol factors are reconstructed separately)
print('\n[5/7] GARCH residuals...')
idio = fe.compute_idiosyncratic(resid, vfr)
gm = GARCHModeler(config)
gr = gm.fit_all(idio)

# Phase 6: Sampling
print('\n[6/7] Sampling...')
n_gen = len(returns)
sampler = DPMSolverPlusPlus(config, sde, sn)
samples = sampler.sample(n_gen, latent.shape[1], num_steps=20)
res_samples = gm.sample_all(n_gen)

recon = ReturnReconstructor(config, mfr, vfr, parametrizer=lp, latent_metadata=meta)
gen = recon.reconstruct(samples, res_samples)
print(f'Generated: {gen.shape}')

# Phase 7: Evaluate
print('\n[7/7] Evaluating...')
ev = Evaluator(config)
metrics = ev.compute_all_metrics(returns, gen)

print('\n' + '=' * 70)
print('PRODUCTION RESULTS')
print('=' * 70)
print(f'Stocks loaded: {len(returns.columns)}')
print(f'Trading days: {len(returns)}')
print(f'Latent dim: {latent.shape[1]}')
print(f'Final training loss: {hist[-1]:.6f}')
print(f'GARCH stocks fitted: {len(gr)}')

print('\n--- Key Metrics ---')
for k, v in metrics.items():
    if isinstance(v, (int, float, np.floating, np.integer)):
        print(f'  {k}: {v:.6f}')
    elif isinstance(v, np.ndarray):
        print(f'  {k}: array({v.shape}) mean={v.mean():.6f}')

# Save results
results = {
    'timestamp': datetime.now().isoformat(),
    'config': {
        'n_stocks': len(returns.columns),
        'n_days': len(returns),
        'latent_dim': latent.shape[1],
        'n_epochs': config.diffusion.n_epochs,
        'final_loss': float(hist[-1]),
        'n_garch_fitted': len(gr)
    },
    'metrics': {}
}

for k, v in metrics.items():
    if isinstance(v, (int, float, np.floating, np.integer)):
        results['metrics'][k] = float(v)
    elif isinstance(v, np.ndarray):
        results['metrics'][k] = {'mean': float(v.mean()), 'std': float(v.std()), 'shape': list(v.shape)}

output_dir = Path('../prod_output')
output_dir.mkdir(exist_ok=True)
with open(output_dir / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\nResults saved to {output_dir}/results.json')
print('=' * 70)
print('PRODUCTION COMPLETE')
print('=' * 70)
