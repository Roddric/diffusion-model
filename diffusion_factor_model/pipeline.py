import numpy as np
import pandas as pd
import torch

# Tiny score net: torch's default thread count costs ~30x in coordination overhead.
torch.set_num_threads(1)
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

if __package__:
    from .config.config import load_config
    from .data.loader import DataPipeline, SyntheticDataPipeline
    from .factors.extractor import FactorExtractor
    from .latent.parametrizer import LatentParametrizer
    from .diffusion.sde import VPSDE
    from .diffusion.score_net import ScoreNetwork
    from .diffusion.trainer import DiffusionTrainer
    from .residuals.garch import GARCHModeler
    from .sampling.sampler import EulerMaruyamaSampler, DPMSolverPlusPlus
    from .reconstruction.reconstructor import ReturnReconstructor
    from .evaluation.metrics import Evaluator
else:
    from config.config import load_config
    from data.loader import DataPipeline, SyntheticDataPipeline
    from factors.extractor import FactorExtractor
    from latent.parametrizer import LatentParametrizer
    from diffusion.sde import VPSDE
    from diffusion.score_net import ScoreNetwork
    from diffusion.trainer import DiffusionTrainer
    from residuals.garch import GARCHModeler
    from sampling.sampler import EulerMaruyamaSampler, DPMSolverPlusPlus
    from reconstruction.reconstructor import ReturnReconstructor
    from evaluation.metrics import Evaluator


def run_pipeline(config_path=None, generate_samples=False):
    config = load_config(config_path)
    Path(config.data_dir).mkdir(parents=True, exist_ok=True)
    print('=' * 60)
    print('DIFFUSION FACTOR MODEL PIPELINE')
    print('=' * 60)

    # Phase 1: Load data
    print('\n[1/7] Loading data...')
    if config.data.source == 'synthetic':
        data_pipeline = SyntheticDataPipeline(config)
    elif config.data.source == 'akshare':
        data_pipeline = DataPipeline(config)
    elif config.data.source == 'yfinance':
        if __package__:
            from .data.yf_loader import YFinanceDataPipeline
        else:
            from data.yf_loader import YFinanceDataPipeline
        data_pipeline = YFinanceDataPipeline(config)
    else:
        raise ValueError(
            f"Unknown data source {config.data.source!r}; "
            "expected 'synthetic', 'akshare', or 'yfinance'."
        )
    stock_returns, market_returns = data_pipeline.load_all_data()
    print(f'Stock returns shape: {stock_returns.shape}')

    # Phase 2: Extract factors
    print('\n[2/7] Extracting factors...')
    factor_extractor = FactorExtractor(config)
    mean_factor_results = factor_extractor.extract_mean_factors(stock_returns, market_returns)
    residuals = factor_extractor.compute_residuals(stock_returns, mean_factor_results)
    vol_factor_results = factor_extractor.extract_volatility_factors(residuals)
    print(f"Mean factors: {len(mean_factor_results['factors'].columns)}")
    print(f"Vol factors: {len(vol_factor_results['factors'].columns)}")

    # Phase 3: Parametrize latent space
    print('\n[3/7] Parametrizing latent space...')
    parametrizer = LatentParametrizer(config)
    latent_matrix, latent_metadata = parametrizer.parametrize(mean_factor_results, vol_factor_results)
    print(f'Latent matrix shape: {latent_matrix.shape}')

    # Phase 4: Train diffusion model
    print('\n[4/7] Training diffusion model...')
    sde = VPSDE(
        beta_min=config.diffusion.beta_min,
        beta_max=config.diffusion.beta_max,
        T=config.diffusion.T
    )
    score_net = ScoreNetwork(
        input_dim=latent_matrix.shape[1],
        hidden_dim=config.diffusion.hidden_dim,
        num_layers=config.diffusion.n_layers,
        dropout=config.diffusion.dropout,
        sde=sde
    )
    trainer = DiffusionTrainer(config, sde, score_net)

    latent_tensor = torch.from_numpy(latent_matrix).float()
    dataset = TensorDataset(latent_tensor)
    dataloader = DataLoader(dataset, batch_size=config.diffusion.batch_size, shuffle=True)

    history = trainer.train(dataloader, num_epochs=config.diffusion.n_epochs, save_dir=config.data_dir)

    # Phase 5: Model the idiosyncratic residual with GARCH. The PCA vol factors are
    # reconstructed separately, so fitting GARCH on the full residual double-counts them.
    print('\n[5/7] Modeling residuals with GARCH...')
    idio_residuals = factor_extractor.compute_idiosyncratic(residuals, vol_factor_results)
    garch_modeler = GARCHModeler(config)
    garch_results = garch_modeler.fit_all(idio_residuals)

    # Phase 6: Generate samples (optional)
    if generate_samples:
        print('\n[6/7] Generating synthetic samples...')
        if config.sampling.solver == 'dpm_solver_pp':
            sampler = DPMSolverPlusPlus(config, sde, score_net)
        elif config.sampling.solver == 'euler_maruyama':
            sampler = EulerMaruyamaSampler(config, sde, score_net)
        else:
            raise ValueError(
                f"Unknown sampling solver {config.sampling.solver!r}; "
                "expected 'dpm_solver_pp' or 'euler_maruyama'."
            )
        num_samples = config.sampling.n_samples
        latent_samples = sampler.sample(
            num_samples,
            latent_matrix.shape[1],
            num_steps=config.sampling.n_steps,
        )

        residual_samples = garch_modeler.sample_all(num_samples)

        reconstructor = ReturnReconstructor(
            config, mean_factor_results, vol_factor_results,
            parametrizer=parametrizer, latent_metadata=latent_metadata
        )
        generated_returns = reconstructor.reconstruct(latent_samples, residual_samples)
        print(f'Generated returns shape: {generated_returns.shape}')

        # Phase 7: Evaluate
        print('\n[7/7] Evaluating generated samples...')
        evaluator = Evaluator(config)
        metrics = evaluator.compute_all_metrics(stock_returns, generated_returns)

        print('\nEvaluation Metrics:')
        print('-' * 40)
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                print(f'{key}: {value:.6f}')

        plot_path = Path(config.data_dir) / 'evaluation_plots.png'
        evaluator.plot_comparison(stock_returns, generated_returns, str(plot_path))

        output_path = Path(config.data_dir) / 'generated_returns.csv'
        generated_returns.to_csv(output_path)
        print(f'\nGenerated returns saved to {output_path}')
    else:
        print('\n[6/7] Skipping sample generation (set generate_samples=True to enable)')
        print('[7/7] Skipping evaluation')

    print('\n' + '=' * 60)
    print('PIPELINE COMPLETE')
    print('=' * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Diffusion Factor Model Pipeline')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--generate', action='store_true', help='Generate samples after training')
    args = parser.parse_args()

    run_pipeline(config_path=args.config, generate_samples=args.generate)
