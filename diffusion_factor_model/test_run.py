"""Manual five-stock smoke run.

This module is intentionally import-safe so pytest collection never starts a live data
download or a training job. Run it directly from ``diffusion_factor_model/``.
"""

__test__ = False


def main():
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

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

    config = load_config('../test_config.yaml')
    print('=' * 60)
    print('DIFFUSION FACTOR MODEL - TEST RUN (5 stocks)')
    print('=' * 60)

    print('\n[1/7] Loading data...')
    returns, market = DataPipeline(config).load_all_data(max_stocks=5)
    print(f'Returns: {returns.shape}')

    print('\n[2/7] Extracting factors...')
    extractor = FactorExtractor(config)
    mean_results = extractor.extract_mean_factors(returns, market)
    residuals = extractor.compute_residuals(returns, mean_results)
    vol_results = extractor.extract_volatility_factors(residuals)
    print(f'Mean factors: {mean_results["factors"].columns.tolist()}')
    print(f'Vol factors: {vol_results["factors"].shape[1]}')

    print('\n[3/7] Parametrizing latent...')
    parametrizer = LatentParametrizer(config)
    latent, metadata = parametrizer.parametrize(mean_results, vol_results)
    print(f'Latent: {latent.shape}')

    print('\n[4/7] Training diffusion...')
    sde = VPSDE(
        beta_min=config.diffusion.beta_min,
        beta_max=config.diffusion.beta_max,
        T=config.diffusion.T,
    )
    score_net = ScoreNetwork(
        input_dim=latent.shape[1],
        hidden_dim=config.diffusion.hidden_dim,
        num_layers=config.diffusion.n_layers,
        dropout=config.diffusion.dropout,
        sde=sde,
    )
    trainer = DiffusionTrainer(config, sde, score_net)
    dataset = TensorDataset(torch.from_numpy(latent).float())
    dataloader = DataLoader(
        dataset, batch_size=config.diffusion.batch_size, shuffle=True
    )
    history = trainer.train(dataloader, num_epochs=config.diffusion.n_epochs)
    print(f'Final loss: {history[-1]:.6f}')

    print('\n[5/7] GARCH residuals...')
    idiosyncratic = extractor.compute_idiosyncratic(residuals, vol_results)
    garch = GARCHModeler(config)
    garch.fit_all(idiosyncratic)

    print('\n[6/7] Sampling...')
    samples = DPMSolverPlusPlus(config, sde, score_net).sample(
        100, latent.shape[1], num_steps=10
    )
    residual_samples = garch.sample_all(100)
    generated = ReturnReconstructor(
        config,
        mean_results,
        vol_results,
        parametrizer=parametrizer,
        latent_metadata=metadata,
    ).reconstruct(samples, residual_samples)
    print(f'Generated: {generated.shape}')

    print('\n[7/7] Evaluating...')
    metrics = Evaluator(config).compute_all_metrics(returns, generated)
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.floating, np.integer)):
            print(f'  {key}: {value:.6f}')
        elif isinstance(value, np.ndarray):
            print(f'  {key}: array({value.shape})')

    print('\n' + '=' * 60)
    print('TEST COMPLETE')
    print('=' * 60)


if __name__ == '__main__':
    main()
