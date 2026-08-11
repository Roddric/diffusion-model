import yaml
import warnings
from pathlib import Path
from dataclasses import dataclass, field, fields
from typing import List, Optional


def _section_kwargs(cls, values, section, aliases=None):
    """Normalize one YAML section and report keys that would otherwise be ignored."""
    values = dict(values or {})
    aliases = aliases or {}

    for old, new in aliases.items():
        if old not in values:
            continue
        if new in values:
            warnings.warn(
                f"Config key '{section}.{old}' is deprecated and ignored because "
                f"'{section}.{new}' is also set.",
                UserWarning,
                stacklevel=3,
            )
        else:
            warnings.warn(
                f"Config key '{section}.{old}' is deprecated; use "
                f"'{section}.{new}' instead.",
                FutureWarning,
                stacklevel=3,
            )
            values[new] = values[old]
        del values[old]

    valid = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - valid)
    if unknown:
        names = ", ".join(f"'{section}.{key}'" for key in unknown)
        warnings.warn(
            f"Unsupported config key(s): {names}. They have no effect.",
            UserWarning,
            stacklevel=3,
        )
    return {k: v for k, v in values.items() if k in valid}


@dataclass
class DataConfig:
    universe: str = 'hs300'
    universe_manifest: Optional[str] = None
    universe_snapshot_date: Optional[str] = None
    market_benchmark: Optional[str] = None
    ticker_convention: str = 'us'
    start_date: str = '20150101'
    end_date: str = '20241231'
    eligibility_end_date: Optional[str] = None
    preprocess_fit_end_date: Optional[str] = None
    frequency: str = 'daily'
    min_history: int = 252
    min_coverage: float = 0.95
    winsorize_std: float = 3.0
    adjustment: str = 'hfq'
    download_workers: int = 1
    akshare_endpoint: str = 'eastmoney'
    handle_limit_days: str = 'flag'
    source: str = 'akshare'
    synthetic_n_stocks: int = 20
    synthetic_n_days: int = 750
    random_seed: int = 42

@dataclass
class FactorConfig:
    mean_factors: List[str] = field(default_factory=lambda: [
        'market', 'momentum', 'reversal', 'turnover', 'illiquidity'
    ])
    n_vol_factors: int = 10
    vol_window: int = 21
    vol_floor: float = 1e-8
    vol_lower_quantile: float = 0.01
    vol_upper_quantile: float = 0.99
    pca_explained_var_threshold: float = 0.8
    use_industry: bool = True
    industry_classification: str = 'citics'

@dataclass
class LatentConfig:
    use_log_variance: bool = False
    standardize: bool = True
    shrinkage: str = 'ledoit_wolf'
    target_correlation: Optional[float] = None

@dataclass
class DiffusionConfig:
    network_type: str = 'residual_mlp'
    hidden_dim: int = 256
    n_layers: int = 6
    dropout: float = 0.1
    sde_type: str = 'vp'
    beta_min: float = 0.1
    beta_max: float = 20.0
    T: float = 1.0
    n_epochs: int = 1000
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    lr_scheduler: str = 'cosine'
    warmup_steps: int = 1000
    use_score_decomposition: bool = True
    factor_dim: Optional[int] = None
    condition_on_macro: bool = True
    condition_on_regime: bool = True

@dataclass
class TemporalDiffusionConfig:
    context_length: int = 60
    horizon: int = 20
    hidden_dim: int = 64
    n_layers: int = 4
    dropout: float = 0.05
    batch_size: int = 64
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    training_steps: int = 8000
    validation_interval: int = 250
    early_stopping_patience: int = 8
    sampling_steps: int = 30
    ensemble_paths: int = 20
    split_output_heads: bool = True
    mean_loss_weight: float = 1.0
    volatility_loss_weight: float = 2.0
    validation_seeds: List[int] = field(
        default_factory=lambda: [42, 314, 2718]
    )
    path_validation_paths: int = 8
    path_validation_steps: int = 10
    residual_scale_grid: List[float] = field(
        default_factory=lambda: [1.0]
    )

@dataclass
class ResidualConfig:
    min_obs: int = 60
    model_type: str = 'gjr_garch'
    distribution: str = 'student_t'
    p: int = 1
    q: int = 1
    o: int = 1

@dataclass
class SamplingConfig:
    solver: str = 'dpm_solver_pp'
    n_steps: int = 20
    n_samples: int = 10000

@dataclass
class EvaluationConfig:
    metrics: List[str] = field(default_factory=lambda: [
        'correlation_distance', 'volatility_acf', 'tail_dependence',
        'sharpe_ratio', 'var_backtest', 'turnover'
    ])
    portfolio_weights: str = 'equal'
    risk_free_rate: float = 0.02

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    factors: FactorConfig = field(default_factory=FactorConfig)
    latent: LatentConfig = field(default_factory=LatentConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    temporal: TemporalDiffusionConfig = field(
        default_factory=TemporalDiffusionConfig
    )
    residuals: ResidualConfig = field(default_factory=ResidualConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    data_dir: str = 'data'
    model_dir: str = 'models'
    output_dir: str = 'outputs'

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            d = yaml.safe_load(f) or {}
        if not isinstance(d, dict):
            raise ValueError("The configuration root must be a YAML mapping.")

        data_values = dict(d.get('data', {}) or {})
        data_dir = d.get('data_dir')
        if 'data_dir' in data_values:
            nested_data_dir = data_values.pop('data_dir')
            if data_dir is None:
                warnings.warn(
                    "Config key 'data.data_dir' is deprecated; use top-level "
                    "'data_dir' instead.",
                    FutureWarning,
                    stacklevel=2,
                )
                data_dir = nested_data_dir
            else:
                warnings.warn(
                    "Config key 'data.data_dir' is ignored because top-level "
                    "'data_dir' is also set.",
                    UserWarning,
                    stacklevel=2,
                )

        root_keys = {
            'data', 'factors', 'latent', 'diffusion', 'temporal', 'residuals',
            'sampling', 'evaluation', 'data_dir', 'model_dir', 'output_dir',
        }
        unknown_root = sorted(set(d) - root_keys)
        if unknown_root:
            names = ", ".join(repr(key) for key in unknown_root)
            warnings.warn(
                f"Unsupported top-level config key(s): {names}. They have no effect.",
                UserWarning,
                stacklevel=2,
            )

        return cls(
            data=DataConfig(**_section_kwargs(DataConfig, data_values, 'data')),
            factors=FactorConfig(**_section_kwargs(
                FactorConfig,
                d.get('factors', {}),
                'factors',
                {
                    'num_vol_factors': 'n_vol_factors',
                    'pca_variance_threshold': 'pca_explained_var_threshold',
                },
            )),
            latent=LatentConfig(**_section_kwargs(
                LatentConfig,
                d.get('latent', {}),
                'latent',
                {'shrinkage_method': 'shrinkage'},
            )),
            diffusion=DiffusionConfig(**_section_kwargs(
                DiffusionConfig,
                d.get('diffusion', {}),
                'diffusion',
                {'num_layers': 'n_layers'},
            )),
            temporal=TemporalDiffusionConfig(**_section_kwargs(
                TemporalDiffusionConfig, d.get('temporal', {}), 'temporal'
            )),
            residuals=ResidualConfig(**_section_kwargs(
                ResidualConfig, d.get('residuals', {}), 'residuals'
            )),
            sampling=SamplingConfig(**_section_kwargs(
                SamplingConfig,
                d.get('sampling', {}),
                'sampling',
                {
                    'method': 'solver',
                    'num_steps': 'n_steps',
                    'num_samples': 'n_samples',
                },
            )),
            evaluation=EvaluationConfig(**_section_kwargs(
                EvaluationConfig, d.get('evaluation', {}), 'evaluation'
            )),
            data_dir=data_dir or 'data',
            model_dir=d.get('model_dir', 'models'),
            output_dir=d.get('output_dir', 'outputs'),
        )

def load_config(path=None):
    if path is None or not Path(path).exists():
        return Config()
    return Config.from_yaml(path)
