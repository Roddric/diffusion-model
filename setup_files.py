import pathlib

# config.py
code = """\"\"\"Configuration module for diffusion factor model.\"\"\"

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DataConfig:
    \"\"\"Data loading and preprocessing configuration.\"\"\"
    universe: str = "hs300"
    start_date: str = "20150101"
    end_date: str = "20241231"
    frequency: str = "daily"
    min_history: int = 252
    winsorize_std: float = 3.0
    handle_limit_days: str = "flag"

@dataclass
class FactorConfig:
    \"\"\"Factor extraction configuration.\"\"\"
    mean_factors: List[str] = field(default_factory=lambda: [
        "market", "size", "value", "momentum", "reversal",
        "turnover", "illiquidity"
    ])
    n_vol_factors: int = 10
    pca_explained_var_threshold: float = 0.8
    use_industry: bool = True
    industry_classification: str = "citics"

@dataclass
class LatentConfig:
    \"\"\"Latent parametrization configuration.\"\"\"
    use_log_vol: bool = True
    shrinkage: str = "ledoit_wolf"
    target_correlation: Optional[float] = None

@dataclass
class DiffusionConfig:
    \"\"\"Diffusion model configuration.\"\"\"
    network_type: str = "residual_mlp"
    hidden_dim: int = 256
    n_layers: int = 6
    dropout: float = 0.1
    sde_type: str = "vp"
    beta_min: float = 0.1
    beta_max: float = 20.0
    T: float = 1.0
    n_epochs: int = 1000
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    lr_scheduler: str = "cosine"
    warmup_steps: int = 1000
    use_score_decomposition: bool = True
    factor_dim: Optional[int] = None
    condition_on_macro: bool = True
    condition_on_regime: bool = True

@dataclass
class ResidualConfig:
    \"\"\"Residual modeling configuration.\"\"\"
    model_type: str = "gjr_garch"
    distribution: str = "student_t"
    p: int = 1
    q: int = 1
    o: int = 1

@dataclass
class SamplingConfig:
    \"\"\"Sampling configuration.\"\"\"
    solver: str = "dpm_solver_pp"
    n_steps: int = 20
    n_samples: int = 10000

@dataclass
class EvaluationConfig:
    \"\"\"Evaluation configuration.\"\"\"
    metrics: List[str] = field(default_factory=lambda: [
        "correlation_distance",
        "volatility_acf",
        "tail_dependence",
        "sharpe_ratio",
        "var_backtest",
        "turnover"
    ])
    portfolio_weights: str = "equal"
    risk_free_rate: float = 0.02

@dataclass
class Config:
    \"\"\"Main configuration class.\"\"\"
    data: DataConfig = field(default_factory=DataConfig)
    factors: FactorConfig = field(default_factory=FactorConfig)
    latent: LatentConfig = field(default_factory=LatentConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    residuals: ResidualConfig = field(default_factory=ResidualConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    data_dir: str = "data"
    model_dir: str = "models"
    output_dir: str = "outputs"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, 'r', encoding='utf-8') as f:
            d = yaml.safe_load(f)
        return cls(
            data=DataConfig(**d.get('data', {})),
            factors=FactorConfig(**d.get('factors', {})),
            latent=LatentConfig(**d.get('latent', {})),
            diffusion=DiffusionConfig(**d.get('diffusion', {})),
            residuals=ResidualConfig(**d.get('residuals', {})),
            sampling=SamplingConfig(**d.get('sampling', {})),
            evaluation=EvaluationConfig(**d.get('evaluation', {})),
            data_dir=d.get('data_dir', 'data'),
            model_dir=d.get('model_dir', 'models'),
            output_dir=d.get('output_dir', 'outputs'),
        )

    def to_yaml(self, path: str) -> None:
        d = {
            'data': self.data.__dict__,
            'factors': self.factors.__dict__,
            'latent': self.latent.__dict__,
            'diffusion': self.diffusion.__dict__,
            'residuals': self.residuals.__dict__,
            'sampling': self.sampling.__dict__,
            'evaluation': self.evaluation.__dict__,
            'data_dir': self.data_dir,
            'model_dir': self.model_dir,
            'output_dir': self.output_dir,
        }
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(d, f, allow_unicode=True, default_flow_style=False)

def load_config(path: Optional[str] = None) -> Config:
    if path is None or not Path(path).exists():
        return Config()
    return Config.from_yaml(path)
"""

pathlib.Path('diffusion_factor_model/config/config.py').write_text(code)
print("Created config.py")
