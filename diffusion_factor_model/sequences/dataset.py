"""Chronological factor-state sequence construction without transformation leakage."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

if __package__ and __package__.startswith("diffusion_factor_model."):
    from ..factors.extractor import FactorExtractor
    from ..latent.parametrizer import LatentParametrizer
    from ..reconstruction.reconstructor import ReturnReconstructor
else:
    from factors.extractor import FactorExtractor
    from latent.parametrizer import LatentParametrizer
    from reconstruction.reconstructor import ReturnReconstructor


@dataclass
class WindowSet:
    context: np.ndarray
    target: np.ndarray
    context_dates: np.ndarray
    target_dates: np.ndarray

    def __len__(self):
        return len(self.context)


@dataclass
class SequenceSplits:
    states: pd.DataFrame
    train_states: pd.DataFrame
    validation_states: pd.DataFrame
    test_states: pd.DataFrame
    train: WindowSet
    validation: WindowSet
    test: WindowSet
    train_returns: pd.DataFrame
    validation_returns: pd.DataFrame
    test_returns: pd.DataFrame
    train_market: pd.Series
    innovations_train: pd.DataFrame
    extractor: FactorExtractor
    parametrizer: LatentParametrizer
    reconstructor: ReturnReconstructor
    latent_metadata: dict


class FactorStateSequenceBuilder:
    """Fit Phase 1 mappings on train, then create disjoint temporal windows."""

    def __init__(
        self,
        config,
        context_length=60,
        horizon=20,
        train_fraction=0.70,
        validation_fraction=0.15,
        evaluation_stride=None,
        train_end_date=None,
        validation_end_date=None,
        allow_empty_test=False,
    ):
        if context_length < 1 or horizon < 1:
            raise ValueError("context_length and horizon must be positive.")
        if not 0 < train_fraction < 1:
            raise ValueError("train_fraction must lie in (0, 1).")
        if not 0 <= validation_fraction < 1 - train_fraction:
            raise ValueError("validation_fraction leaves no test observations.")
        self.config = config
        self.context_length = context_length
        self.horizon = horizon
        self.train_fraction = train_fraction
        self.validation_fraction = validation_fraction
        self.evaluation_stride = evaluation_stride or horizon
        self.train_end_date = train_end_date
        self.validation_end_date = validation_end_date
        self.allow_empty_test = allow_empty_test
        if (train_end_date is None) != (validation_end_date is None):
            raise ValueError(
                "train_end_date and validation_end_date must be provided together."
            )
        if train_end_date and (
            pd.to_datetime(train_end_date) >= pd.to_datetime(validation_end_date)
        ):
            raise ValueError("train_end_date must precede validation_end_date.")

    def build(self, returns, market_returns):
        market_returns = market_returns.reindex(returns.index)
        if market_returns.isna().any():
            raise ValueError("Market returns do not cover the complete return index.")
        if len(returns) < 500:
            raise ValueError("At least 500 observations are required for sequence splits.")

        n = len(returns)
        if self.train_end_date:
            train_cutoff_requested = pd.to_datetime(self.train_end_date)
            validation_cutoff_requested = pd.to_datetime(
                self.validation_end_date
            )
            train_returns = returns.loc[
                returns.index <= train_cutoff_requested
            ]
            validation_returns = returns.loc[
                (returns.index > train_cutoff_requested)
                & (returns.index <= validation_cutoff_requested)
            ]
            test_returns = returns.loc[
                returns.index > validation_cutoff_requested
            ]
        else:
            train_end = int(n * self.train_fraction)
            validation_end = int(
                n * (self.train_fraction + self.validation_fraction)
            )
            train_returns = returns.iloc[:train_end]
            validation_returns = returns.iloc[train_end:validation_end]
            test_returns = returns.iloc[validation_end:]
        required_lengths = [len(train_returns), len(validation_returns)]
        if not self.allow_empty_test:
            required_lengths.append(len(test_returns))
        if min(required_lengths) == 0:
            raise ValueError(
                "Required train, validation, and test periods must be non-empty."
            )
        train_market = market_returns.reindex(train_returns.index)
        future_returns = pd.concat(
            [validation_returns, test_returns]
        ).sort_index()
        future_market = market_returns.reindex(future_returns.index)

        extractor = FactorExtractor(self.config)
        train_mean = extractor.fit_mean_factors(train_returns, train_market)
        train_residuals = extractor.compute_residuals(train_returns, train_mean)
        train_vol = extractor.fit_volatility_factors(train_residuals)

        parametrizer = LatentParametrizer(self.config)
        train_latent, metadata = parametrizer.fit_transform(
            train_mean, train_vol
        )
        train_states = pd.DataFrame(
            train_latent,
            index=metadata["dates"],
            columns=metadata["mean_names"] + metadata["vol_names"],
        )
        innovations_train = extractor.compute_idiosyncratic(
            train_residuals, train_vol
        )

        future_mean = extractor.transform_mean_factors(
            future_returns,
            future_market,
            history_returns=train_returns,
            history_market=train_market,
        )
        future_residuals = extractor.compute_residuals(
            future_returns, future_mean
        )
        future_vol = extractor.transform_volatility_factors(
            future_residuals,
            history_residuals=train_residuals,
        )
        future_latent, future_metadata = parametrizer.transform(
            future_mean, future_vol
        )
        future_states = pd.DataFrame(
            future_latent,
            index=future_metadata["dates"],
            columns=train_states.columns,
        )
        states = pd.concat([train_states, future_states]).sort_index()

        train_cutoff = train_returns.index[-1]
        validation_cutoff = validation_returns.index[-1]
        validation_states = states[
            (states.index > train_cutoff)
            & (states.index <= validation_cutoff)
        ]
        test_states = states[states.index > validation_cutoff]
        train_states = states[states.index <= train_cutoff]

        reconstructor = ReturnReconstructor(
            self.config,
            train_mean,
            train_vol,
            parametrizer=parametrizer,
            latent_metadata=metadata,
        )
        return SequenceSplits(
            states=states,
            train_states=train_states,
            validation_states=validation_states,
            test_states=test_states,
            train=self._make_windows(train_states, stride=1),
            validation=self._make_windows(
                validation_states, stride=self.evaluation_stride
            ),
            test=self._make_windows(
                test_states, stride=self.evaluation_stride
            ),
            train_returns=train_returns,
            validation_returns=validation_returns,
            test_returns=test_returns,
            train_market=train_market,
            innovations_train=innovations_train,
            extractor=extractor,
            parametrizer=parametrizer,
            reconstructor=reconstructor,
            latent_metadata=metadata,
        )

    def _make_windows(self, states, stride):
        width = self.context_length + self.horizon
        starts = range(0, max(len(states) - width + 1, 0), stride)
        contexts = []
        targets = []
        context_dates = []
        target_dates = []
        values = states.values
        dates = states.index.to_numpy()
        for start in starts:
            split = start + self.context_length
            stop = split + self.horizon
            contexts.append(values[start:split])
            targets.append(values[split:stop])
            context_dates.append(dates[start:split])
            target_dates.append(dates[split:stop])

        state_dim = states.shape[1]
        return WindowSet(
            context=np.asarray(contexts, dtype=float).reshape(
                -1, self.context_length, state_dim
            ),
            target=np.asarray(targets, dtype=float).reshape(
                -1, self.horizon, state_dim
            ),
            context_dates=np.asarray(context_dates).reshape(
                -1, self.context_length
            ),
            target_dates=np.asarray(target_dates).reshape(-1, self.horizon),
        )
