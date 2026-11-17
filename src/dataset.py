import numpy as np
import pandas as pd

from src.config import HORIZON, LOOKBACK, Settings
from src.features import build_features
from src.splits import holdout_indices
from src.target import forward_realised_volatility


def modelling_dates(prices: pd.DataFrame, settings: Settings) -> pd.DatetimeIndex:
    if settings.horizon != HORIZON or settings.lookback != LOOKBACK:
        raise ValueError("This experiment fixes a five-day horizon and 63-row warm-up")
    if settings.gap < settings.horizon or settings.calendar_features != "weekday only":
        raise ValueError("The purge must cover the horizon; only weekday is configured")
    return prices.index[settings.lookback : -settings.horizon]


def prepare_dataset(
    prices: pd.DataFrame, settings: Settings, development_only: bool
) -> tuple[pd.DataFrame, pd.Series]:
    origins = modelling_dates(prices, settings)
    development, _, _ = holdout_indices(len(origins), settings.test_size, settings.gap)
    selected_dates = origins[development] if development_only else origins
    source = prices
    if development_only:
        # Only dates and row counts from the holdout are needed to locate the split.
        # Development labels end before its first forecast origin.
        last_label_position = (
            prices.index.get_loc(selected_dates[-1]) + settings.horizon
        )
        source = prices.iloc[: last_label_position + 1]
    features = build_features(source, settings.ewma_decay).loc[selected_dates]
    target = forward_realised_volatility(source, settings.horizon).loc[selected_dates]
    if (
        not np.isfinite(features.to_numpy()).all()
        or not np.isfinite(target.to_numpy()).all()
    ):
        raise ValueError(
            "Modelling rows must have complete features and five-day targets"
        )
    return features, target
