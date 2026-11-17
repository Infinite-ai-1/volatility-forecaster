import numpy as np
import pandas as pd

from src.config import ANNUALISATION
from src.target import log_returns


def trailing_volatility(returns: pd.Series, window: int) -> pd.Series:
    if window < 1:
        raise ValueError("window must be positive")
    return np.sqrt(ANNUALISATION / window * returns.pow(2).rolling(window).sum())


def ewma_volatility(
    returns: pd.Series, decay: float = 0.94, seed_window: int = 21
) -> pd.Series:
    if not 0 < decay < 1 or seed_window < 1:
        raise ValueError("Use a decay in (0, 1) and a positive seed window")
    values = returns.to_numpy(dtype=float)
    output = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) < seed_window:
        return pd.Series(output, index=returns.index, name="ewma_vol")
    if not np.isfinite(values[valid[0] :]).all():
        raise ValueError(
            "Returns may only have missing values before the first observation"
        )
    seed_end = int(valid[seed_window - 1])
    variance = float(np.mean(values[valid[:seed_window]] ** 2))
    output[seed_end] = np.sqrt(ANNUALISATION * variance)
    for i in range(seed_end + 1, len(values)):
        variance = decay * variance + (1 - decay) * values[i] ** 2
        output[i] = np.sqrt(ANNUALISATION * variance)
    # Earlier rows stay missing: the seed itself was not known before seed_end.
    return pd.Series(output, index=returns.index, name="ewma_vol")


def build_features(prices: pd.DataFrame, decay: float = 0.94) -> pd.DataFrame:
    returns = log_returns(prices)
    features = pd.DataFrame(index=prices.index)
    for lag in range(21):
        features[f"return_lag_{lag}"] = returns.shift(lag)
    for window in (5, 10, 21, 63):
        features[f"rv_{window}"] = trailing_volatility(returns, window)
        features[f"mean_ret_{window}"] = returns.rolling(window).mean()
    features["ewma_vol"] = ewma_volatility(returns, decay)
    range_squared = np.log(prices.high / prices.low).pow(2)
    for window in (5, 21):
        features[f"range_{window}"] = np.sqrt(
            ANNUALISATION / (4 * np.log(2)) * range_squared.rolling(window).mean()
        )
    features["log_volume_ratio"] = np.log(
        prices.volume / prices.volume.shift(1).rolling(21).mean()
    )
    features["weekday"] = prices.index.day_of_week
    return features
