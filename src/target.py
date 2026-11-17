import numpy as np
import pandas as pd

from src.config import ANNUALISATION, HORIZON


def log_returns(prices: pd.DataFrame) -> pd.Series:
    return np.log(prices.adj_close / prices.adj_close.shift(1)).rename("return")


def forward_realised_volatility(
    prices: pd.DataFrame, horizon: int = HORIZON
) -> pd.Series:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    returns = log_returns(prices)
    future = pd.concat([returns.shift(-i) for i in range(1, horizon + 1)], axis=1)
    squares = future.pow(2).sum(axis=1, min_count=horizon)
    return np.sqrt(ANNUALISATION / horizon * squares).rename("target")
