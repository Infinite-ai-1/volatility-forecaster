import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.config import EPSILON, SEED


def parameter_grid() -> list[dict]:
    names = ("learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf")
    values = itertools.product([0.03, 0.1], [100, 300], [8, 16], [20, 50])
    return [dict(zip(names, row, strict=True)) for row in values]


class LogVolatilityRegressor:
    def __init__(self, parameters: dict, seed: int = SEED):
        self.parameters = parameters
        self.seed = seed

    def fit(
        self, features: pd.DataFrame, target: pd.Series
    ) -> "LogVolatilityRegressor":
        values = np.asarray(target, dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Training targets must be finite and non-negative")
        self.estimator_ = HistGradientBoostingRegressor(
            loss="squared_error",
            early_stopping=False,
            random_state=self.seed,
            **self.parameters,
        ).fit(features, np.log(np.maximum(values, EPSILON)))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        predictions = np.exp(self.estimator_.predict(features))
        if not np.isfinite(predictions).all() or (predictions <= 0).any():
            raise ValueError("Model produced invalid volatility forecasts")
        return predictions
