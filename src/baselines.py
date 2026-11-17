import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.config import EPSILON

HAR_COLUMNS = ("rv_5", "rv_21", "rv_63")


class HARRegressor:
    def fit(self, features: pd.DataFrame, target: pd.Series) -> "HARRegressor":
        x = np.log(np.maximum(features[list(HAR_COLUMNS)].to_numpy(), EPSILON))
        y = np.log(np.maximum(np.asarray(target), EPSILON))
        self.estimator_ = LinearRegression().fit(x, y)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        x = np.log(np.maximum(features[list(HAR_COLUMNS)].to_numpy(), EPSILON))
        return np.exp(self.estimator_.predict(x))


def baseline_forecasts(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    forecast_features: pd.DataFrame,
) -> dict[str, np.ndarray]:
    har = HARRegressor().fit(train_features, train_target)
    return {
        "prev_5": forecast_features.rv_5.to_numpy(),
        "rv_21": forecast_features.rv_21.to_numpy(),
        "ewma": forecast_features.ewma_vol.to_numpy(),
        "har": har.predict(forecast_features),
    }
