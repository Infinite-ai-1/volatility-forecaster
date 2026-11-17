import numpy as np
import pandas as pd
import pytest

from src.baselines import baseline_forecasts
from src.features import build_features, ewma_volatility
from src.models import LogVolatilityRegressor
from src.target import forward_realised_volatility


def test_ewma_matches_hand_loop():
    returns = pd.Series(
        [0.01, -0.02, 0.03, -0.01, 0.02, 0.04, -0.03, 0.01, 0.02, -0.01]
    )
    result = ewma_volatility(returns, decay=0.94, seed_window=3)
    variance = float((returns.iloc[:3] ** 2).mean())
    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx(np.sqrt(252 * variance))
    for i in range(3, len(returns)):
        variance = 0.94 * variance + 0.06 * returns.iloc[i] ** 2
        assert result.iloc[i] == pytest.approx(np.sqrt(252 * variance))


def test_previous_week_equals_lagged_target(prices):
    features = build_features(prices)
    target = forward_realised_volatility(prices)
    np.testing.assert_allclose(
        features.rv_5.iloc[5:].to_numpy(),
        target.shift(5).iloc[5:].to_numpy(),
        rtol=1e-10,
    )


def test_all_models_produce_positive_forecasts(prices):
    features = build_features(prices).dropna()
    target = forward_realised_volatility(prices).reindex(features.index)
    x_train, y_train = features.iloc[:800], target.iloc[:800]
    x_test = features.iloc[800:900]
    forecasts = baseline_forecasts(x_train, y_train, x_test)
    model = LogVolatilityRegressor({"max_iter": 5, "max_leaf_nodes": 8}).fit(
        x_train, y_train
    )
    forecasts["hgb"] = model.predict(x_test)
    assert model.estimator_.early_stopping is False
    assert model.estimator_.n_iter_ == 5
    for forecast in forecasts.values():
        assert np.isfinite(forecast).all() and (forecast > 0).all()
