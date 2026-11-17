import numpy as np
import pandas as pd
import pytest

from src.features import build_features


def test_features_do_not_see_the_future(prices):
    full = build_features(prices)
    for cut in (10, 21, 63, 400, 800, 1200):
        truncated = build_features(prices.iloc[: cut + 1])
        pd.testing.assert_series_equal(
            full.iloc[cut], truncated.iloc[-1], check_names=False
        )


def test_future_tampering_cannot_change_past_features(prices):
    changed = prices.copy()
    changed.iloc[801:, :] *= 3
    pd.testing.assert_frame_equal(
        build_features(prices).iloc[:801], build_features(changed).iloc[:801]
    )


def test_volume_ratio_excludes_current_day(prices):
    row = build_features(prices).iloc[100]
    expected = np.log(prices.volume.iloc[100] / prices.volume.iloc[79:100].mean())
    assert row.log_volume_ratio == pytest.approx(expected)


def test_feature_windows_and_columns(prices):
    features = build_features(prices)
    assert features.dropna().index[0] == prices.index[63]
    assert len(features.columns) == 34
    assert "month" not in features
    assert features.iloc[100].weekday == prices.index[100].weekday()
