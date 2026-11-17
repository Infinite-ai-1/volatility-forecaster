import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.dataset import modelling_dates, prepare_dataset
from src.splits import holdout_indices, walk_forward_splits


def test_expanding_windows_purge_overlapping_labels():
    folds = list(walk_forward_splits(45, min_train=10, val_size=5, gap=5))
    lengths = []
    seen = set()
    for train, val in folds:
        lengths.append(len(train))
        assert train[0] == 0
        assert val[0] - train[-1] - 1 == 5
        assert train[-1] + 5 == val[0] - 1
        assert set(val).isdisjoint(seen)
        seen.update(val)
    assert lengths == sorted(set(lengths))


@pytest.mark.parametrize(
    "arguments",
    [
        (40, 0, 5, 5, None),
        (40, 10, 0, 5, None),
        (40, 10, 5, -1, None),
        (40, 10, 5, 5, 4),
    ],
)
def test_invalid_split_settings_rejected(arguments):
    with pytest.raises(ValueError):
        list(walk_forward_splits(*arguments))


def test_final_holdout_has_exact_purge():
    development, gap, test = holdout_indices(1000, 252, 5)
    assert len(test) == 252 and len(gap) == 5
    assert development[-1] + 5 == test[0] - 1
    np.testing.assert_array_equal(np.r_[development, gap, test], np.arange(1000))


def test_final_test_prices_cannot_affect_development(prices):
    settings = Settings()
    origins = modelling_dates(prices, settings)
    _, _, test = holdout_indices(len(origins), settings.test_size, settings.gap)
    cutoff = origins[test[0]]
    base_x, base_y = prepare_dataset(prices, settings, development_only=True)
    changed = prices.copy()
    changed.loc[cutoff:, :] *= 4
    changed_x, changed_y = prepare_dataset(changed, settings, development_only=True)
    pd.testing.assert_frame_equal(base_x, changed_x)
    pd.testing.assert_series_equal(base_y, changed_y)
    assert base_y.index[-1] < cutoff


def test_a_purge_shorter_than_horizon_is_rejected(prices):
    with pytest.raises(ValueError, match="purge"):
        prepare_dataset(prices, Settings(gap=4), development_only=True)
