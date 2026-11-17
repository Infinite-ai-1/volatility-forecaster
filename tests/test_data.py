import numpy as np
import pandas as pd
import pytest

from src.data import load_prices, validate_prices
from src.fetch_data import normalise_download


def test_valid_prices(prices):
    assert validate_prices(prices) == []


@pytest.mark.parametrize(
    "column,value",
    [
        ("adj_close", 0),
        ("close", -1),
        ("high", np.nan),
        ("volume", 0),
        ("open", np.inf),
    ],
)
def test_bad_prices_rejected(prices, column, value):
    changed = prices.copy()
    changed.loc[changed.index[0], column] = value
    with pytest.raises(ValueError, match="finite and positive"):
        validate_prices(changed)


def test_duplicate_and_out_of_order_dates_rejected(prices):
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_prices(pd.concat([prices.iloc[:1], prices]))
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_prices(prices.iloc[::-1])


def test_missing_column_and_inverted_range_rejected(prices):
    with pytest.raises(ValueError, match="Missing"):
        validate_prices(prices.drop(columns="adj_close"))
    changed = prices.copy()
    changed.loc[changed.index[0], "high"] = changed.low.iloc[0] / 2
    with pytest.raises(ValueError, match="High"):
        validate_prices(changed)


def test_long_calendar_gap_warns_and_continues(prices):
    changed = prices.iloc[[0, 10]].copy()
    with pytest.warns(UserWarning, match="Calendar gap"):
        gaps = validate_prices(changed)
    assert gaps[0]["days"] > 5


def test_csv_round_trip(tmp_path, prices):
    path = tmp_path / "prices.csv"
    prices.to_csv(path)
    loaded = load_prices(path)
    np.testing.assert_allclose(loaded.adj_close, prices.adj_close)
    assert loaded.index.equals(prices.index)


def test_download_layout_is_flattened(prices):
    frame = prices.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adj_close": "Adj Close",
            "volume": "Volume",
        }
    )
    frame.columns = pd.MultiIndex.from_product([frame.columns, ["SPY"]])
    normalised = normalise_download(frame)
    pd.testing.assert_frame_equal(normalised, prices, check_freq=False)


def test_adjusted_close_is_required(prices):
    frame = prices.rename(columns=str.title)
    with pytest.raises(ValueError, match="exactly once"):
        normalise_download(frame)


def test_empty_download_fails():
    with pytest.raises(RuntimeError, match="no daily bars"):
        normalise_download(pd.DataFrame())
