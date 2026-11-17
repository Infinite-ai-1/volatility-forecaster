import numpy as np
import pandas as pd
import pytest

from src.target import forward_realised_volatility


def test_target_matches_hand_calculation():
    returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    close = np.r_[100.0, 100 * np.exp(np.cumsum(returns))]
    frame = pd.DataFrame({"adj_close": close})
    target = forward_realised_volatility(frame)
    assert target.iloc[0] == pytest.approx(np.sqrt(252 / 5 * (returns**2).sum()))
    assert target.iloc[-5:].isna().all()


def test_target_uses_the_five_future_returns(prices):
    base = forward_realised_volatility(prices)
    t = 100
    for location, should_change in [
        (t - 1, False),
        (t, True),
        (t + 3, True),
        (t + 6, False),
    ]:
        changed = prices.copy()
        changed.iloc[location, changed.columns.get_loc("adj_close")] *= 1.1
        actual = forward_realised_volatility(changed).iloc[t]
        assert (not np.isclose(base.iloc[t], actual)) == should_change


def test_last_five_targets_are_missing(prices):
    target = forward_realised_volatility(prices)
    assert target.iloc[:-5].notna().all()
    assert target.iloc[-5:].isna().all()
