import math

import numpy as np
import pytest

from src.metrics import diebold_mariano, qlike, score_forecasts


def test_qlike_is_zero_when_perfect_and_handles_zeros():
    assert qlike([0.1, 0.2], [0.1, 0.2]) == 0
    assert qlike([0], [0]) == 0


def test_qlike_penalises_under_prediction_more():
    assert qlike([0.2], [0.1]) == pytest.approx(4 - math.log(4) - 1)
    assert qlike([0.2], [0.1]) > qlike([0.2], [0.4])


def test_error_metrics_are_in_percentage_points():
    scores = score_forecasts([0.1, 0.3], [0.2, 0.1])
    assert scores["mae"] == pytest.approx(15)
    assert scores["rmse"] == pytest.approx(100 * math.sqrt(0.025))


@pytest.mark.parametrize(
    "actual,forecast", [([], []), ([1], [1, 2]), ([np.nan], [1]), ([1], [-1])]
)
def test_invalid_metric_inputs_rejected(actual, forecast):
    with pytest.raises(ValueError):
        qlike(actual, forecast)


def test_dm_lag_zero_matches_hand_calculation():
    statistic, p_value = diebold_mariano([1, 2, 3, 4, 5], [0] * 5, lag=0)
    expected = 3 / math.sqrt(2 / 5)
    assert statistic == pytest.approx(expected)
    assert p_value == pytest.approx(math.erfc(expected / math.sqrt(2)))


def test_dm_uses_bartlett_covariances_and_correct_sign():
    differences = np.array([1.0, -0.2, 0.6, 1.5, -0.1, 0.9, 0.2, 1.1])
    centred = differences - differences.mean()
    distances = np.abs(np.arange(8)[:, None] - np.arange(8)[None, :])
    weights = np.maximum(1 - distances / 5, 0)
    long_run_variance = centred @ weights @ centred / 8
    expected = differences.mean() / np.sqrt(long_run_variance / 8)
    statistic, p_value = diebold_mariano(differences, np.zeros(8), lag=4)
    reverse, reverse_p = diebold_mariano(np.zeros(8), differences, lag=4)
    assert statistic == pytest.approx(expected)
    assert reverse == pytest.approx(-statistic)
    assert reverse_p == pytest.approx(p_value)


def test_dm_identical_losses_are_a_tie():
    assert diebold_mariano(np.ones(20), np.ones(20)) == (0.0, 1.0)
