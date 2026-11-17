import math

import numpy as np
from numpy.typing import ArrayLike

from src.config import EPSILON


def checked_arrays(
    realised_vol: ArrayLike, forecast_vol: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    actual, predicted = (
        np.asarray(realised_vol, dtype=float),
        np.asarray(forecast_vol, dtype=float),
    )
    if actual.ndim != 1 or predicted.shape != actual.shape or not len(actual):
        raise ValueError("Forecasts and targets must be non-empty matching vectors")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Forecasts and targets must be finite")
    if (actual < 0).any() or (predicted < 0).any():
        raise ValueError("Volatility cannot be negative")
    return actual, predicted


def qlike_losses(
    realised_vol: ArrayLike, forecast_vol: ArrayLike, eps: float = EPSILON
) -> np.ndarray:
    actual, predicted = checked_arrays(realised_vol, forecast_vol)
    if eps <= 0:
        raise ValueError("eps must be positive")
    ratio = np.maximum(actual**2, eps) / np.maximum(predicted**2, eps)
    return np.maximum(ratio - np.log(ratio) - 1, 0.0)


def qlike(
    realised_vol: ArrayLike, forecast_vol: ArrayLike, eps: float = EPSILON
) -> float:
    return float(qlike_losses(realised_vol, forecast_vol, eps).mean())


def score_forecasts(
    realised_vol: ArrayLike, forecast_vol: ArrayLike
) -> dict[str, float]:
    actual, predicted = checked_arrays(realised_vol, forecast_vol)
    error = actual - predicted
    return {
        "qlike": qlike(actual, predicted),
        "mae": float(100 * np.abs(error).mean()),
        "rmse": float(100 * np.sqrt(np.mean(error**2))),
    }


def diebold_mariano(
    loss_a: ArrayLike, loss_b: ArrayLike, lag: int = 4
) -> tuple[float, float]:
    """Use a Bartlett Newey-West variance; positive means A has higher loss."""
    a, b = np.asarray(loss_a, dtype=float), np.asarray(loss_b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or len(a) <= lag + 1 or lag < 0:
        raise ValueError("Loss vectors must match and contain more than lag + 1 rows")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Losses must be finite")
    difference = a - b
    if np.all(difference == 0):
        return 0.0, 1.0
    centred = difference - difference.mean()
    n = len(difference)
    long_run_variance = float(centred @ centred / n)
    for k in range(1, lag + 1):
        long_run_variance += (
            2 * (1 - k / (lag + 1)) * float(centred[k:] @ centred[:-k]) / n
        )
    if long_run_variance <= 0:
        raise ValueError("The long-run loss variance is not positive")
    statistic = float(difference.mean() / math.sqrt(long_run_variance / n))
    return statistic, math.erfc(abs(statistic) / math.sqrt(2))
