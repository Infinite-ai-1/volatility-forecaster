import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_COLUMNS = ("open", "high", "low", "close", "adj_close")
REQUIRED_COLUMNS = (*PRICE_COLUMNS, "volume")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_prices(prices: pd.DataFrame) -> list[dict]:
    missing = set(REQUIRED_COLUMNS) - set(prices.columns)
    if missing:
        raise ValueError(f"Missing price columns: {sorted(missing)}")
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans:
        raise ValueError("Dates must form a non-missing DatetimeIndex")
    if (
        prices.empty
        or not prices.index.is_monotonic_increasing
        or prices.index.has_duplicates
    ):
        raise ValueError("Dates must be strictly increasing with no duplicates")
    values = prices[list(REQUIRED_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Prices and volume must be finite and positive")
    if (prices.high < prices.low).any():
        raise ValueError("High must be at least low")
    gaps = []
    for previous, current in zip(prices.index[:-1], prices.index[1:], strict=True):
        days = (current - previous).days
        if days > 5:
            gap = {
                "from": previous.date().isoformat(),
                "to": current.date().isoformat(),
                "days": days,
            }
            gaps.append(gap)
            warnings.warn(
                f"Calendar gap longer than five days: {gap}", UserWarning, stacklevel=2
            )
    return gaps


def load_prices(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"No price file at {path}. Run python -m src.fetch_data"
        )
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    validate_prices(frame)
    return frame


def price_summary(prices: pd.DataFrame, path: Path) -> dict:
    return {
        "rows": len(prices),
        "first_date": prices.index[0].date().isoformat(),
        "last_date": prices.index[-1].date().isoformat(),
        "sha256": file_sha256(path),
    }
