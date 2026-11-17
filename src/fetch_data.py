import argparse
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import END_DATE, RAW_PATH, ROOT, START_DATE, TICKER
from src.data import load_prices, price_summary, validate_prices

SOURCE_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")


def normalise_download(download: pd.DataFrame) -> pd.DataFrame:
    if download is None or download.empty:
        raise RuntimeError(
            "Yahoo returned no daily bars. No alternative source is substituted"
        )
    result = download.copy()
    if isinstance(result.columns, pd.MultiIndex):
        levels = [
            i
            for i in range(result.columns.nlevels)
            if set(SOURCE_COLUMNS).issubset(result.columns.get_level_values(i))
        ]
        if len(levels) != 1:
            raise ValueError("Cannot identify Yahoo's OHLCV column level")
        result.columns = result.columns.get_level_values(levels[0])
    if result.columns.has_duplicates or not set(SOURCE_COLUMNS).issubset(
        result.columns
    ):
        raise ValueError(f"Yahoo must return each of {SOURCE_COLUMNS} exactly once")
    result = result[list(SOURCE_COLUMNS)].rename(
        columns=lambda column: column.lower().replace(" ", "_")
    )
    result.index = pd.DatetimeIndex(result.index).tz_localize(None)
    result.index.name = "date"
    validate_prices(result)
    return result


def fetch_prices(output: Path = RAW_PATH) -> dict:
    if output.exists():
        summary = price_summary(load_prices(output), output)
        print("Keeping the existing snapshot:", json.dumps(summary, indent=2))
        return summary
    yf.set_tz_cache_location(str(ROOT / ".cache/yfinance"))
    exclusive_end = (date.fromisoformat(END_DATE) + timedelta(days=1)).isoformat()
    download = yf.download(
        TICKER,
        start=START_DATE,
        end=exclusive_end,
        auto_adjust=False,
        back_adjust=False,
        actions=False,
        interval="1d",
        progress=False,
        threads=False,
        multi_level_index=True,
        keepna=True,
        timeout=30,
    )
    prices = normalise_download(download)
    output.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(
        output, date_format="%Y-%m-%d", lineterminator="\n", float_format="%.12g"
    )
    saved = load_prices(output)
    metadata = {
        "source": "Yahoo Finance via yfinance",
        "ticker": TICKER,
        "requested_start": START_DATE,
        "requested_end_inclusive": END_DATE,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "yfinance_version": yf.__version__,
        **price_summary(saved, output),
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the fixed SPY daily snapshot")
    parser.add_argument("--output", type=Path, default=RAW_PATH)
    fetch_prices(parser.parse_args().output)


if __name__ == "__main__":
    main()
