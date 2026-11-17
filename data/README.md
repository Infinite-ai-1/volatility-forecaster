# Data record

- Source: Yahoo Finance, fetched with yfinance.
- Ticker: SPY.
- Requested interval: 2010-01-01 through 2026-11-16, inclusive.
- Observed interval: 2010-01-04 through 2026-11-16.
- Downloaded at UTC: 2026-11-17T12:59:34.460100+00:00.
- yfinance version: 1.8.0.
- Rows: 4244.
- Canonical CSV: `data/raw/spy_daily.csv`.
- SHA-256: `28a02fb5b1853aa3bc14436055fc84837a45fbfd53be70c725db4f6241670b59`.

The file contains `date, open, high, low, close, adj_close, volume`. Prices are daily Yahoo bars, not simulated market observations. Raw quotes are ignored by Git; the report CSVs contain derived forecasts and scores.

## Fetch and verify

```sh
python -m src.fetch_data
python -m src.reproduce
```

`END_DATE` in `src/config.py` is fixed to the last completed US session before the first download. The [yfinance download API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html) treats `end` as exclusive, so the request adds one calendar day. `auto_adjust=False` requests both close and adjusted close. Multi-level columns are flattened, and missing OHLCV or adjusted-close columns cause an error.

An existing CSV is kept rather than overwritten. Validation rejects duplicate or unsorted dates, missing values, non-positive prices or volume, and high below low. Calendar gaps longer than five days are printed as warnings and retained. Prices are not forward-filled.

Adjusted closes can be revised when dividends are paid. Common adjustment factors cancel in adjacent log-return ratios, but vendor corrections and numerical rounding can still change later downloads, sometimes in the fourth decimal place of metrics. Exact reproduction requires the original CSV snapshot with the checksum above. The report also checks numeric-library versions and the implementation fingerprint.

## A separately labelled experiment

To fetch a fresh copy without changing the original snapshot:

```sh
python -m src.fetch_data --output data/raw/spy_refresh.csv
python -m src.run_backtest --data data/raw/spy_refresh.csv --output-root experiments/refreshed
python -m src.run_final_test --data data/raw/spy_refresh.csv --output-root experiments/refreshed
```

Or replace the last two commands with `python -m src.reproduce --data data/raw/spy_refresh.csv --output-root experiments/refreshed`. The first run searches development folds and evaluates the frozen winner. Later calls verify frozen choices. Directly rerunning search or final evaluation against a completed experiment is rejected.

Use a genuinely new holdout when making choices based on this project's test result. Relabelling an output folder does not make reused test information unseen.

Tests use a fixed-seed, volatility-clustered simulated series and make no provider requests. The simulator is a test fixture, not a substitute source for published SPY results.
