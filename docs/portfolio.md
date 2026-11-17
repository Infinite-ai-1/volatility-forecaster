# CV and website wording

## CV

Implemented a purged walk-forward pipeline for next-week SPY realised volatility, comparing four statistical baselines with gradient boosting across 12 development folds and a held-out year; added leakage tests, regime analysis and an overlap-adjusted forecast comparison.

## Website

Volatility Forecaster asks whether daily market information can improve a forecast of the following week's realised volatility. It uses a fixed Yahoo Finance SPY snapshot, expanding validation windows, five-day purges and a final holdout. Reports include every fold, baseline comparisons, volatility-regime errors and a Diebold-Mariano test. The result favours boosting on point estimates, with weak evidence for a dependable gain. It is forecasting research, not a trading or profitability claim.

## Before discussing it in an interview

- Trace the six closes used for one target in `src/target.py` and its hand-calculation test.
- Explain why the five-row purge removes training origins rather than all recent price history.
- Walk through the truncation and final-period tampering tests.
- Compare QLIKE with percentage-point RMSE and MAE.
- Explain why the development ML scores benefit from hyperparameter selection.
- Read the losing folds and the DM assumptions, not just the final table.
- Keep descriptions of personal contributions and assistance accurate.
