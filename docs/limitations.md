# Limitations

This is not investment advice or a trading strategy. It makes no claim about profitability. There are no positions, transaction costs, execution rules or profit backtests.

Only SPY is studied. A model ranking on one large US equity ETF does not establish performance on other assets, markets or sampling frequencies.

The target uses only five daily returns. It is a noisy volatility proxy, not an observed latent volatility or an intraday realised measure. The zero-mean convention is consistent across forward targets and trailing RV features, but daily sampling misses price variation inside each day. Range features add some intraday information without replacing a full intraday dataset.

The final test contains 252 overlapping daily five-day targets, about 50 non-overlapping weeks. Volatility clustering can reduce the effective independent information further. One year is weak evidence for or against a forecasting advantage.

The selected booster has lower final-test QLIKE than EWMA, but its development wins are not universal. The recorded DM p-value is 0.224942. That comparison does not establish a dependable improvement, and it does not establish equality either. The [recorded metrics](../reports/metrics.json) and [fold table](../reports/fold_scores.csv) are the evidence, including losing folds.

Boosting parameters were selected using the same development folds shown in the report, so those ML scores benefit from selection. No nested validation or multiple-asset replication is provided. The holdout was frozen before evaluation; changes proposed from its results require a genuinely new evaluation period.

The DM calculation uses a lag-four Bartlett Newey-West variance and an asymptotic normal reference. This addresses target overlap but may miss dependence at longer lags. There is no finite-sample correction. Regime groups also contain fewer observations than the full holdout. Their boundaries come from development, while their test membership is known only after outcomes occur.

Log-target squared-error training and QLIKE selection optimise different objectives. Exponentiating a log prediction does not by itself give an unbiased conditional mean volatility forecast. Under-prediction in high-volatility periods remains a practical concern.

Yahoo Finance is a free source with no data-quality or availability guarantee supplied by this project. Adjusted history can be revised after dividends and other corrections. The CSV is a downloaded snapshot, not archived point-in-time data. A future fetch may change the checksum and slightly change reported scores. Long calendar gaps produce warnings; they are not filled with invented bars.

A real risk system would need broader asset coverage, better realised-volatility estimates, point-in-time data controls, monitoring, failure handling and independent model review. A better volatility forecast alone says nothing about profitable trading.
