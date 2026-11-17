# Final test results

Forecast origins: 2025-11-07 to 2026-11-09. Rows: 252. Final label ends 2026-11-16.

| Model | QLIKE | RMSE (volatility points) | MAE (volatility points) |
| --- | ---: | ---: | ---: |
| prev_5 | 0.838669 | 6.7868 | 4.8033 |
| rv_21 | 0.517881 | 6.3519 | 4.6511 |
| ewma | 0.463336 | 6.0474 | 4.4233 |
| har | 0.542763 | 5.9556 | 4.0665 |
| hgb | 0.404929 | 5.2868 | 3.5731 |

Development: HGB beats EWMA on QLIKE in 8 of 12 folds. It beats the best baseline within each fold in 8 folds.

The baseline for the final DM comparison was chosen on development mean QLIKE: `ewma`.

MAE and RMSE use percentage points: an error of 0.01 in fractional annualised volatility is one point. QLIKE uses variance ratios. Daily five-day targets overlap, so the test period contains only about 50 non-overlapping weeks.

This is not investment advice, a trading strategy or a profitability claim.
