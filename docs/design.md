# Design

The question is whether information known at each close improves a forecast of SPY's next five trading days of realised volatility. Forecast quality is the outcome. No positions, returns or profit are backtested.

```mermaid
flowchart LR
    Prices[Fixed Yahoo daily snapshot] --> Validate[Schema and date checks]
    Validate --> Past[Past-only features]
    Validate --> Target[Five future returns as labels]
    Past --> Folds[Expanding development folds with purge]
    Target --> Folds
    Folds --> Baselines[Previous RV, EWMA and HAR-style OLS]
    Folds --> Search[Small boosting grid]
    Search --> Freeze[Freeze mean-QLIKE winner]
    Baselines --> Freeze
    Freeze --> Final[One final holdout evaluation]
    Final --> Reports[Fold tables, regimes, forecasts and DM test]
```

## Target and timing

The horizon is five observed trading rows, roughly a week. Five returns require six closes: the current close and the following five. The target uses adjusted-close log returns, the zero-mean sum of squares and annualisation by 252. The trailing RV features use the same convention. Subtracting a sample mean over five observations would change the quantity being forecast.

Features at close t include that day's return, range and volume. They never include a later bar. Their windows include the current observation unless the definition says otherwise, such as the volume denominator, which uses the previous 21 observations. EWMA begins only when its initial 21 returns are known. Earlier values stay missing instead of receiving a seed computed from later data. The first 63 raw rows are removed as feature warm-up; the final five have incomplete targets.

Adjusted prices are a revised snapshot rather than a point-in-time archive. A common multiplicative back-adjustment factor cancels in adjacent log-price ratios. This removes the mechanical effect of later dividend back-adjustments within earlier return intervals, but does not guarantee that every historical vendor correction is harmless. The raw checksum is frozen. Unadjusted high and low are used together; their within-day ratio is unchanged by a common adjustment factor.

## Purging and available labels

Random K-fold validation would train on later market conditions while evaluating earlier ones. Here each fold trains on an expanding history. Its validation block has 252 origins, advances by 252 rows and follows a five-origin purge. Minimum training length is 756 rows.

If validation begins at position v, the last training origin is v - 6. Its label ends at v - 1, before the first validation forecast. The gap removes training origins with overlapping labels. It does not remove observed market history from trailing features.

The same rule separates development from the final test. Split boundaries are located using dates and counts. During selection, feature and target construction receives only the price prefix whose final return precedes the first test origin. A test changes the entire final-period price path and checks that development features and labels remain identical.

The final 252 complete targets are reserved for evaluation. HAR and boosting are fitted once on all available development rows and held fixed throughout that period. Rolling statistics and input observations update at each close. Development rows after the last complete validation block contribute to this final fit, not to extra partial folds. Their count is recorded in the split metadata.

## Models and selection

Previous-five-day RV and 21-day RV copy trailing estimates. EWMA fixes its decay at 0.94 and seeds daily variance from the first 21 squared returns. HAR-style OLS fits the log target on log RV at 5, 21 and 63 days. It is a daily-return proxy for a HAR benchmark, not an intraday realised-volatility model.

Boosting uses the 34 documented features and squared-error loss on log volatility. Exponentiating predictions keeps forecasts positive. A small positive floor handles zero volatility before logs. No lognormal smearing correction is fitted, so these forecasts should not be described as unbiased conditional mean volatility.

The grid tries learning rates 0.03 and 0.1, 100 and 300 iterations, 8 and 16 leaves, and minimum leaf sizes 20 and 50. QLIKE is primary, selected before running the grid. Weekday is included; month is excluded. Calendar alternatives are not searched after seeing the test result.

The four baselines are evaluated before the ML grid. Boosting is chosen by mean QLIKE over the same development folds, then frozen in `reports/development.json`. These reported ML fold scores are optimistic because they also selected its parameters. Baselines have no corresponding hyperparameter search. The final holdout is the cleaner comparison.

`early_stopping=False` avoids the estimator's random internal validation subset. See the [scikit-learn regressor documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html). A small histogram model keeps this comparison manageable without adding a recurrent network or more data requirements.

## Metrics and the comparison test

QLIKE squares volatility forecasts and targets, then compares their variance ratio. Under-predicting volatility by half costs more than over-predicting it by double. MAE and RMSE are reported in annualised volatility percentage points: 0.01 fractional volatility equals one point.

All models use identical forecast origins within a fold and within the final test. Win counts include comparison against EWMA and against the best of the four baselines within each fold. The latter is a retrospective diagnostic, not a baseline that could be selected with advance knowledge.

Regime cutoffs are terciles of development targets. Final-test regimes are assigned from realised outcomes for error analysis only. A future regime label is never an input. The final DM baseline is selected by development mean QLIKE, not by final-year performance.

The DM loss difference is boosting QLIKE minus baseline QLIKE. A Bartlett Newey-West variance at lag four accounts for the mechanical overlap of five-day targets. The two-sided p-value uses an asymptotic normal reference. Longer-lived volatility dependence and a short holdout make this a rough diagnostic. A non-small p-value does not prove equal model quality.

## Reproduction

The raw CSV checksum, library versions, source fingerprint, window settings, selected parameters and development results are frozen together. Each completed grid candidate is checkpointed locally. Interrupted searches can resume if those inputs still match.

After final evaluation, direct search and test commands reject overwriting that experiment. `make reproduce` refits frozen choices, checks every fold, recomputes final metrics and forecasts, and compares them to four decimal places without modifying published reports. It does not search again. A different source snapshot or implementation belongs in a separate output directory. Raw quotes and checkpoints remain outside Git.

## Details that needed care

The hand-calculation wording in the blueprint mentioned five closes. The target needs six, and the unit test uses six. EWMA seeding also needed an explicit availability time to keep early rows causal. MAE and RMSE are multiplied by 100 so the tables mean percentage points rather than unlabelled fractions.

## Next steps

Pre-register another evaluation period, examine more assets and measure the effect of alternative training windows without reusing this holdout for selection. A risk application would also need point-in-time data, better realised-volatility measurements, monitoring and model review. No trading strategy follows from the forecast comparison.
