# Experiment decisions

The settings below were fixed before final-test evaluation. They record the run's assumptions and defaults.

| Choice | Setting | Reason |
| --- | --- | --- |
| Instrument and snapshot | SPY, 2010-01-01 through 2026-11-16 inclusive | Fix the last completed session before the first download. |
| Windows | Expanding, minimum train 756, validation 252, step 252 | Keep time order and compare full blocks on equal rows. |
| Purge | Five forecast origins before each validation block and the final test | Training labels must mature before the next forecast period. |
| Selection metric | Mean development QLIKE | Compare variance forecasts with an asymmetric penalty for under-prediction. |
| Calendar fields | Weekday only | Keep the proposed feature set small. No final-test-driven calendar search. |
| ML model | Histogram gradient boosting on log volatility | Positive forecasts and a small, inspectable grid. Internal early stopping is off. |
| Parameter storage | Frozen versioned development report | Keep measured choices separate from source configuration. |
| Final fits | HAR and boosting fit once on development, with the same purge | The test compares fixed models while past-only inputs update. |
| Statistical comparison | DM test, Newey-West lag four, two-sided normal reference | Account for five-day target overlap, while retaining small-sample caveats. |
| DM baseline | Lowest development mean QLIKE among the four baselines | Avoid choosing the comparator from final-test losses. |
| Regimes | Development-target terciles | Test outcomes assign error-analysis buckets only. |
| Evidence | Preserve completed reports and source checksum | Reproduction verifies frozen choices; new experiments use a new output root. |

No feature, model grid, window or calendar choice was changed in response to the final-test result. The fetch uses `auto_adjust=False` and an exclusive API end date one day after the inclusive configured end date.
