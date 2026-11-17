import argparse
import json
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from src.baselines import baseline_forecasts
from src.config import MODEL_NAMES, MODEL_THREADS, RAW_PATH, ROOT, RunConfig
from src.data import load_prices
from src.dataset import prepare_dataset
from src.metrics import diebold_mariano, qlike_losses, score_forecasts
from src.models import LogVolatilityRegressor
from src.plots import plot_regime_errors, plot_test_forecasts
from src.research import check_context, experiment_context, write_json
from src.splits import holdout_indices


def compute_final(
    prices: pd.DataFrame, config: RunConfig, selection: dict
) -> tuple[dict, pd.DataFrame]:
    features, target = prepare_dataset(prices, config.settings, development_only=False)
    development, _, test = holdout_indices(
        len(features), config.settings.test_size, config.settings.gap
    )
    with threadpool_limits(limits=MODEL_THREADS):
        forecasts = baseline_forecasts(
            features.iloc[development], target.iloc[development], features.iloc[test]
        )
        model = LogVolatilityRegressor(
            selection["chosen_parameters"], config.settings.seed
        )
        model.fit(features.iloc[development], target.iloc[development])
        forecasts["hgb"] = model.predict(features.iloc[test])
    actual = target.iloc[test]
    predictions = pd.DataFrame({"realised": actual, **forecasts}, index=actual.index)
    test_scores = {
        name: score_forecasts(actual, predictions[name]) for name in MODEL_NAMES
    }
    cutoffs = selection["regime_cutoffs"]
    low, high = cutoffs["low_upper"], cutoffs["medium_upper"]
    buckets = {
        "low": actual <= low,
        "medium": (actual > low) & (actual <= high),
        "high": actual > high,
    }
    regimes = {}
    for bucket, mask in buckets.items():
        regimes[bucket] = {
            "rows": int(mask.sum()),
            "models": {
                name: score_forecasts(actual.loc[mask], predictions.loc[mask, name])
                if mask.any()
                else None
                for name in MODEL_NAMES
            },
        }
    baseline = selection["best_baseline_by_development_qlike"]
    try:
        statistic, p_value = diebold_mariano(
            qlike_losses(actual, predictions.hgb),
            qlike_losses(actual, predictions[baseline]),
            config.settings.dm_lag,
        )
        dm = {"estimable": True, "statistic": statistic, "p_value": p_value}
    except ValueError as error:
        dm = {"estimable": False, "reason": str(error)}
    dm.update(
        {
            "model_a": "hgb",
            "model_b": baseline,
            "loss": "per-row QLIKE",
            "newey_west_lag": config.settings.dm_lag,
            "positive_statistic_means": "gradient boosting has higher loss",
            "reference_distribution": "asymptotic standard normal, two-sided",
        }
    )
    metadata = {
        key: value for key, value in selection.items() if key != "candidate_grid"
    }
    metadata.update(
        {
            "candidate_count": len(selection["candidate_grid"]),
            "test_scores": test_scores,
            "error_by_regime": regimes,
            "diebold_mariano": dm,
            "non_overlapping_five_day_blocks": len(test) // config.settings.horizon,
            "final_fit_rows": len(development),
            "forecast_policy": "Fixed HAR and HGB fits; "
            "trailing inputs update at each observed close",
        }
    )
    return metadata, predictions


def results_markdown(metadata: dict) -> str:
    split = metadata["splits"]
    rows = [
        "# Final test results",
        "",
        f"Forecast origins: {split['test_start']} to {split['test_end']}. "
        f"Rows: {split['test_rows']}. Final label ends {split['test_label_end']}.",
        "",
        "| Model | QLIKE | RMSE (volatility points) | MAE (volatility points) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in MODEL_NAMES:
        score = metadata["test_scores"][name]
        rows.append(
            f"| {name} | {score['qlike']:.6f} | "
            f"{score['rmse']:.4f} | {score['mae']:.4f} |"
        )
    wins = metadata["win_counts"]
    rows.extend(
        [
            "",
            f"Development: HGB beats EWMA on QLIKE in {wins['hgb_vs_ewma']} "
            f"of {wins['folds']} folds. It beats the best baseline within each "
            f"fold in {wins['hgb_vs_best_baseline_in_each_fold']} folds.",
            "",
            "The baseline for the final DM comparison was chosen "
            "on development mean QLIKE: "
            f"`{metadata['best_baseline_by_development_qlike']}`.",
            "",
            "MAE and RMSE use percentage points: an error of 0.01 in fractional "
            "annualised volatility is one point. QLIKE uses variance ratios. "
            "Daily five-day targets overlap, so the test period contains only "
            f"about {metadata['non_overlapping_five_day_blocks']} "
            "non-overlapping weeks.",
            "",
            "This is not investment advice, a trading strategy "
            "or a profitability claim.",
        ]
    )
    return "\n".join(rows) + "\n"


def run_final_test(config: RunConfig) -> dict:
    destination = config.reports / "metrics.json"
    if destination.exists():
        raise FileExistsError(
            "Final test already evaluated. Published results will not be overwritten"
        )
    selection = json.loads((config.reports / "development.json").read_text())
    prices = load_prices(config.data_path)
    check_context(selection, experiment_context(prices, config))
    metadata, predictions = compute_final(prices, config, selection)
    plot_test_forecasts(predictions, config.reports / "figures")
    plot_regime_errors(metadata["error_by_regime"], config.reports / "figures")
    predictions.to_csv(
        config.reports / "test_predictions.csv",
        index_label="date",
        lineterminator="\n",
        float_format="%.12g",
    )
    text = results_markdown(metadata)
    (config.reports / "results.md").write_text(text, encoding="utf-8")
    write_json(destination, metadata)
    print(text)
    print(json.dumps(metadata["diebold_mariano"], indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen model choices on the final holdout"
    )
    parser.add_argument("--data", type=Path, default=RAW_PATH)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    arguments = parser.parse_args()
    run_final_test(RunConfig(arguments.data, arguments.output_root))


if __name__ == "__main__":
    main()
