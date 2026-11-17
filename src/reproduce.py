import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RAW_PATH, ROOT, RunConfig
from src.data import load_prices
from src.research import check_context, experiment_context, frozen_development
from src.run_backtest import run_development
from src.run_final_test import compute_final, run_final_test


def assert_same_results(expected, actual, path: str = "metrics") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError(f"Different keys at {path}")
        for key in expected:
            assert_same_results(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, float):
        if not math.isclose(expected, actual, rel_tol=0, abs_tol=0.00005):
            raise ValueError(f"Different result at {path}: {expected} vs {actual}")
    elif expected != actual:
        raise ValueError(f"Different result at {path}: {expected} vs {actual}")


def verify_reproduction(config: RunConfig) -> None:
    prices = load_prices(config.data_path)
    selection = json.loads((config.reports / "development.json").read_text())
    check_context(selection, experiment_context(prices, config))
    scores, summary = frozen_development(prices, config, selection["chosen_parameters"])
    for field in summary:
        assert_same_results(selection[field], summary[field], field)
    expected_folds = pd.read_csv(config.reports / "fold_scores.csv")
    if list(expected_folds.columns) != list(scores.columns) or len(
        expected_folds
    ) != len(scores):
        raise ValueError("Fold table structure changed")
    for column in scores:
        if pd.api.types.is_numeric_dtype(scores[column]):
            np.testing.assert_allclose(
                expected_folds[column],
                scores[column],
                rtol=0,
                atol=0.00005,
                err_msg=f"Fold column changed: {column}",
            )
        elif expected_folds[column].tolist() != scores[column].tolist():
            raise ValueError(f"Fold column changed: {column}")
    updated_selection = {**selection, **summary}
    actual, predictions = compute_final(prices, config, updated_selection)
    expected = json.loads((config.reports / "metrics.json").read_text())
    assert_same_results(expected, actual)
    saved_predictions = pd.read_csv(
        config.reports / "test_predictions.csv", index_col="date", parse_dates=True
    )
    pd.testing.assert_index_equal(saved_predictions.index, predictions.index)
    np.testing.assert_allclose(
        saved_predictions.to_numpy(), predictions.to_numpy(), rtol=0, atol=0.00005
    )
    print(
        "Reproduction passed: all fold scores, final metrics and forecasts "
        "match to four decimal places"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce or verify the frozen experiment"
    )
    parser.add_argument("--data", type=Path, default=RAW_PATH)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    arguments = parser.parse_args()
    config = RunConfig(arguments.data, arguments.output_root)
    if (config.reports / "metrics.json").exists():
        verify_reproduction(config)
    else:
        run_development(config)
        run_final_test(config)


if __name__ == "__main__":
    main()
