import json
from unittest.mock import patch

import pandas as pd
import pytest

from src.config import RunConfig
from src.reproduce import assert_same_results, verify_reproduction
from src.run_backtest import run_development
from src.run_final_test import run_final_test


def test_complete_experiment_and_frozen_reproduction(tmp_path, prices):
    raw = tmp_path / "data/raw/prices.csv"
    raw.parent.mkdir(parents=True)
    prices.to_csv(raw, lineterminator="\n")
    config = RunConfig(data_path=raw, output_root=tmp_path)
    grid = [
        {
            "learning_rate": 0.1,
            "max_iter": 8,
            "max_leaf_nodes": 8,
            "min_samples_leaf": 20,
        }
    ]
    selection = run_development(config, grid)
    assert "test_scores" not in selection
    assert selection["settings"]["gap"] == 5
    assert selection["splits"]["test_rows"] == 252
    with patch(
        "src.run_backtest.score_development_ml",
        side_effect=AssertionError("Unexpected refit"),
    ):
        assert run_development(config, grid) == selection
        # Simulate an interruption after checkpointing but before publishing selection.
        (config.reports / "development.json").unlink()
        assert run_development(config, grid) == selection
    results = run_final_test(config)
    assert results["final_fit_rows"] == selection["splits"]["development_rows"]
    assert (
        results["diebold_mariano"]["model_b"]
        == selection["best_baseline_by_development_qlike"]
    )
    assert sum(group["rows"] for group in results["error_by_regime"].values()) == 252
    folds = pd.read_csv(config.reports / "fold_scores.csv")
    assert folds.groupby("fold").model.nunique().eq(5).all()
    assert (
        pd.to_datetime(folds.last_train_label_end)
        < pd.to_datetime(folds.validation_start)
    ).all()
    verify_reproduction(config)
    assert json.loads((config.reports / "metrics.json").read_text()) == results
    with pytest.raises(FileExistsError):
        run_development(config, grid)
    with pytest.raises(FileExistsError):
        run_final_test(config)
    raw.write_text(raw.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="data"):
        verify_reproduction(config)


def test_changed_result_is_not_silently_accepted():
    with pytest.raises(ValueError, match="qlike"):
        assert_same_results({"qlike": 0.3}, {"qlike": 0.31})
