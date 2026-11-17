import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import RAW_PATH, ROOT, RunConfig
from src.data import load_prices
from src.dataset import prepare_dataset
from src.models import parameter_grid
from src.plots import plot_folds, plot_splits
from src.research import (
    check_context,
    experiment_context,
    fold_summary,
    score_development_baselines,
    score_development_ml,
    write_json,
)


def run_development(config: RunConfig, grid: list[dict] | None = None) -> dict:
    if (config.reports / "metrics.json").exists():
        raise FileExistsError("Final test exists. Start a new experiment to retune")
    prices = load_prices(config.data_path)
    context = experiment_context(prices, config)
    destination = config.reports / "development.json"
    grid = parameter_grid() if grid is None else grid
    if not grid:
        raise ValueError("The parameter grid cannot be empty")
    if destination.exists():
        selection = json.loads(destination.read_text())
        check_context(selection, context)
        if selection["candidate_grid"] != grid:
            raise ValueError("The frozen search grid differs")
        print("Development choices already frozen")
        return selection
    features, target = prepare_dataset(prices, config.settings, development_only=True)
    baseline_rows = score_development_baselines(features, target, config.settings)
    print("Baseline folds completed before ML search", flush=True)
    checkpoint_path = config.models / "development_checkpoint.json"
    checkpoint = {"context": context, "grid": grid, "candidates": []}
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        check_context(checkpoint["context"], context)
        if checkpoint["grid"] != grid:
            raise ValueError("The checkpoint parameter grid differs")
    for index, parameters in enumerate(grid):
        if index < len(checkpoint["candidates"]):
            continue
        rows = score_development_ml(features, target, config.settings, parameters)
        mean_qlike = float(pd.DataFrame(rows).qlike.mean())
        checkpoint["candidates"].append(
            {"parameters": parameters, "mean_qlike": mean_qlike, "rows": rows}
        )
        write_json(checkpoint_path, checkpoint)
        print(
            f"Candidate {index + 1}/{len(grid)}: "
            f"mean development QLIKE={mean_qlike:.6f}",
            flush=True,
        )
    selected_index = min(
        range(len(grid)), key=lambda i: checkpoint["candidates"][i]["mean_qlike"]
    )
    winner = checkpoint["candidates"][selected_index]
    scores = (
        pd.DataFrame(baseline_rows + winner["rows"])
        .sort_values(["fold", "model"])
        .reset_index(drop=True)
    )
    selection = {
        **context,
        "selection_metric": "mean development QLIKE",
        "candidate_grid": grid,
        "chosen_parameters": winner["parameters"],
        **fold_summary(scores),
        "regime_cutoffs": {
            "low_upper": float(target.quantile(1 / 3)),
            "medium_upper": float(target.quantile(2 / 3)),
        },
    }
    config.reports.mkdir(parents=True, exist_ok=True)
    scores.to_csv(
        config.reports / "fold_scores.csv",
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    )
    search_rows = [
        {"candidate": index + 1, **candidate["parameters"], **row}
        for index, candidate in enumerate(checkpoint["candidates"])
        for row in candidate["rows"]
    ]
    pd.DataFrame(search_rows).to_csv(
        config.reports / "grid_scores.csv",
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    )
    plot_splits(prices, config.settings, config.reports / "figures")
    plot_folds(scores, config.reports / "figures")
    write_json(destination, selection)
    print(scores[["fold", "model", "qlike", "rmse", "mae"]].to_string(index=False))
    print(json.dumps(selection["win_counts"], indent=2))
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run purged development folds and freeze model choices"
    )
    parser.add_argument("--data", type=Path, default=RAW_PATH)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    arguments = parser.parse_args()
    run_development(RunConfig(arguments.data, arguments.output_root))


if __name__ == "__main__":
    main()
