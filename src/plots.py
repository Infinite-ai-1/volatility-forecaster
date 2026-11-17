from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import LABELS, MODEL_NAMES, Settings
from src.dataset import modelling_dates
from src.splits import holdout_indices, walk_forward_splits


def save_figure(figure: plt.Figure, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(directory / name, dpi=150)
    plt.close(figure)


def plot_splits(prices: pd.DataFrame, settings: Settings, directory: Path) -> None:
    dates = modelling_dates(prices, settings)
    development, _, test = holdout_indices(len(dates), settings.test_size, settings.gap)
    folds = list(
        walk_forward_splits(
            len(development),
            settings.min_train,
            settings.validation_size,
            settings.gap,
            settings.step,
        )
    )
    figure, (axis, detail) = plt.subplots(
        1,
        2,
        figsize=(12, max(4, len(folds) * 0.36 + 1.5)),
        gridspec_kw={"width_ratios": [4, 1]},
        sharey=True,
    )
    for row, (train, val) in enumerate([*folds, (development, test)]):
        starts = [dates[0], dates[train[-1] + 1], dates[val[0]]]
        ends = [
            dates[train[-1] + 1],
            dates[val[0]],
            dates[val[-1]] + pd.Timedelta(days=1),
        ]
        labels = [
            "Train",
            "Purged origins",
            "Validation" if row < len(folds) else "Final test",
        ]
        colours = ["C0", "C1", "C2" if row < len(folds) else "C3"]
        for start, end, label, colour in zip(
            starts, ends, labels, colours, strict=True
        ):
            show_label = (row == 0 and label != "Final test") or label == "Final test"
            axis.barh(
                row,
                mdates.date2num(end) - mdates.date2num(start),
                left=mdates.date2num(start),
                height=0.65,
                color=colour,
                label=label if show_label else None,
            )
        detail.barh(row, 5, left=-settings.gap - 5, height=0.65, color="C0")
        detail.barh(row, settings.gap, left=-settings.gap, height=0.65, color="C1")
        detail.barh(
            row, 5, left=0, height=0.65, color="C2" if row < len(folds) else "C3"
        )
    axis.set_yticks(
        range(len(folds) + 1), [f"Fold {i + 1}" for i in range(len(folds))] + ["Final"]
    )
    axis.invert_yaxis()
    axis.xaxis_date()
    axis.xaxis.set_major_locator(mdates.YearLocator(2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.set(
        xlabel="Forecast-origin date",
        title=f"Expanding windows with {settings.gap}-row purges",
    )
    axis.legend(loc="upper left", ncols=4, fontsize=8)
    detail.set(title="Purge detail", xlabel="Rows from forecast start")
    detail.set_xticks([-settings.gap - 5, -settings.gap, 0, 5])
    detail.axvline(0, color="0.4", linewidth=0.5)
    save_figure(figure, directory, "splits.png")


def plot_folds(scores: pd.DataFrame, directory: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    for name in MODEL_NAMES:
        rows = scores.loc[scores.model == name].sort_values("fold")
        axis.plot(
            pd.to_datetime(rows.validation_start),
            rows.qlike,
            marker="o",
            label=LABELS[name],
        )
    axis.set(
        xlabel="Validation block start",
        ylabel="QLIKE (variance loss)",
        title="Development folds: lower QLIKE is better",
    )
    axis.legend(fontsize=9)
    save_figure(figure, directory, "fold_qlike.png")


def plot_test_forecasts(predictions: pd.DataFrame, directory: Path) -> None:
    figure, axes = plt.subplots(
        len(MODEL_NAMES), 1, figsize=(11, 11), sharex=True, sharey=True
    )
    for name, axis in zip(MODEL_NAMES, axes, strict=True):
        axis.plot(
            predictions.index,
            predictions.realised * 100,
            color="0.4",
            alpha=0.7,
            label="Realised",
        )
        axis.plot(
            predictions.index, predictions[name] * 100, label=LABELS[name], linewidth=1
        )
        axis.set_ylabel("Volatility (%)")
        axis.legend(loc="upper left", fontsize=8, ncols=2)
    axes[0].set_title(
        "Final test: next-five-day realised and forecast annualised volatility"
    )
    axes[-1].set_xlabel("Forecast-origin date (close of trading day)")
    save_figure(figure, directory, "realised_vs_predicted_test.png")


def plot_regime_errors(regimes: dict, directory: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(regimes))
    width = 0.15
    for i, name in enumerate(MODEL_NAMES):
        for axis, metric in zip(axes, ("rmse", "qlike"), strict=True):
            values = [
                regimes[regime]["models"][name][metric]
                if regimes[regime]["rows"]
                else np.nan
                for regime in regimes
            ]
            axis.bar(x + (i - 2) * width, values, width, label=LABELS[name])
    for axis in axes:
        axis.set_xticks(x, [name.title() for name in regimes])
        axis.set_xlabel("Realised target regime (development terciles)")
    axes[0].set(
        ylabel="RMSE (annualised volatility points)",
        title="Final test: error by regime",
    )
    axes[1].set(ylabel="QLIKE (variance loss)", title="Final test: error by regime")
    axes[1].legend(fontsize=7)
    save_figure(figure, directory, "error_by_regime.png")
