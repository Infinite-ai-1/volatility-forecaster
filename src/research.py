import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yfinance
from threadpoolctl import threadpool_limits

from src.baselines import baseline_forecasts
from src.config import BASELINES, MODEL_THREADS, ROOT, RunConfig, Settings
from src.data import file_sha256, price_summary
from src.dataset import modelling_dates, prepare_dataset
from src.metrics import score_forecasts
from src.models import LogVolatilityRegressor
from src.splits import holdout_indices, walk_forward_splits


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def implementation_sha256() -> str:
    files = sorted((ROOT / "src").glob("*.py"))
    signatures = "\n".join(f"{path.name}:{file_sha256(path)}" for path in files)
    return hashlib.sha256(signatures.encode()).hexdigest()


def library_versions() -> dict[str, str]:
    return {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "yfinance": yfinance.__version__,
    }


def experiment_context(prices: pd.DataFrame, config: RunConfig) -> dict:
    settings = config.settings
    dates = modelling_dates(prices, settings)
    development, purge, test = holdout_indices(
        len(dates), settings.test_size, settings.gap
    )
    folds = list(
        walk_forward_splits(
            len(development),
            settings.min_train,
            settings.validation_size,
            settings.gap,
            settings.step,
        )
    )
    if not folds:
        raise ValueError("Not enough development rows for a complete purged fold")
    return {
        "schema_version": 1,
        "data": price_summary(prices, config.data_path),
        "settings": settings.as_dict(),
        "library_versions": library_versions(),
        "implementation_sha256": implementation_sha256(),
        "feature_count": 34,
        "splits": {
            "modelling_rows": len(dates),
            "development_rows": len(development),
            "development_start": dates[development[0]].date().isoformat(),
            "development_end": dates[development[-1]].date().isoformat(),
            "purge_start": dates[purge[0]].date().isoformat(),
            "purge_end": dates[purge[-1]].date().isoformat(),
            "test_start": dates[test[0]].date().isoformat(),
            "test_end": dates[test[-1]].date().isoformat(),
            "test_rows": len(test),
            "test_label_end": prices.index[-1].date().isoformat(),
            "fold_count": len(folds),
            "development_tail_rows_outside_complete_folds": len(development)
            - int(folds[-1][1][-1])
            - 1,
        },
        "metric_units": {
            "qlike": "dimensionless loss on variance",
            "mae": "annualised volatility percentage points",
            "rmse": "annualised volatility percentage points",
        },
    }


def check_context(expected: dict, actual: dict) -> None:
    for field in (
        "data",
        "settings",
        "library_versions",
        "implementation_sha256",
        "splits",
    ):
        if expected[field] != actual[field]:
            raise ValueError(
                f"Frozen experiment differs on {field}. "
                "Use a new output root for a new experiment"
            )


def fold_metadata(
    features: pd.DataFrame,
    train: np.ndarray,
    val: np.ndarray,
    fold: int,
    settings: Settings,
) -> dict:
    def date(position: int) -> str:
        return features.index[position].date().isoformat()

    return {
        "fold": fold,
        "train_start": date(train[0]),
        "train_end": date(train[-1]),
        "last_train_label_end": date(train[-1] + settings.horizon),
        "validation_start": date(val[0]),
        "validation_end": date(val[-1]),
        "train_rows": len(train),
        "n_rows": len(val),
        "gap_rows": settings.gap,
    }


def score_development_baselines(
    features: pd.DataFrame, target: pd.Series, settings: Settings
) -> list[dict]:
    rows = []
    with threadpool_limits(limits=MODEL_THREADS):
        for fold, (train, val) in enumerate(
            walk_forward_splits(
                len(features),
                settings.min_train,
                settings.validation_size,
                settings.gap,
                settings.step,
            ),
            start=1,
        ):
            forecasts = baseline_forecasts(
                features.iloc[train], target.iloc[train], features.iloc[val]
            )
            metadata = fold_metadata(features, train, val, fold, settings)
            for name, forecast in forecasts.items():
                rows.append(
                    {
                        **metadata,
                        "model": name,
                        **score_forecasts(target.iloc[val], forecast),
                    }
                )
    return rows


def score_development_ml(
    features: pd.DataFrame, target: pd.Series, settings: Settings, parameters: dict
) -> list[dict]:
    rows = []
    with threadpool_limits(limits=MODEL_THREADS):
        for fold, (train, val) in enumerate(
            walk_forward_splits(
                len(features),
                settings.min_train,
                settings.validation_size,
                settings.gap,
                settings.step,
            ),
            start=1,
        ):
            model = LogVolatilityRegressor(parameters, settings.seed).fit(
                features.iloc[train], target.iloc[train]
            )
            forecast = model.predict(features.iloc[val])
            rows.append(
                {
                    **fold_metadata(features, train, val, fold, settings),
                    "model": "hgb",
                    **score_forecasts(target.iloc[val], forecast),
                }
            )
    return rows


def fold_summary(scores: pd.DataFrame) -> dict:
    means = (
        scores.groupby("model", sort=False)[["qlike", "rmse", "mae"]]
        .mean()
        .to_dict("index")
    )
    pivot = scores.pivot(index="fold", columns="model", values="qlike")
    best_baseline = min(BASELINES, key=lambda name: means[name]["qlike"])
    return {
        "fold_means": means,
        "best_baseline_by_development_qlike": best_baseline,
        "win_counts": {
            "folds": len(pivot),
            "hgb_vs_ewma": int((pivot.hgb < pivot.ewma).sum()),
            "hgb_vs_best_baseline_in_each_fold": int(
                (pivot.hgb < pivot[list(BASELINES)].min(axis=1)).sum()
            ),
        },
    }


def frozen_development(
    prices: pd.DataFrame, config: RunConfig, parameters: dict
) -> tuple[pd.DataFrame, dict]:
    features, target = prepare_dataset(prices, config.settings, development_only=True)
    rows = score_development_baselines(features, target, config.settings)
    rows += score_development_ml(features, target, config.settings, parameters)
    scores = pd.DataFrame(rows).sort_values(["fold", "model"]).reset_index(drop=True)
    summary = fold_summary(scores)
    summary["regime_cutoffs"] = {
        "low_upper": float(target.quantile(1 / 3)),
        "medium_upper": float(target.quantile(2 / 3)),
    }
    return scores, summary
