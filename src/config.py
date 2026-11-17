from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TICKER = "SPY"
START_DATE = "2010-01-01"
END_DATE = "2026-11-16"
RAW_PATH = ROOT / "data/raw/spy_daily.csv"
SEED = 42
ANNUALISATION = 252
HORIZON = 5
LOOKBACK = 63
EPSILON = 1e-12
MODEL_THREADS = 2
BASELINES = ("prev_5", "rv_21", "ewma", "har")
MODEL_NAMES = (*BASELINES, "hgb")
LABELS = {
    "prev_5": "Previous 5-day RV",
    "rv_21": "21-day RV",
    "ewma": "EWMA",
    "har": "HAR-style OLS",
    "hgb": "Gradient boosting",
}


@dataclass(frozen=True)
class Settings:
    min_train: int = 756
    validation_size: int = 252
    step: int = 252
    gap: int = HORIZON
    test_size: int = 252
    horizon: int = HORIZON
    lookback: int = LOOKBACK
    ewma_decay: float = 0.94
    seed: int = SEED
    calendar_features: str = "weekday only"
    dm_lag: int = HORIZON - 1

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RunConfig:
    data_path: Path = RAW_PATH
    output_root: Path = ROOT
    settings: Settings = Settings()

    @property
    def reports(self) -> Path:
        return self.output_root / "reports"

    @property
    def models(self) -> Path:
        return self.output_root / "models"
