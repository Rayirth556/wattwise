"""Shared data processing, labeling, prediction and energy helpers for WattWise."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = "household_power_consumption.txt"
ARTIFACT_DIR = ROOT / "model" / "artifacts"

CLASSES = ["Normal", "Unusually High", "Standby-like"]
WINDOW = 15            # rolling window in minutes (current + past readings only)
MIN_PERIODS = 5        # minimum valid readings inside the window
HIGH_Q = 0.95          # "Unusually High": power above this training quantile
LOW_Q = 0.15           # "Standby-like": power at/below this training quantile ...
STABLE_Q = 0.50        # ... AND rolling std at/below this training quantile
TRAIN_FRACTION = 0.80  # chronological split

FEATURES = ["Global_active_power", "Global_reactive_power", "Voltage",
            "Global_intensity", "hour", "day_of_week", "is_weekend",
            "roll_mean_15", "roll_std_15"]
NUMERIC = ["Global_active_power", "Global_reactive_power", "Voltage",
           "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]


def find_dataset(data_dir=None):
    """Locate the UCI text file (or CSV) inside the data directory."""
    d = Path(data_dir or ROOT / "data")
    for pattern in ("household_power_consumption.txt", "*.txt", "*.csv"):
        hits = sorted(d.glob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No dataset found in {d}")


def load_raw(path) -> pd.DataFrame:
    """Parse the raw file: ';' delimited, '?' = missing. Returns a minute-indexed frame
    covering the full time span; absent/missing minutes are NaN rows."""
    df = pd.read_csv(path, sep=";", na_values=["?", ""], low_memory=False)
    ts = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df = df[NUMERIC].apply(pd.to_numeric, errors="coerce").astype("float32")
    df.index = ts
    df = df[~df.index.duplicated()].sort_index()
    return df.reindex(pd.date_range(df.index[0], df.index[-1], freq="min"))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time + rolling features. Rolling windows are trailing (current and past only)."""
    out = df.copy()
    p = out["Global_active_power"]
    out["roll_mean_15"] = p.rolling(WINDOW, min_periods=MIN_PERIODS).mean()
    out["roll_std_15"] = p.rolling(WINDOW, min_periods=MIN_PERIODS).std()
    out["hour"] = out.index.hour.astype("int8")
    out["day_of_week"] = out.index.day_of_week.astype("int8")
    out["is_weekend"] = (out["day_of_week"] >= 5).astype("int8")
    out["valid"] = out[FEATURES].notna().all(axis=1)
    return out


def fit_thresholds(train: pd.DataFrame) -> dict:
    """Labeling thresholds derived from TRAINING rows only."""
    t = train[train["valid"]]
    return {
        "high_power_kw": float(t["Global_active_power"].quantile(HIGH_Q)),
        "standby_power_kw": float(t["Global_active_power"].quantile(LOW_Q)),
        "standby_roll_std_kw": float(t["roll_std_15"].quantile(STABLE_Q)),
    }


def assign_labels(df: pd.DataFrame, th: dict) -> pd.Series:
    """Deterministic pseudo-labels (see README). Not ground truth."""
    p, s = df["Global_active_power"], df["roll_std_15"]
    lab = pd.Series("Normal", index=df.index, dtype="object")
    lab[(p <= th["standby_power_kw"]) & (s <= th["standby_roll_std_kw"])] = "Standby-like"
    lab[p > th["high_power_kw"]] = "Unusually High"
    return lab


def predict(model, df: pd.DataFrame) -> np.ndarray:
    """Predict class names for rows of a feature frame (rows must be valid)."""
    return model.predict(df[FEATURES].to_numpy(dtype="float32"))


def energy_kwh(power_kw: pd.Series, interval_min: float = 1.0) -> dict:
    """Energy from average-power readings. Each valid reading (kW) represents one
    sampling interval (1 min in this dataset): kWh = sum(kW) * interval/60.
    Missing readings are NOT interpolated; they're counted and reported."""
    p = power_kw.dropna()
    if p.empty:
        return {"kwh": 0.0, "n_valid": 0, "n_missing": 0}
    span = int(round((p.index[-1] - p.index[0]).total_seconds() / 60 / interval_min)) + 1
    return {"kwh": float(p.sum() * interval_min / 60.0),
            "n_valid": int(len(p)), "n_missing": span - int(len(p))}


def load_metadata() -> dict:
    return json.loads((ARTIFACT_DIR / "metadata.json").read_text())
