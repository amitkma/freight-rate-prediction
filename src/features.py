"""Loading, cleaning and features."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TARGET = "posted_rate"
WEIGHT_CAP = 47_500.0
START_DATE = pd.Timestamp("2025-01-01")


def load_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train_test.csv", parse_dates=["date"])
    valid = pd.read_csv(data_dir / "validation.csv", parse_dates=["date"])
    december = pd.read_csv(data_dir / "december_chart_inputs.csv", parse_dates=["date"])
    return train, valid, december


def label_outliers(df: pd.DataFrame, low: float = 0.5, high: float = 2.0) -> pd.Series:
    """Rates far off what similar loads pay (same equipment and distance band)."""
    rpm = df[TARGET] / df["distance"]
    band = pd.qcut(df["distance"], 10, labels=False, duplicates="drop")
    ratio = rpm / rpm.groupby([df["equipment"], band]).transform("median")
    return (ratio < low) | (ratio > high)


def market_calendar(*frames: pd.DataFrame) -> pd.Series:
    """Daily mean market_index. Pass validation too so Nov-Dec dates are covered.

    quote_signal is not used: it leaks the target in some months and is noise in Nov-Dec.
    """
    loads = pd.concat([f[["date", "market_index"]] for f in frames], ignore_index=True)
    daily = loads.groupby("date")["market_index"].mean().asfreq("D")
    return daily.interpolate(limit_direction="both").rename("mi_day")


class Features:
    """Learns weight medians and city coordinates from train, then builds features."""

    def __init__(self, calendar: pd.Series):
        self.calendar = calendar

    def fit(self, train: pd.DataFrame) -> Features:
        self.weight_median_ = train.assign(weight=train["weight"].abs()).groupby("equipment")["weight"].median()
        pick = train.groupby("pickup")[["pickup_lat", "pickup_lon"]].first()
        drop = train.groupby("delivery")[["delivery_lat", "delivery_lon"]].first()
        pick.columns = drop.columns = ["lat", "lon"]
        self.city_coords_ = pd.concat([pick, drop]).groupby(level=0).first()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        X["pickup"], X["delivery"] = df["pickup"], df["delivery"]
        X["equipment"] = pd.Categorical(df["equipment"], categories=["Dry Van", "Flatbed", "Reefer"])
        X["is_flatbed"] = (df["equipment"] == "Flatbed").astype(int)
        X["is_reefer"] = (df["equipment"] == "Reefer").astype(int)

        X["distance"] = df["distance"]
        # fixed costs per load -> rate per mile falls with distance
        X["inv_distance"] = 100.0 / df["distance"]
        X["log_distance"] = np.log(df["distance"])

        # negative weights look like sign errors
        weight = df["weight"].abs()
        X["weight_missing"] = weight.isna().astype(int)
        X["weight_capped"] = np.isclose(weight, WEIGHT_CAP).astype(int)
        X["weight_k"] = weight.fillna(df["equipment"].map(self.weight_median_)) / 1000.0

        for side in ["pickup", "delivery"]:
            for axis in ["lat", "lon"]:
                col = f"{side}_{axis}"
                X[col] = df[col] if col in df else df[side].map(self.city_coords_[axis])

        X["mi_day"] = self.calendar.reindex(df["date"]).to_numpy()
        X["days_since_start"] = (df["date"] - START_DATE).dt.days
        return X


def target(df: pd.DataFrame) -> pd.Series:
    """log(rate per mile)"""
    return np.log(df[TARGET] / df["distance"])


def predict_rate(log_rpm_pred: np.ndarray, X: pd.DataFrame) -> np.ndarray:
    return np.exp(log_rpm_pred) * X["distance"].to_numpy()
