"""Models and walk-forward evaluation."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import TARGET, Features, label_outliers, predict_rate, target

# pick models on the selection folds, check the winner on the holdout
SELECTION_STARTS = ["2025-05-01", "2025-07-01"]
HOLDOUT_START = "2025-09-01"

LINEAR_FEATURES = [
    "inv_distance", "log_distance", "weight_k", "weight_missing", "weight_capped",
    "is_flatbed", "is_reefer", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "mi_day",
]
TREE_FEATURES = [
    "distance", "log_distance", "equipment", "weight_k", "weight_missing", "weight_capped",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "mi_day", "days_since_start",
]


class RateModel:
    """Fits on raw loads, predicts dollars."""

    def __init__(self, calendar: pd.Series, drop_outliers: bool = True):
        self.calendar = calendar
        self.drop_outliers = drop_outliers

    def fit(self, train: pd.DataFrame) -> RateModel:
        if self.drop_outliers:
            train = train[~label_outliers(train)]
        self.features_ = Features(self.calendar).fit(train)
        self._fit(self.features_.transform(train), target(train))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self.features_.transform(df)
        return predict_rate(self.predict_log_rpm(X), X)

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        raise NotImplementedError

    def predict_log_rpm(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class BaselineModel(RateModel):
    """Median rate per mile by equipment and distance decile."""

    def _fit(self, X, y):
        _, self.edges_ = pd.qcut(X["distance"], 10, retbins=True)
        self.edges_[0], self.edges_[-1] = -np.inf, np.inf
        self.table_ = y.groupby([X["equipment"].astype(str), self._band(X)]).median()
        self.fallback_ = y.median()

    def _band(self, X):
        return pd.cut(X["distance"], self.edges_, labels=False)

    def predict_log_rpm(self, X):
        index = pd.MultiIndex.from_arrays([X["equipment"].astype(str), self._band(X)])
        return self.table_.reindex(index).fillna(self.fallback_).to_numpy()


class LinearModel(RateModel):
    """Huber regression with city dummies.

    level: "trend" = linear time trend, "none" = no time term,
    "recent" = no trend, shift to the median residual of the last level_days.
    features / city_dummies: for the feature ablation.
    """

    def __init__(self, calendar, level="recent", level_days=56, features=LINEAR_FEATURES, city_dummies=True, **kwargs):
        super().__init__(calendar, **kwargs)
        self.level = level
        self.level_days = level_days
        self.features = features
        self.city_dummies = city_dummies

    def design(self, X):
        D = X[self.features + (["days_since_start"] if self.level == "trend" else [])].astype(float)
        if not self.city_dummies:
            return D
        # unseen cities get all zeros and fall back on coordinates
        for side in ["pickup", "delivery"]:
            dummies = pd.get_dummies(pd.Categorical(X[side], categories=self.cities_), prefix=side, dtype=float)
            D = pd.concat([D, dummies.set_axis(X.index)], axis=1)
        return D

    def _fit(self, X, y):
        self.cities_ = sorted(set(X["pickup"]) | set(X["delivery"]))[1:]  # drop one as reference
        self.linear_ = make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=2000))
        self.linear_.fit(self.design(X), y)
        self.offset_ = 0.0
        if self.level == "recent":
            residual = y - self.linear_.predict(self.design(X))
            recent = X["days_since_start"] > X["days_since_start"].max() - self.level_days
            self.offset_ = float(residual[recent].median() - residual.median())

    def predict_log_rpm(self, X):
        return self.linear_.predict(self.design(X)) + self.offset_


LGB_PARAMS = {"objective": "huber", "alpha": 0.3, "learning_rate": 0.03, "n_estimators": 1200, "num_leaves": 31,
              "min_child_samples": 50, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
              "verbose": -1, "random_state": 0}


class LightGBMModel(RateModel):
    """LightGBM on the same inputs plus the date."""

    def _fit(self, X, y):
        self.trees_ = lgb.LGBMRegressor(**LGB_PARAMS).fit(X[TREE_FEATURES], y)

    def predict_log_rpm(self, X):
        return self.trees_.predict(X[TREE_FEATURES])


# --- evaluation --------------------------------------------------------------

def walk_forward_folds(df: pd.DataFrame, test_starts: list[str], horizon_months: int = 2):
    """Train on everything before each start date, test on the next 2 months."""
    for start in map(pd.Timestamp, test_starts):
        end = start + pd.DateOffset(months=horizon_months)
        yield start, df[df["date"] < start], df[(df["date"] >= start) & (df["date"] < end)]


def metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    err = pred - actual
    return {
        "MAE": np.abs(err).mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "MdAPE %": 100 * np.median(np.abs(err) / actual),
        "bias %": 100 * (pred.sum() / actual.sum() - 1),
    }


def evaluate(make_model, df: pd.DataFrame, test_starts: list[str]) -> pd.DataFrame:
    rows = []
    for start, fit_df, test_df in walk_forward_folds(df, test_starts):
        pred = make_model().fit(fit_df).predict(test_df)
        actual = test_df[TARGET].to_numpy()
        clean = ~label_outliers(test_df).to_numpy()
        rows.append({"fold": start.strftime("%b"), **metrics(pred, actual),
                     # MAE without the corrupted labels
                     "clean MAE": np.abs(pred[clean] - actual[clean]).mean()})
    return pd.DataFrame(rows).set_index("fold")
