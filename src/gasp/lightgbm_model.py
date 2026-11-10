"""LightGBM forecaster (CONTRACTS.md, section 3).

make_lightgbm takes a feature table (feature_table.build_feature_table) when the
forecaster is created, as section 3 requires for models with extra inputs. Each
call to the forecaster:
  1. finds the feature-table row for the target week (error if missing)
  2. trains only on rows with period < target and a known weekly_change,
     retraining every refit_every weeks and reusing the model in between
  3. predicts weekly_change for the target row

Leakage: the feature table already follows the leakage rule (piece 2). This file
only ever trains on earlier rows and reads the target row's features, never its
weekly_change.
"""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd
import numpy as np

from gasp.feature_table import FEATURE_COLUMNS

TARGET_COLUMN = "weekly_change"

# Simple starting settings, per the issue: a few hundred small trees, a small
# learning rate, a fixed random_state so results repeat, and no console output.
# Tuning comes in Week 5.
DEFAULT_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 20,
    "random_state": 42,
    "verbose": -1,
}


def _features(rows: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Feature columns as plain floats. LightGBM rejects some pandas dtypes
    (e.g. the UInt32 that isocalendar() returns); NaN stays NaN."""
    return rows[feature_columns].astype(float)


def train_lightgbm(
    feature_table: pd.DataFrame,
    before,
    feature_columns=FEATURE_COLUMNS,
    params=None,
) -> lgb.LGBMRegressor:
    """Train on rows with period < before and a known weekly_change."""
    before = pd.Timestamp(before)
    feature_columns = list(feature_columns)
    train = feature_table[
        (feature_table["period"] < before) & feature_table[TARGET_COLUMN].notna()
    ]
    if train.empty:
        raise ValueError(f"No training rows before {before:%Y-%m-%d}")

    model = lgb.LGBMRegressor(**{**DEFAULT_PARAMS, **(params or {})})
    model.fit(_features(train, feature_columns), train[TARGET_COLUMN])
    return model


def make_lightgbm(
    feature_table: pd.DataFrame,
    feature_columns=FEATURE_COLUMNS,
    refit_every: int = 4,
    params=None,
):
    """Return a forecaster(history, target) -> float backed by LightGBM."""
    if refit_every < 1:
        raise ValueError("refit_every must be at least 1")

    feature_columns = list(feature_columns)
    table = feature_table.sort_values("period").reset_index(drop=True)

    missing = [c for c in ["period", TARGET_COLUMN, *feature_columns] if c not in table.columns]
    if missing:
        raise ValueError(f"feature table is missing columns: {missing}")

    # The trained model, the target it was trained for, and how many calls it has served
    state = {"model": None, "fit_target": None, "calls": 0}

    def forecast(history: pd.DataFrame, target: pd.Timestamp) -> float:
        target = pd.Timestamp(target)

        row = table[table["period"] == target]
        if row.empty:
            raise ValueError(f"No feature-table row for target week {target:%Y-%m-%d}")

        # Retrain every refit_every calls. Also retrain if called for an earlier week
        # than the current model was trained for, so a reused model never saw the future.
        needs_fit = (
            state["model"] is None
            or state["calls"] >= refit_every
            or target < state["fit_target"]
        )
        if needs_fit:
            state["model"] = train_lightgbm(table, target, feature_columns, params)
            state["fit_target"] = target
            state["calls"] = 0

        state["calls"] += 1
        return float(state["model"].predict(_features(row, feature_columns))[0])

    return forecast

# ---------------------------------------------------------------------------
# 90% interval (CONTRACTS.md, section 7)
# ---------------------------------------------------------------------------

# The 5th and 95th percentiles bound a 90% interval.
LOWER_QUANTILE = 0.05
UPPER_QUANTILE = 0.95

# Conformalized quantile regression: hold out the most recent two years of
# training weeks, score how far the raw quantile range misses on them, and
# widen (or narrow) the range by the 90th percentile of those scores.
CALIBRATION_WEEKS = 104
CALIBRATION_LEVEL = 0.90


def _fit_quantile_pair(train: pd.DataFrame, feature_columns: list[str], params=None):
    """Train the 5% and 95% quantile models on the given rows."""
    models = []
    for alpha in (LOWER_QUANTILE, UPPER_QUANTILE):
        model = lgb.LGBMRegressor(
            **{**DEFAULT_PARAMS, **(params or {}), "objective": "quantile", "alpha": alpha}
        )
        model.fit(_features(train, feature_columns), train[TARGET_COLUMN])
        models.append(model)
    return models[0], models[1]


def _predict_range(low_model, high_model, rows: pd.DataFrame, feature_columns: list[str]):
    """Raw (low, high) arrays, sorted per row in case the two quantile models cross."""
    X = _features(rows, feature_columns)
    a = low_model.predict(X)
    b = high_model.predict(X)
    return np.minimum(a, b), np.maximum(a, b)


def make_lightgbm_interval(
    feature_table: pd.DataFrame,
    feature_columns=FEATURE_COLUMNS,
    refit_every: int = 4,
    params=None,
    calibrate: bool = True,
):
    """Return forecast_interval(history, target) -> (low, high), a 90% interval.

    Two quantile models (alpha 0.05 and 0.95) with the same refit rules as
    make_lightgbm. With calibrate=True, each refit holds out the most recent
    CALIBRATION_WEEKS training weeks, trains the quantile models on the rows
    before them, scores each held-out week as max(low - actual, actual - high),
    and shifts the range by q = the 90th percentile of those scores:
    (low - q, high + q). Everything uses only rows before the target.

    The range always contains make_lightgbm's forecast for the same target
    (CONTRACTS.md, section 7): if the forecast falls outside, the range widens to it.
    """
    if refit_every < 1:
        raise ValueError("refit_every must be at least 1")

    feature_columns = list(feature_columns)
    table = feature_table.sort_values("period").reset_index(drop=True)

    missing = [c for c in ["period", TARGET_COLUMN, *feature_columns] if c not in table.columns]
    if missing:
        raise ValueError(f"feature table is missing columns: {missing}")

    # The point forecast the range must contain: a make_lightgbm forecaster with the
    # same settings and refit schedule, so it matches what walk_forward compares against.
    point_forecast = make_lightgbm(table, feature_columns, refit_every, params)

    state = {"models": None, "q": 0.0, "fit_target": None, "calls": 0}

    def fit(target: pd.Timestamp):
        """Train the quantile pair (and calibrate it) using only rows before target."""
        train = table[(table["period"] < target) & table[TARGET_COLUMN].notna()]

        if not calibrate:
            if train.empty:
                raise ValueError(f"No training rows before {target:%Y-%m-%d}")
            return _fit_quantile_pair(train, feature_columns, params), 0.0

        if len(train) <= CALIBRATION_WEEKS:
            raise ValueError(
                f"Need more than {CALIBRATION_WEEKS} training weeks before "
                f"{target:%Y-%m-%d} to calibrate"
            )
        proper = train.iloc[:-CALIBRATION_WEEKS]
        held_out = train.iloc[-CALIBRATION_WEEKS:]

        models = _fit_quantile_pair(proper, feature_columns, params)
        low, high = _predict_range(*models, held_out, feature_columns)
        actual = held_out[TARGET_COLUMN].to_numpy(dtype=float)

        # Negative when the actual is inside the range, positive when it's outside
        scores = np.maximum(low - actual, actual - high)
        q = float(np.quantile(scores, CALIBRATION_LEVEL))
        return models, q

    def forecast_interval(history: pd.DataFrame, target: pd.Timestamp) -> tuple[float, float]:
        target = pd.Timestamp(target)

        row = table[table["period"] == target]
        if row.empty:
            raise ValueError(f"No feature-table row for target week {target:%Y-%m-%d}")

        # Same refit rules as make_lightgbm
        needs_fit = (
            state["models"] is None
            or state["calls"] >= refit_every
            or target < state["fit_target"]
        )
        if needs_fit:
            state["models"], state["q"] = fit(target)
            state["fit_target"] = target
            state["calls"] = 0
        state["calls"] += 1

        low, high = _predict_range(*state["models"], row, feature_columns)
        low = float(low[0]) - state["q"]
        high = float(high[0]) + state["q"]
        if low > high:
            # A negative q can narrow past the middle; collapse to the midpoint
            low = high = (low + high) / 2

        # Section 7: low <= forecast <= high every week
        point = point_forecast(history, target)
        return min(low, point), max(high, point)

    return forecast_interval

def feature_importance(model: lgb.LGBMRegressor, feature_columns=FEATURE_COLUMNS) -> pd.DataFrame:
    """How much each feature reduced error across all trees (gain), most important first."""
    gain = model.booster_.feature_importance(importance_type="gain")
    return (
        pd.DataFrame({"feature": list(feature_columns), "importance": gain})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )