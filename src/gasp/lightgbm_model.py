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


def feature_importance(model: lgb.LGBMRegressor, feature_columns=FEATURE_COLUMNS) -> pd.DataFrame:
    """How much each feature reduced error across all trees (gain), most important first."""
    gain = model.booster_.feature_importance(importance_type="gain")
    return (
        pd.DataFrame({"feature": list(feature_columns), "importance": gain})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )