"""What-If grid for LightGBM (CONTRACTS.md, section 11).

lightgbm_whatif trains once, exactly as a fresh make_lightgbm and
make_lightgbm_interval would for the target (calibrated range included), then
predicts the target row once per weather step.

Scaling, for the target week only:
- every raw weather input (US hdd_utility_gas, hdd_population, cdd_population,
  and each regional hdd_div*) is multiplied by 1 + step / 100
- each matching _dev input changes by raw * step / 100, since the normal stays
  the same (scaled - normal = dev + raw * step / 100), so no normals are recomputed
- last week's weather (*_lag_1) and every other feature stay unchanged
"""

from __future__ import annotations

import pandas as pd

from gasp import lightgbm_model
from gasp.feature_table import FEATURE_COLUMNS

MODEL_NAME = "lightgbm"
US_WEATHER_COLUMNS = ("hdd_utility_gas", "hdd_population", "cdd_population")
REGIONAL_PREFIX = "hdd_div"
WHATIF_COLUMNS = ["model", "target", "weather_change_pct", "predicted", "low", "high"]


def raw_weather_columns(feature_columns=FEATURE_COLUMNS) -> list[str]:
    """The raw weather inputs that get scaled: the US ones plus each regional hdd_div*.
    Never a _dev column (shifted separately) or a lagged one (last week stays as it was)."""
    return [
        c for c in feature_columns
        if c in US_WEATHER_COLUMNS
        or (c.startswith(REGIONAL_PREFIX) and not c.endswith("_dev") and "_lag" not in c)
    ]


def scale_weather(row: pd.DataFrame, step: int, feature_columns=FEATURE_COLUMNS) -> pd.DataFrame:
    """A copy of the target row with its weather scaled by step percent."""
    scaled = row.copy()
    factor = step / 100
    for raw in raw_weather_columns(feature_columns):
        base = scaled[raw].astype(float)
        scaled[raw] = base * (1 + factor)
        dev = f"{raw}_dev"
        if dev in feature_columns:
            scaled[dev] = scaled[dev].astype(float) + base * factor
    return scaled


def lightgbm_whatif(feature_table: pd.DataFrame, target, steps=range(-30, 31, 5)) -> pd.DataFrame:
    """CONTRACTS.md section 11 rows for model = "lightgbm", one per weather step."""
    target = pd.Timestamp(target)
    steps = [int(s) for s in steps]
    feature_columns = list(FEATURE_COLUMNS)
    table = feature_table.sort_values("period").reset_index(drop=True)

    row = table[table["period"] == target]
    if row.empty:
        raise ValueError(f"No feature-table row for target week {target:%Y-%m-%d}")

    # Train once, exactly as a fresh make_lightgbm / make_lightgbm_interval would for target
    point_model = lightgbm_model.train_lightgbm(table, target, feature_columns)
    interval_models, q = lightgbm_model.fit_interval_models(table, target, feature_columns)

    # One copy of the target row per step, with only its weather changed
    grid = pd.concat([scale_weather(row, s, feature_columns) for s in steps], ignore_index=True)

    predicted = point_model.predict(lightgbm_model._features(grid, feature_columns))
    raw_low, raw_high = lightgbm_model._predict_range(*interval_models, grid, feature_columns)

    rows = []
    for step, point, lo, hi in zip(steps, predicted, raw_low, raw_high):
        low, high = lightgbm_model.finish_range(lo, hi, q, point)
        rows.append({
            "model": MODEL_NAME,
            "target": target,
            "weather_change_pct": step,
            "predicted": float(point),
            "low": low,
            "high": high,
        })
    return pd.DataFrame(rows, columns=WHATIF_COLUMNS)