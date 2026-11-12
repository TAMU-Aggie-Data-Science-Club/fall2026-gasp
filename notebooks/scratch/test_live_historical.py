
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from gasp import db, pipeline
from gasp.live import (
    forecast,
    score,
    load_live_forecasts,
    load_explanations,
)

# Read source data without changing it.
storage = db.read_series(
    pipeline.LOWER_48_SERIES,
    path=db.DB_PATH,
)

weather = db.read_degree_days(
    region="US",
    path=db.DB_PATH,
)

# Simulate forecasting two weeks before the latest observation.
latest = pd.Timestamp(storage["period"].max())
cutoff = latest - pd.Timedelta(days=14)
target = cutoff + pd.Timedelta(days=7)

print("Historical cutoff:", cutoff.date())
print("Forecast target:", target.date())

# Only observations through the cutoff are available.
historical_storage = storage.loc[
    storage["period"] <= cutoff
].copy()

# Preserve the actual target-week value for scoring later.
target_observation = storage.loc[
    storage["period"] == target
].copy()

assert len(target_observation) == 1

# Weather for the target is assumed available for this
# retrospective test. It is not necessarily an authentic
# point-in-time weather snapshot.
historical_weather = weather.loc[
    weather["week_end"] <= target
].copy()

with tempfile.TemporaryDirectory() as folder:
    test_path = Path(folder) / "historical.db"

    # Build an isolated database.
    db.write_series(
        historical_storage,
        path=test_path,
    )

    db.write_degree_days(
        historical_weather,
        path=test_path,
    )

    # Disable only the release-date check in this test.
    # Production release protection remains unchanged.
    with patch("gasp.live.report_is_released", return_value=False):
        print("\nFIRST FORECAST RUN")
        first = forecast(path=test_path, refresh=False)

        print("\nSECOND FORECAST RUN")
        second = forecast(path=test_path, refresh=False)

    saved = load_live_forecasts(path=test_path)
    explanations = load_explanations(path=test_path)

    print("\nSaved forecasts:")
    print(saved.to_string(index=False))

    print("\nSaved explanations:", len(explanations))

    # Three unique forecasts; rerun must not duplicate them.
    assert len(saved) == 3
    assert saved[["model", "target"]].duplicated().sum() == 0
    assert len(explanations) == 13

    # Point forecasts should be reproducible.
    pd.testing.assert_frame_equal(
        first.reset_index(drop=True),
        second.reset_index(drop=True),
    )

    # Interval checks.
    for model in ["sarimax", "lightgbm"]:
        row = saved.loc[saved["model"] == model].iloc[0]
        assert row["low"] <= row["predicted"] <= row["high"]

    # Restore the target week's actual storage observation.
    db.write_series(
        target_observation,
        path=test_path,
    )

    print("\nSCORING")
    scored_first = score(path=test_path)
    scored_second = score(path=test_path)

    print("First score:", scored_first)
    print("Second score:", scored_second)

    after_scoring = load_live_forecasts(path=test_path)

    previous_value = historical_storage.loc[
        historical_storage["period"] == cutoff,
        "value",
    ].iloc[0]

    actual_value = target_observation["value"].iloc[0]
    expected_change = float(actual_value - previous_value)

    assert scored_first == 3
    assert scored_second == 0
    assert np.allclose(
        after_scoring["actual"].astype(float),
        expected_change,
    )

    # Predictions and intervals must remain unchanged.
    for column in ["predicted", "low", "high", "made_at"]:
        pd.testing.assert_series_equal(
            saved[column],
            after_scoring[column],
        )

    print("\nAll historical integration tests passed!")
