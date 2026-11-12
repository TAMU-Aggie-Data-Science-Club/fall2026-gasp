
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from gasp.live import (
    init_live_tables,
    save_live_forecasts,
    load_live_forecasts,
)

with tempfile.TemporaryDirectory() as folder:
    test_path = Path(folder) / "release_test.db"

    init_live_tables(test_path)

    # A historical report that has already been released.
    target = pd.Timestamp("2025-10-03")

    test_forecast = pd.DataFrame([
        {
            "model": "lightgbm",
            "target": target,
            "predicted": 50.0,
            "low": 30.0,
            "high": 70.0,
        }
    ])

    # Simulate a report that is already released.
    with patch(
        "gasp.live.report_is_released",
        return_value=True,
    ):
        try:
            save_live_forecasts(test_forecast, path=test_path)
        except ValueError as error:
            print("Correctly blocked released forecast:")
            print(error)
        else:
            raise AssertionError(
                "Released forecast was incorrectly accepted."
            )

    saved = load_live_forecasts(path=test_path)

    assert saved.empty

    print("\nNo forecasts were saved.")
    print("Release safeguard test passed!")
