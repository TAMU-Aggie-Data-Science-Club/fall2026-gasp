
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

import pandas as pd

from gasp import db, pipeline
from gasp.live import init_live_tables, score


with tempfile.TemporaryDirectory() as folder:
    test_path = Path(folder) / "test_score.db"

    # Create the test database and live tables.
    init_live_tables(test_path)

    # Insert a previously generated forecast.
    with closing(sqlite3.connect(test_path)) as conn:
        conn.execute(
            """
            INSERT INTO live_forecasts
                (model, target, predicted, low, high, made_at, actual)
            VALUES (?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                "sarimax",
                "2026-11-06",
                49.91,
                15.40,
                84.42,
                "2026-11-05T15:00:00+00:00",
            ),
        )
        conn.commit()

    # Create two consecutive weeks of actual storage data.
    observations = pd.DataFrame({
        "period": pd.to_datetime([
            "2026-10-30",
            "2026-11-06",
        ]),
        "series": [pipeline.LOWER_48_SERIES] * 2,
        "value": [3801.0, 3833.0],
    })

    # Write the observations to the temporary database.
    db.write_series(observations, path=test_path)

    # First scoring run.
    print("First scoring run:")
    first_updated = score(path=test_path)

    # Second scoring run (should update zero rows).
    print("\nSecond scoring run:")
    second_updated = score(path=test_path)

    # Retrieve the saved forecast.
    with closing(sqlite3.connect(test_path)) as conn:
        row = conn.execute(
            """
            SELECT model, target, predicted, low, high, made_at, actual
            FROM live_forecasts
            WHERE model = 'sarimax'
              AND target = '2026-11-06'
            """
        ).fetchone()

    print("\nSaved forecast after scoring:")
    print(row)

    # Verify the actual weekly change.
    assert row[6] == 32.0

    # Verify the original prediction was preserved.
    assert row[2] == 49.91

    # Verify the prediction interval was preserved.
    assert row[3] == 15.40
    assert row[4] == 84.42

    # Verify the timestamp was preserved.
    assert row[5] == "2026-11-05T15:00:00+00:00"

    # Verify scoring is idempotent.
    assert first_updated == 1
    assert second_updated == 0

    print("\nAll scoring tests passed!")
