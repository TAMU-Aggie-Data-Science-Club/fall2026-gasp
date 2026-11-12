
"""Generate and score live EIA natural gas storage forecasts."""
import pandas as pd
import sqlite3
from gasp.explain import explain
from datetime import datetime, time
from zoneinfo import ZoneInfo
from contextlib import closing
import numpy as np
from gasp.explain import explain
from gasp import (
    db,
    pipeline,
    feature_table,
    weather_features,
    baselines,
)

from gasp.sarimax import make_sarimax
from gasp.lightgbm_model import make_lightgbm, make_lightgbm_interval

def add_upcoming_week(storage_df: pd.DataFrame) -> pd.DataFrame:
    """Append a placeholder for the next weekly storage report."""

    if storage_df.empty:
        raise ValueError("Cannot forecast without historical storage data.")

    # Sort observations by date without modifying the original DataFrame.
    storage = storage_df.sort_values("period").copy()

    # Find the Friday following the latest available storage week.
    latest_period = pd.Timestamp(storage["period"].max())
    next_period = latest_period + pd.Timedelta(days=7)

    # Create a placeholder with an unknown storage value.
    upcoming = pd.DataFrame({
        "period": [next_period],
        "series": [storage.iloc[-1]["series"]],
        "value": [float("nan")],
    })

    # Return the historical data plus the upcoming week.
    return pd.concat([storage, upcoming], ignore_index=True)


def report_is_released(target):
    """Return True once the target week's EIA report release time has passed."""

    target = pd.Timestamp(target)

    # The Friday ending the reported week is followed by
    # the next Thursday's EIA publication.
    report_date = target + pd.Timedelta(days=6)

    release_time = datetime.combine(
        report_date.date(),
        time(10, 30),
        tzinfo=ZoneInfo("America/New_York"),
    )

    now = datetime.now(ZoneInfo("America/New_York"))

    return now >= release_time


def init_live_tables(path=db.DB_PATH):
    """Create the live forecast tables if they do not exist."""

    db.init_db(path)

    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS live_forecasts (
                model TEXT NOT NULL,
                target TEXT NOT NULL,
                predicted REAL NOT NULL,
                low REAL,
                high REAL,
                made_at TEXT NOT NULL,
                actual REAL,
                PRIMARY KEY (model, target)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS live_explanations (
                target TEXT NOT NULL,
                feature TEXT NOT NULL,
                label TEXT NOT NULL,
                value REAL NOT NULL,
                contribution REAL NOT NULL,
                base_value REAL NOT NULL,
                PRIMARY KEY (target, feature)
            )
        """)


def save_live_forecasts(results, path=db.DB_PATH):
    """Save new forecasts without overwriting existing predictions."""

    init_live_tables(path)

    made_at = datetime.now(ZoneInfo("UTC")).isoformat()

    saved = 0

    with closing(sqlite3.connect(path)) as conn, conn:
        for _, row in results.iterrows():
            target = pd.Timestamp(row["target"])

            # Never save a forecast after its report is released.
            if report_is_released(target):
                raise ValueError(
                    f"Cannot save forecast for {target.date()}: "
                    "the report has already been released."
                )

            # Do not save forecasts for weeks with known actual storage.
            existing_storage = db.read_series(
                pipeline.LOWER_48_SERIES,
                path=path,
            )

            if target in set(pd.to_datetime(existing_storage["period"])):
                raise ValueError(
                    f"Cannot forecast {target.date()}: "
                    "actual storage is already available."
                )

            # Convert missing interval values to SQLite NULL.
            low = None if pd.isna(row["low"]) else float(row["low"])
            high = None if pd.isna(row["high"]) else float(row["high"])

            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO live_forecasts
                    (model, target, predicted, low, high, made_at, actual)
                VALUES (?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    row["model"],
                    target.strftime("%Y-%m-%d"),
                    float(row["predicted"]),
                    low,
                    high,
                    made_at,
                ),
            )

            saved += cursor.rowcount

    print(f"Saved {saved} new live forecasts.")
    return saved


def forecast(path=db.DB_PATH, refresh=True):
    """Refresh data and prepare the upcoming EIA forecast."""

    # Step 1: Refresh data for normal live forecasts.
    # Historical tests can disable refresh.
    if refresh:
        print("Refreshing storage...")
        pipeline.refresh_storage(
            pipeline.LOWER_48_SERIES,
            path=path,
        )

        print("Refreshing weather...")
        pipeline.refresh_weather(path=path)


    # Step 2: Read the updated data from SQLite.
    storage = db.read_series(
        pipeline.LOWER_48_SERIES,
        path=path,
    )

    weekly_weather = db.read_degree_days(
        region="US",
        path=path,
    )

    if storage.empty or weekly_weather.empty:
        raise ValueError(
            "Storage or weather data is missing."
        )

    # Step 3: Determine the upcoming target week.
    latest_period = pd.Timestamp(storage["period"].max())
    target = latest_period + pd.Timedelta(days=7)

    if report_is_released(target):
        print(
            f"Skipping forecast: the report for {target.date()} "
            "has already been released."
        )
        return

    print(f"Latest storage week: {latest_period.date()}")
    print(f"Upcoming target week: {target.date()}")

    # Step 4: Check that weather exists for the target.
    target_weather = weekly_weather.loc[
        weekly_weather["week_end"] == target
    ]

    if target_weather.empty:
        raise ValueError(
            f"Missing weather data for target week {target.date()}"
        )

    # Step 5: Add a placeholder for the upcoming week.
    storage_extended = add_upcoming_week(storage)

    # Step 6: Build the LightGBM feature table.
    lightgbm_features = feature_table.build_feature_table(
        storage_extended,
        weekly_weather,
    )

    # Step 7: Build the SARIMAX weather frame.
    storage_with_weather = weather_features.add_weather(
        storage_extended,
        weekly_weather,
    )

    sarimax_weather = storage_with_weather.dropna(
        subset=[
            "hdd_utility_gas_dev",
            "cdd_population_dev",
        ]
    ).copy()


    print("Feature table and SARIMAX weather frame created.")

    # Step 8: Create the forecasting models.
    print("Creating forecasting models...")

    sarimax_forecaster, sarimax_interval = make_sarimax(
        weather=sarimax_weather,
        columns=("hdd_utility_gas_dev", "cdd_population_dev"),
        order=(1, 0, 0),
        refit_every=4,
        use_fourier=True,
    )

    lightgbm_forecaster = make_lightgbm(
        lightgbm_features,
        refit_every=4,
    )

    lightgbm_interval = make_lightgbm_interval(
        lightgbm_features,
        refit_every=4,
        calibrate=True,
    )

    # Step 9: Only pass actual historical observations.
    history = storage_with_weather.loc[
        storage_with_weather["period"] < target
    ].copy()

    # Step 10: Generate point forecasts.
    print("Generating predictions...")

    predictions = {
        "five_year_average": baselines.five_year_average(
            history, target
        ),
        "sarimax": sarimax_forecaster(
            history, target
        ),
        "lightgbm": lightgbm_forecaster(
            history, target
        ),
    }

    # Step 11: Generate 90% prediction intervals.
    sarimax_low, sarimax_high = sarimax_interval(
        history, target
    )

    lightgbm_low, lightgbm_high = lightgbm_interval(
        history, target
    )

    # Step 12: Combine results.
    results = pd.DataFrame([
        {
            "model": "five_year_average",
            "target": target,
            "predicted": predictions["five_year_average"],
            "low": float("nan"),
            "high": float("nan"),
        },
        {
            "model": "sarimax",
            "target": target,
            "predicted": predictions["sarimax"],
            "low": sarimax_low,
            "high": sarimax_high,
        },
        {
            "model": "lightgbm",
            "target": target,
            "predicted": predictions["lightgbm"],
            "low": lightgbm_low,
            "high": lightgbm_high,
        },
    ])


    print("\nLive forecast predictions:")

    # Explain the LightGBM forecast using the same feature table.
    explanations = explain(
        lightgbm_features,
        target,
    )

    # Verify that the contributions reconstruct the saved prediction.
    reconstructed = (
        explanations["base_value"].iloc[0]
        + explanations["contribution"].sum()
    )

    lightgbm_prediction = results.loc[
        results["model"] == "lightgbm",
        "predicted",
    ].iloc[0]

    if not np.isclose(
        reconstructed,
        lightgbm_prediction,
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError(
            "LightGBM explanations do not match the forecast."
        )

    # Save only after the explanation has been validated.
    saved = save_live_forecasts(results, path=path)
    save_live_explanations(explanations, path=path)

    print(f"Forecasting complete. {saved} new rows saved.")

    return results


def score(path=db.DB_PATH):
    """Fill actual weekly changes for previously saved live forecasts."""

    init_live_tables(path)

    # Read the latest available storage observations.
    storage = db.read_series(
        pipeline.LOWER_48_SERIES,
        path=path,
    )

    if storage.empty:
        print("No storage observations available.")
        return 0

    storage = storage.sort_values("period").reset_index(drop=True)

    # Calculate actual weekly storage changes.
    storage["actual_change"] = storage["value"].diff()

    # Create a lookup: target week -> actual change.
    actuals = dict(
        zip(
            storage["period"].dt.strftime("%Y-%m-%d"),
            storage["actual_change"],
        )
    )

    updated = 0

    with closing(sqlite3.connect(path)) as conn, conn:
        # Only find forecasts that haven't been scored yet.
        forecasts = conn.execute(
            """
            SELECT model, target
            FROM live_forecasts
            WHERE actual IS NULL
            """
        ).fetchall()

        for model, target in forecasts:
            actual = actuals.get(target)

            # Skip weeks where actual data is not available.
            if actual is None or pd.isna(actual):
                continue

            cursor = conn.execute(
                """
                UPDATE live_forecasts
                SET actual = ?
                WHERE model = ?
                  AND target = ?
                  AND actual IS NULL
                """,
                (float(actual), model, target),
            )

            updated += cursor.rowcount

    print(f"Scored {updated} live forecasts.")
    return updated


def load_live_forecasts(path=db.DB_PATH):
    """Load saved live forecasts for the dashboard."""

    init_live_tables(path)

    with closing(sqlite3.connect(path)) as conn:
        forecasts = pd.read_sql_query(
            """
            SELECT
                model,
                target,
                predicted,
                low,
                high,
                made_at,
                actual
            FROM live_forecasts
            ORDER BY target DESC, model
            """,
            conn,
        )

    if not forecasts.empty:
        forecasts["target"] = pd.to_datetime(forecasts["target"])

    return forecasts


def load_explanations(path=db.DB_PATH):
    """Load saved LightGBM explanations for the dashboard."""

    init_live_tables(path)

    with closing(sqlite3.connect(path)) as conn:
        explanations = pd.read_sql_query(
            """
            SELECT
                target,
                feature,
                label,
                value,
                contribution,
                base_value
            FROM live_explanations
            ORDER BY target DESC, feature
            """,
            conn,
        )

    if not explanations.empty:
        explanations["target"] = pd.to_datetime(explanations["target"])

    return explanations


def save_live_explanations(explanations, path=db.DB_PATH):
    """Save feature contributions without overwriting existing rows."""

    init_live_tables(path)

    rows_added = 0

    with closing(sqlite3.connect(path)) as conn, conn:
        for row in explanations.itertuples(index=False):
            target = pd.Timestamp(row.target)
            target_str = target.strftime("%Y-%m-%d")

            # Never save an explanation after the report is released.
            if report_is_released(target):
                raise ValueError(
                    f"Cannot save explanations for {target_str}: "
                    "EIA report has already been released."
                )

            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO live_explanations (
                    target,
                    feature,
                    label,
                    value,
                    contribution,
                    base_value
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    target_str,
                    row.feature,
                    row.label,
                    float(row.value),
                    float(row.contribution),
                    float(row.base_value),
                ),
            )

            rows_added += cursor.rowcount

    return rows_added


def main():
    """Command-line interface for live forecasting."""
    import argparse

    parser = argparse.ArgumentParser(
        description="G.A.S.P. live forecasting system"
    )

    parser.add_argument(
        "command",
        choices=["forecast", "score"],
        help="Generate live forecasts or score existing predictions",
    )

    args = parser.parse_args()

    if args.command == "forecast":
        forecast()
    elif args.command == "score":
        score()


if __name__ == "__main__":
    main()
