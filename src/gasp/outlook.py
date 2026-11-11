"""Project weekly gas storage through March under past winters' weather."""

from contextlib import closing
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from gasp import db, pipeline, sarimax, weather_features


_WEATHER_COLUMNS = [column.removesuffix("_dev") for column in sarimax.DEFAULT_WEATHER_COLUMNS]
_NORMAL_YEARS = 10
_SCENARIOS = range(2001, 2027)
_OUTLOOK_COLUMNS = ["scenario", "period", "storage"]


def _last_friday_of_march(year: int) -> pd.Timestamp:
    return pd.offsets.Week(weekday=4).rollback(pd.Timestamp(year=year, month=3, day=31))


def _future_fridays(latest_week: pd.Timestamp) -> pd.DatetimeIndex:
    """List the Fridays after the latest storage week through March."""
    latest_week = pd.Timestamp(latest_week)
    if pd.isna(latest_week) or latest_week.dayofweek != 4:
        raise ValueError("The latest storage week must end on a Friday.")

    march_year = latest_week.year + (latest_week.month > 3)
    last_friday = _last_friday_of_march(march_year)
    return pd.date_range(latest_week + pd.Timedelta(weeks=1), last_friday, freq="W-FRI")


def _scenario_weather(
    weekly_dd: pd.DataFrame, latest_week: pd.Timestamp, scenarios=_SCENARIOS,
) -> pd.DataFrame:
    """Build the raw weather and deviations for each winter and future Friday."""
    latest_week = pd.Timestamp(latest_week)
    periods = _future_fridays(latest_week)

    # Key the two model inputs by ISO year and week, keeping their source Friday.
    us = weekly_dd.loc[weekly_dd["region"] == "US"].copy()
    us["week_end"] = pd.to_datetime(us["week_end"])
    us["feature"] = us["measure"] + "_" + us["weighting"]
    us = us.loc[us["feature"].isin(_WEATHER_COLUMNS)]
    raw = us.pivot(index="week_end", columns="feature", values="value")
    raw = raw.reindex(columns=_WEATHER_COLUMNS)
    raw.columns.name = None
    iso = raw.index.isocalendar()
    raw = raw.assign(iso_year=iso["year"], iso_week=iso["week"])
    raw = raw.reset_index(names="source_week").set_index(["iso_year", "iso_week"])

    # Every scenario uses the same ten complete years at the outlook's start.
    # The week 53 fallback applies to both the source weather and the normal.
    iso_week = periods.isocalendar()["week"].replace(53, 52).astype(int)
    years = range(latest_week.year - _NORMAL_YEARS, latest_week.year)
    recent = raw.loc[raw.index.get_level_values("iso_year").isin(years), _WEATHER_COLUMNS]
    grouped = recent.groupby(level="iso_week")
    normals = grouped.mean().where(grouped.count() == _NORMAL_YEARS)
    normals = normals.reindex(iso_week.to_numpy()).set_axis(periods)
    missing = normals.isna().any(axis=1)
    if missing.any():
        weeks = iso_week[missing].unique().tolist()
        raise ValueError(f"Need ten complete years of weather for ISO weeks {weeks}.")

    # The scenario ID is its ending year. July through December use the prior year.
    rows = []
    for scenario in scenarios:
        source_year = scenario - (iso_week >= 27).astype(int)
        keys = pd.MultiIndex.from_arrays([source_year, iso_week], names=raw.index.names)
        source = raw.reindex(keys)
        missing = source[_WEATHER_COLUMNS].isna().any(axis=1)
        if missing.any():
            year, week = keys[missing.to_numpy()][0]
            raise ValueError(f"Missing weather for scenario {scenario}, ISO {year}-W{week:02}.")

        source.index = periods.rename("period")
        deviations = (source[_WEATHER_COLUMNS] - normals).add_suffix("_dev")
        scenario_weather = source.join(deviations).reset_index()
        scenario_weather.insert(0, "scenario", scenario)
        rows.append(scenario_weather)

    return pd.concat(rows, ignore_index=True)


def build_winter_outlook(
    storage_df: pd.DataFrame, weekly_dd: pd.DataFrame, *, scenarios=_SCENARIOS,
    fit_before: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fit once and project a storage path for each selected winter.

    For a historical replay, fit_before excludes that date and later dates from
    fitting. Observations through the starting week update the state without refitting.
    """
    scenarios = tuple(scenarios)
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("Supply at least one distinct winter scenario.")

    history = storage_df.loc[:, ["period", "value"]].copy()
    history["period"] = pd.to_datetime(history["period"])
    history = history.sort_values("period").reset_index(drop=True)
    levels = history["value"].to_numpy(dtype=float)
    if len(levels) < 11 or not np.isfinite(levels).all():
        raise ValueError("Need at least 11 valid weekly storage levels.")
    periods = history["period"]
    if periods.isna().any() or not periods.diff().iloc[1:].eq(pd.Timedelta(weeks=1)).all():
        raise ValueError("Storage history must have one observation per consecutive week.")

    latest_week = periods.iloc[-1]
    future = _scenario_weather(weekly_dd, latest_week, scenarios)
    anchors = pd.DataFrame({"scenario": scenarios, "period": latest_week, "storage": levels[-1]})
    if future.empty:
        return anchors.sort_values("scenario", ignore_index=True)

    # The model fits changes. The first storage level has no preceding change.
    changes = np.diff(levels)
    change_weeks = periods.iloc[1:]
    historical_weather = weather_features.add_weather(history, weekly_dd).set_index("period")
    historical_inputs = sarimax.build_sarimax_inputs(historical_weather, change_weeks)
    fit_count = len(changes)
    if fit_before is not None:
        fit_count = int((change_weeks < pd.Timestamp(fit_before)).sum())
        if fit_count < 10:
            raise ValueError("Need at least 10 weekly storage changes before fit_before.")

    fitted = sarimax.fit_sarimax(changes[:fit_count], historical_inputs[:fit_count])
    if fit_count < len(changes):
        fitted = fitted.append(changes[fit_count:], exog=historical_inputs[fit_count:], refit=False)

    # Reuse the fitted state. Each scenario supplies a different future weather path.
    paths = [anchors]
    for scenario, rows in future.groupby("scenario", sort=False):
        weather = rows.set_index("period")
        inputs = sarimax.build_sarimax_inputs(weather, weather.index, start=len(changes))
        forecast = fitted.get_forecast(steps=len(weather), exog=inputs)
        projected = levels[-1] + np.cumsum(np.asarray(forecast.predicted_mean, dtype=float))
        if not np.isfinite(projected).all():
            raise ValueError(f"Invalid storage projection for winter ending {scenario}.")
        paths.append(pd.DataFrame({"scenario": scenario, "period": weather.index, "storage": projected}))

    return pd.concat(paths, ignore_index=True).sort_values(["scenario", "period"], ignore_index=True)


def summarize_outlook(outlook: pd.DataFrame, *, path=db.DB_PATH) -> dict:
    """Report March percentiles, the five-year actual average, and weather extremes.

    Read historical context from the same database as the outlook. Rank winters
    by total gas-weighted heating degree days over the projected weeks.
    """
    if outlook.empty:
        raise ValueError("Build or load a winter outlook before summarizing it.")

    latest_week = pd.Timestamp(outlook["period"].min())
    endpoints = outlook.sort_values("period").groupby("scenario")["storage"].last()
    p10, p50, p90 = endpoints.quantile([0.10, 0.50, 0.90]).tolist()

    last_march_year = latest_week.year
    if latest_week < _last_friday_of_march(last_march_year):
        last_march_year -= 1
    march_weeks = [_last_friday_of_march(year) for year in range(last_march_year - 4, last_march_year + 1)]
    actual = db.read_series(pipeline.LOWER_48_SERIES, path=path).set_index("period")["value"]
    march_levels = actual.reindex(march_weeks)
    if march_levels.isna().any():
        raise ValueError("Need the five most recent actual March endpoints to summarize the outlook.")

    weekly_dd = db.read_degree_days(region="US", path=path)
    weather = _scenario_weather(weekly_dd, latest_week, endpoints.index)
    hdd_totals = weather.groupby("scenario")["hdd_utility_gas"].sum()
    warmest = coldest = None
    if not hdd_totals.empty:
        warmest_year, coldest_year = hdd_totals.idxmin(), hdd_totals.idxmax()
        warmest = f"{warmest_year - 1}–{warmest_year % 100:02d}"
        coldest = f"{coldest_year - 1}–{coldest_year % 100:02d}"

    return {
        "p10": p10, "p50": p50, "p90": p90,
        "five_year_average": float(march_levels.mean()),
        "warmest_scenario": warmest, "coldest_scenario": coldest,
    }


def save_outlook(outlook: pd.DataFrame, path=db.DB_PATH) -> None:
    """Replace the saved outlook in one transaction."""
    data = outlook.loc[:, _OUTLOOK_COLUMNS]
    if data.isna().any().any() or not np.isfinite(data["storage"].to_numpy(dtype=float)).all():
        raise ValueError("Outlook rows must have a scenario, date, and finite storage level.")
    rows = [
        (int(scenario), pd.Timestamp(period).isoformat(), float(storage))
        for scenario, period, storage in data.itertuples(index=False, name=None)
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS winter_outlook (
                scenario INTEGER NOT NULL,
                period TEXT NOT NULL,
                storage REAL NOT NULL,
                PRIMARY KEY (scenario, period)
            )
        """)
        connection.execute("DELETE FROM winter_outlook")
        connection.executemany(
            "INSERT INTO winter_outlook (scenario, period, storage) VALUES (?, ?, ?)", rows,
        )


def load_outlook(path=db.DB_PATH) -> pd.DataFrame:
    """Read the saved paths in scenario and date order."""
    path = Path(path).resolve()
    if path.exists():
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'winter_outlook'"
            ).fetchone()
            if exists:
                return pd.read_sql_query(
                    "SELECT scenario, period, storage FROM winter_outlook ORDER BY scenario, period",
                    connection, parse_dates=["period"],
                ).astype({"scenario": int, "storage": float})

    return pd.DataFrame({
        "scenario": pd.Series(dtype=int),
        "period": pd.Series(dtype="datetime64[ns]"),
        "storage": pd.Series(dtype=float),
    })
