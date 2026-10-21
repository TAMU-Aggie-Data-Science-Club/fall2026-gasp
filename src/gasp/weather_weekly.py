import pandas as pd

def to_eia_weeks(daily: pd.DataFrame) -> pd.DataFrame:
    # Sum Saturday-Friday weeks, labeled by Friday, separately for each region,
    # measure and weighting. Count valid daily values to identify complete weeks.
    weekly = (
        daily.set_index("date")
        .groupby(["region", "measure", "weighting"])["value"]
        .resample("W-FRI", closed="right", label="right")
        .agg(["sum", "count"])
        .reset_index()
    )

    # Keep only complete seven-day weeks. Partial totals look artificially mild
    # and would distort the weather history used in later comparisons.
    weekly = weekly.loc[weekly["count"] == 7]

    # Return the contract's weekly columns, ordered from oldest to newest.
    weekly = weekly.rename(
        columns={"date": "week_end", "sum": "value"}
    )

    return (
        weekly[["week_end", "region", "measure", "weighting", "value"]]
        .sort_values("week_end")
        .reset_index(drop=True)
    )

def add_normal(weekly: pd.DataFrame, years: int = 10) -> pd.DataFrame:
    if years < 1:
        raise ValueError("years must be at least 1")

    # Prepare a copy with Friday's calendar year and ISO week number to locate
    # the matching historical weeks without changing the caller's table.
    result = weekly.copy()
    result["_year"] = result["week_end"].dt.year
    result["_iso_week"] = result["week_end"].dt.isocalendar().week

    normals = []

    # Compare each row with the same weather series and ISO week in exactly the
    # requested prior years. For example, a 2020 row with years=10 uses 2010-2019;
    # its own year, future years and older replacement years are excluded.
    for _, row in result.iterrows():
        history = result.loc[
            (result["region"] == row["region"])
            & (result["measure"] == row["measure"])
            & (result["weighting"] == row["weighting"])
            & (result["_iso_week"] == row["_iso_week"])
            & (result["_year"] >= row["_year"] - years)
            & (result["_year"] < row["_year"])
        ]

        # Average only a complete history; missing years or values leave normal as NaN.

        # ISO weeks belong to the year containing at least four of their days.
        # Around New Year, the next week may already count as week 1 of the next
        # year leaving no week 53 to use in the historical comparison, thus causing it to be NaN.
        if (
            history["_year"].nunique() == years
            and history["value"].count() == years
        ):
            normals.append(history["value"].mean())
        else:
            normals.append(float("nan"))

    # Add the historical average and deviation, then remove the helper columns.
    # Positive HDD deviation means colder than usual. A missing normal also
    # gives a missing deviation; the original weekly value stays available.
    result["normal"] = normals
    result["deviation"] = result["value"] - result["normal"]

    return result.drop(columns=["_year", "_iso_week"])
