"""Data validation checks for the storage frame and degree-day frames (CONTRACTS.md, section 1).

Each check takes a storage frame and returns a list of problem messages.
An empty list means that check passed. validate_storage runs them all.
This file never fetches or loads data; it only checks what it is given.
"""

import pandas as pd

REQUIRED_COLUMNS = ["period", "series", "value"]

# Largest believable week-to-week change in storage, in Bcf.
# From the real Lower 48 data (2010 to 2026, 875 weekly changes):
#   biggest withdrawal: -359 Bcf (weeks ending 2018-01-05 and 2026-01-30)
#   biggest injection:  +132 Bcf
#   typical week: std about 98 Bcf
# 500 Bcf is about 40% above the record, so a new record cold week
# still passes, but data errors like a dropped or extra digit
# (3500 -> 350 or 35000) or a value from the wrong series get caught.
MAX_WEEKLY_CHANGE = 500.0


def _day(ts) -> str:
    """Format a date as YYYY-MM-DD for problem messages."""
    return "NaT" if pd.isna(ts) else ts.strftime("%Y-%m-%d")


def check_columns(df: pd.DataFrame) -> list[str]:
    """The three contract columns exist with the right types."""
    problems = []
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            problems.append(f"missing column: {col}")

    if "period" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["period"]):
        problems.append(f"wrong type: period is {df['period'].dtype}, expected datetime")
    if "series" in df.columns and not pd.api.types.is_string_dtype(df["series"]):
        problems.append(f"wrong type: series is {df['series'].dtype}, expected str")
    if "value" in df.columns and not pd.api.types.is_float_dtype(df["value"]):
        problems.append(f"wrong type: value is {df['value'].dtype}, expected float")
    return problems


def check_fridays(df: pd.DataFrame) -> list[str]:
    """Every period is a Friday (the last day of the EIA week)."""
    periods = df["period"].dropna()
    bad = periods[periods.dt.dayofweek != 4]
    return [f"not a Friday: {_day(p)} ({p.day_name()})" for p in bad]


def check_duplicates(df: pd.DataFrame) -> list[str]:
    """No (series, period) pair appears more than once."""
    dupes = df[df.duplicated(subset=["series", "period"], keep="first")]
    return [
        f"duplicate row: {_day(p)} ({s})"
        for s, p in zip(dupes["series"], dupes["period"])
    ]


def check_missing_weeks(df: pd.DataFrame) -> list[str]:
    """No Fridays missing between the first and last period of each series."""
    problems = []
    for series, group in df.dropna(subset=["period"]).groupby("series"):
        fridays = group["period"][group["period"].dt.dayofweek == 4]
        if fridays.empty:
            continue
        expected = pd.date_range(fridays.min(), fridays.max(), freq="W-FRI")
        missing = expected.difference(pd.DatetimeIndex(fridays))
        problems += [f"missing week: {_day(p)} ({series})" for p in missing]
    return problems


def check_missing_values(df: pd.DataFrame) -> list[str]:
    """No NaN or NaT in any contract column."""
    problems = []
    for col in REQUIRED_COLUMNS:
        for idx in df.index[df[col].isna()]:
            where = f"row {idx}" if col == "period" else _day(df.at[idx, "period"])
            problems.append(f"missing value: {col} at {where}")
    return problems


def check_negative(df: pd.DataFrame) -> list[str]:
    """Storage can't be below zero."""
    neg = df[df["value"] < 0]
    return [
        f"negative storage: {v:.0f} Bcf at {_day(p)}"
        for p, v in zip(neg["period"], neg["value"])
    ]


def check_jumps(df: pd.DataFrame) -> list[str]:
    """No week-to-week change bigger than MAX_WEEKLY_CHANGE."""
    problems = []
    ordered = df.dropna(subset=["period"]).sort_values("period")
    for series, group in ordered.groupby("series"):
        change = group["value"].diff()
        too_big = change.abs() > MAX_WEEKLY_CHANGE
        for p, c in zip(group.loc[too_big, "period"], change[too_big]):
            problems.append(
                f"impossible jump: {c:+.0f} Bcf into {_day(p)} "
                f"(limit {MAX_WEEKLY_CHANGE:.0f}) ({series})"
            )
    return problems


def check_sorted(df: pd.DataFrame) -> list[str]:
    """Rows are sorted by period, oldest first (CONTRACTS.md, section 1)."""
    periods = df["period"]
    went_back = periods.diff() < pd.Timedelta(0)
    if not went_back.any():
        return []
    first = periods[went_back].iloc[0]
    return [f"not sorted by period: {_day(first)} comes after a later date"]


def validate_storage(df: pd.DataFrame) -> list[str]:
    """Run every check. Returns all problems found; [] means clean."""
    problems = check_columns(df)
    if problems:
        # the other checks need the right columns and types to run
        return problems

    for check in (
        check_fridays,
        check_duplicates,
        check_missing_weeks,
        check_missing_values,
        check_negative,
        check_jumps,
        check_sorted,
    ):
        problems += check(df)
    return problems



# A weekly degree-day total sums seven daily values. A daily
# degree-day value of 100 already implies an extremely large
# temperature departure from the 65°F baseline, since
# 2000 the highest weekly total the CPC has reported is 451,
#  which 700 is 55% above.
MAX_WEEKLY_DEGREE_DAYS = 700.0


def validate_degree_days(df: pd.DataFrame) -> list[str]:
    """Check the weekly degree-day contract; return problems or []."""
    problems = []
    required = ["week_end", "region", "measure", "weighting", "value"]

    # 1. Required columns
    for col in required:
        if col not in df.columns:
            problems.append(f"missing column: {col}")

    # 2. Column types
    if "week_end" in df and not pd.api.types.is_datetime64_any_dtype(df["week_end"]):
        problems.append(f"wrong type: week_end is {df['week_end'].dtype}, expected datetime")

    for col in ["region", "measure", "weighting"]:
        if col in df and not pd.api.types.is_string_dtype(df[col]):
            problems.append(f"wrong type: {col} is {df[col].dtype}, expected str")

    if "value" in df and not pd.api.types.is_float_dtype(df["value"]):
        problems.append(f"wrong type: value is {df['value'].dtype}, expected float")

    if problems:
        return problems

    # 3. Missing values
    for col in required:
        for idx in df.index[df[col].isna()]:
            problems.append(f"missing value: {col} at row {idx}")

    keys = ["week_end", "region", "measure", "weighting"]
    group_keys = ["region", "measure", "weighting"]

    # 4. Friday check
    for date in df["week_end"].dropna():
        if date.dayofweek != 4:
            problems.append(
                f"not a Friday: {date.strftime('%Y-%m-%d')} ({date.day_name()})"
            )

    # 5. Duplicate combinations
    duplicates = df[df.duplicated(subset=keys, keep="first")]
    for _, row in duplicates.iterrows():
        problems.append(
            #implemented the _day function for duplicate combinations
            f"duplicate row: {_day(row['week_end'])} "
            f"({row['region']} {row['measure']} {row['weighting']})"
        )

    # 6. Missing weeks within each combination
    valid_dates = df.dropna(subset=keys)

    for (region, measure, weighting), group in valid_dates.groupby(group_keys):
        fridays = group.loc[
            group["week_end"].dt.dayofweek == 4, "week_end"
        ]
        if fridays.empty:
            continue

        expected = pd.date_range(
            fridays.min(), fridays.max(), freq="W-FRI"
        )
        missing = expected.difference(pd.DatetimeIndex(fridays))

        for date in missing:
            problems.append(
                f"missing week: {date.strftime('%Y-%m-%d')} "
                f"({region} {measure} {weighting})"
            )

    # 7. Allowed measure and weighting values
    for idx, row in df.iterrows():
        if pd.notna(row["measure"]) and row["measure"] not in {"hdd", "cdd"}:
            problems.append(f"invalid measure: {row['measure']} at row {idx}")

        if pd.notna(row["weighting"]) and row["weighting"] not in {
            "population", "utility_gas"
        }:
            problems.append(f"invalid weighting: {row['weighting']} at row {idx}")

    # 8. Negative and impossible weekly totals
    for idx, row in df.iterrows():
        value = row["value"]
        if pd.isna(value):
            continue
        if value < 0:
            problems.append(f"negative degree days: {value} at row {idx}")
        elif value > MAX_WEEKLY_DEGREE_DAYS:
            problems.append(
                f"impossible weekly total: {value} at row {idx} "
                f"(limit {MAX_WEEKLY_DEGREE_DAYS})"
            )

    return problems
