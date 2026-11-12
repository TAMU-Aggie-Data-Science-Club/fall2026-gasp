from datetime import date, timedelta
import numpy as np
import pandas as pd

def add_calendar(df) -> pd.DataFrame:
    df = df.copy()

    calendar = df["period"].dt.isocalendar()

    df["iso_year"] = calendar["year"]
    df["iso_week"] = calendar["week"]

    return df

def add_change(df) -> pd.DataFrame:
    df = df.copy()

    df["weekly_change"] = df["value"].diff()

    return df

def add_lags(df, lags=(1, 2, 52), column="value"):
    result = df.copy()

    if column not in result.columns:
        raise ValueError(f"Column '{column}' not found")

    for lag in lags:
        if column == "value":
            name = f"lag_{lag}"
        else:
            name = f"{column}_lag_{lag}"

        result[name] = result[column].shift(lag)

    return result

# lag_52 refers to 52 observations earlier, not necessarily
# the same ISO week in the previous year. In a 53-week ISO
# year, these can differ.


def add_five_year_stats(df):
    result = df.copy()

    result["period"] = pd.to_datetime(result["period"])

    iso = result["period"].dt.isocalendar()
    result["iso_year"] = iso.year.astype(int)
    result["iso_week"] = iso.week.astype(int)

    result["five_year_min"] = np.nan
    result["five_year_max"] = np.nan
    result["five_year_mean"] = np.nan

    for i, row in result.iterrows():
        year = row["iso_year"]
        week = row["iso_week"]

        historical = result.loc[
            (result["iso_week"] == week)
            & (result["iso_year"] >= year - 5)
            & (result["iso_year"] < year),
            "value"
        ].dropna()

        if historical.size == 5:
            result.at[i, "five_year_min"] = historical.min()
            result.at[i, "five_year_max"] = historical.max()
            result.at[i, "five_year_mean"] = historical.mean()

    return result


def add_seasonal_change(df) -> pd.DataFrame:
    result = add_change(df)
    result = add_calendar(result)

    result["seasonal_change"] = np.nan

    for i, row in result.iterrows():
        year = int(row["iso_year"])
        week = int(row["iso_week"])

        if week == 53:
            week = 52

        historical = result.loc[
            (result["iso_week"] == week)
            & (result["iso_year"] >= year - 5)
            & (result["iso_year"] < year),
            "weekly_change"
        ].dropna()

        if len(historical) == 5:
            result.at[i, "seasonal_change"] = historical.mean()

    return result


def add_surplus(df) -> pd.DataFrame:
    result = add_five_year_stats(df)

    result["surplus_lag_1"] = (
        result["value"] - result["five_year_mean"]
    ).shift(1)

    return result


def add_holiday_week(df) -> pd.DataFrame:
    result = df.copy()

    #Check whether the week contains a major holiday.
    def is_holiday_week(period):
        end = pd.Timestamp(period).date()
        start = end - timedelta(days=6)

        for year in range(start.year, end.year + 1):
            #Find the fourth Thursday of November.
            november_1 = date(year, 11, 1)
            first_thursday = november_1 + timedelta(
                days=(3 - november_1.weekday()) % 7
            )
            thanksgiving = first_thursday + timedelta(weeks=3)

            holidays = [
                thanksgiving,
                date(year, 12, 25),
                date(year, 1, 1),
            ]

            if any(start <= holiday <= end for holiday in holidays):
                return 1

        return 0

    result["holiday_week"] = result["period"].apply(is_holiday_week)

    return result
