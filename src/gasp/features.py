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

def add_lags(df, lags=(1, 2, 52)) -> pd.DataFrame:
    df = df.copy()

    for lag in lags:
        df[f"lag_{lag}"] = df["value"].shift(lag)

    return df

# lag_52 refers to 52 observations earlier, not necessarily
# the same ISO week in the previous year. In a 53-week ISO
# year, these can differ.


def add_five_year_stats(df) -> pd.DataFrame:
    df = df.copy()

    calendar = df["period"].dt.isocalendar()
    years = calendar["year"]
    weeks = calendar["week"]

    df["five_year_min"] = float("nan")
    df["five_year_max"] = float("nan")
    df["five_year_mean"] = float("nan")

    for i in range(len(df)):
        year = years.iloc[i]
        week = weeks.iloc[i]

        mask = (
            (weeks == week)
            & (years >= year - 5)
            & (years < year)
        )

        historical = df.loc[mask, "value"]

        if not historical.empty:
            df.loc[df.index[i], "five_year_min"] = historical.min()
            df.loc[df.index[i], "five_year_max"] = historical.max()
            df.loc[df.index[i], "five_year_mean"] = historical.mean()

    return df


