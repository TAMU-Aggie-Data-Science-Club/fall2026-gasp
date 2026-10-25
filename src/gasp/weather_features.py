import pandas as pd
from gasp import weather_weekly


def add_weather(
    storage_df: pd.DataFrame,
    weekly_dd: pd.DataFrame,
    years: int = 10,
) -> pd.DataFrame:
    # Calculate normals only for the US to avoid processing every region.
    us_weekly = weekly_dd.loc[weekly_dd["region"] == "US"].copy()
    weather = weather_weekly.add_normal(us_weekly, years=years)
    weather = weather.sort_values(
        ["measure", "weighting", "week_end"]
    ).reset_index(drop=True)

    # Week 53 borrows the immediately preceding week 52 normal in its series.
    previous = weather.groupby(["measure", "weighting"])[
        ["week_end", "normal"]
    ].shift(1)

    use_week_52 = (
        (weather["week_end"].dt.isocalendar().week == 53)
        & (previous["week_end"].dt.isocalendar().week == 52)
        & (
            weather["week_end"] - previous["week_end"]
            == pd.Timedelta(days=7)
        )
    )

    weather.loc[use_week_52, "normal"] = previous.loc[
        use_week_52, "normal"
    ]
    weather["deviation"] = weather["value"] - weather["normal"]

    # Put each series into its own column, with one row per Friday.
    features = [
        "hdd_utility_gas",
        "hdd_population",
        "cdd_population",
    ]
    weather["feature"] = weather["measure"] + "_" + weather["weighting"]

    values = weather.pivot(
        index="week_end",
        columns="feature",
        values="value",
    ).reindex(columns=features)

    deviations = weather.pivot(
        index="week_end",
        columns="feature",
        values="deviation",
    ).reindex(columns=features).add_suffix("_dev")

    wide = values.join(deviations)
    wide.columns.name = None

    # Keep every storage row, filling unmatched weather columns with NaN.
    return storage_df.join(
        wide,
        on="period",
        how="left",
        validate="many_to_one",
    )