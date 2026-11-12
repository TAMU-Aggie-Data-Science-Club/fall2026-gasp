import pandas as pd
from gasp import weather_weekly

# Census divisions used as regional inputs: where gas heating is concentrated
# (1-3) and where much of the gas is produced (7)
REGIONAL_DIVISIONS = {
    "1": "New England",
    "2": "Middle Atlantic",
    "3": "East North Central",
    "7": "West South Central",
}


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


def add_regional_weather(
    storage_df: pd.DataFrame,
    weekly_dd: pd.DataFrame,
    regions=tuple(REGIONAL_DIVISIONS),
    years: int = 10,
) -> pd.DataFrame:
    """Add gas-weighted HDD and its deviation from normal for each region.

    Adds hdd_div<region> and hdd_div<region>_dev columns, joined on period.
    weekly_dd must include the regional rows (read it with region=None).
    Normals follow the same rules as add_weather, including week 53 using
    the preceding week 52's normal.
    """
    frames = []
    for region in regions:
        rows = weekly_dd.loc[
            (weekly_dd["region"] == region)
            & (weekly_dd["measure"] == "hdd")
            & (weekly_dd["weighting"] == "utility_gas")
        ].copy()
        if rows.empty:
            raise ValueError(
                f"No gas-weighted HDD for region {region}; "
                "read degree days with region=None"
            )

        # One region at a time keeps add_normal fast
        rows = weather_weekly.add_normal(rows, years=years)
        rows = rows.sort_values("week_end").reset_index(drop=True)

        # Week 53 borrows the immediately preceding week 52 normal
        previous = rows[["week_end", "normal"]].shift(1)
        use_week_52 = (
            (rows["week_end"].dt.isocalendar().week == 53)
            & (previous["week_end"].dt.isocalendar().week == 52)
            & (rows["week_end"] - previous["week_end"] == pd.Timedelta(days=7))
        )
        rows.loc[use_week_52, "normal"] = previous.loc[use_week_52, "normal"]

        frames.append(
            pd.DataFrame(
                {
                    f"hdd_div{region}": rows["value"].to_numpy(),
                    f"hdd_div{region}_dev": (rows["value"] - rows["normal"]).to_numpy(),
                },
                index=rows["week_end"],
            )
        )

    wide = pd.concat(frames, axis=1)

    # Keep every storage row, filling unmatched weather columns with NaN.
    return storage_df.join(
        wide,
        on="period",
        how="left",
        validate="many_to_one",
    )