import pandas as pd
from gasp import features, weather_features

#Features used to predict weekly natural gas storage changes.
FEATURE_COLUMNS = [
    "iso_week",
    "weekly_change_lag_1",
    "weekly_change_lag_2",
    "weekly_change_lag_52",
    "lag_1",
    "surplus_lag_1",
    "seasonal_change",
    "hdd_utility_gas",
    "hdd_population",
    "cdd_population",
    "hdd_utility_gas_dev",
    "hdd_population_dev",
    "cdd_population_dev",
    "hdd_utility_gas_dev_lag_1",
    "cdd_population_dev_lag_1",
    "holiday_week",
    "hdd_div1",
    "hdd_div1_dev",
    "hdd_div2",
    "hdd_div2_dev",
    "hdd_div3",
    "hdd_div3_dev",
    "hdd_div7",
    "hdd_div7_dev",
]


#Combines storage and weather data into one feature table.
#weekly_dd must include the regional rows (read it with region=None).
def build_feature_table(storage_df, weekly_dd) -> pd.DataFrame:

    #Sort storage data by week.
    df = storage_df.sort_values("period").reset_index(drop=True).copy()
    df["period"] = pd.to_datetime(df["period"])

    #Add the ISO week and weekly storage change.
    df = features.add_calendar(df)
    df = features.add_change(df)

    #Add previous storage values.
    df = features.add_lags(df, lags=(1,))

    #Add previous weekly changes.
    df = features.add_lags(
        df, lags=(1, 2, 52), column="weekly_change"
    )

    #Add the six weather features.
    df = weather_features.add_weather(df, weekly_dd)
        #Add last week's weather deviations.
    df["hdd_utility_gas_dev_lag_1"] = (
        df["hdd_utility_gas_dev"].shift(1)
    )

    df["cdd_population_dev_lag_1"] = (
        df["cdd_population_dev"].shift(1)
    )

    #Add regional heating weather for divisions 1, 2, 3, and 7.
    df = weather_features.add_regional_weather(df, weekly_dd)

    #Mark major holiday weeks.
    df = features.add_holiday_week(df)

    #Calculate five-year storage statistics.
    df = features.add_five_year_stats(df)

    #Calculate last week's storage surplus.
    df = features.add_surplus(df)

    #Calculate the seasonal weekly change.
    df = features.add_seasonal_change(df)

    #Keep the target and required forecasting features.
    return df[["period", "weekly_change"] + FEATURE_COLUMNS]


FEATURE_LABELS = {
    "iso_week": "Week of the year",
    "weekly_change_lag_1": "Last week's storage change",
    "weekly_change_lag_2": "Storage change two weeks ago",
    "weekly_change_lag_52": "Storage change 52 weeks ago",
    "lag_1": "Last week's storage level",
    "surplus_lag_1": "Last week's storage surplus",
    "seasonal_change": "Average change for this week over 5 years",
    "hdd_utility_gas": "Gas-weighted heating degree days",
    "hdd_population": "Population-weighted heating degree days",
    "cdd_population": "Population-weighted cooling degree days",
    "hdd_utility_gas_dev": "Gas-weighted heating vs normal",
    "hdd_population_dev": "Population-weighted heating vs normal",
    "cdd_population_dev": "Population-weighted cooling vs normal",
    "hdd_utility_gas_dev_lag_1": "Last week's gas-weighted heating deviation",
    "cdd_population_dev_lag_1": "Last week's population-weighted cooling deviation",
    "holiday_week": "Thanksgiving, Christmas, or New Year's week",
    "hdd_div1": "New England heating degree days",
    "hdd_div1_dev": "New England heating vs normal",
    "hdd_div2": "Mid-Atlantic heating degree days",
    "hdd_div2_dev": "Mid-Atlantic heating vs normal",
    "hdd_div3": "Great Lakes heating degree days",
    "hdd_div3_dev": "Great Lakes heating vs normal",
    "hdd_div7": "Texas and Oklahoma heating degree days",
    "hdd_div7_dev": "Texas and Oklahoma heating vs normal",
}