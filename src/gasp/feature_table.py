
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
    "cdd_population_dev"
]


#Combines storage and weather data into one feature table.
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

    #Calculate five-year storage statistics.
    df = features.add_five_year_stats(df)

    #Calculate last week's storage surplus.
    df = features.add_surplus(df)

    #Calculate the seasonal weekly change.
    df = features.add_seasonal_change(df)

    #Keep the target and required forecasting features.
    return df[["period", "weekly_change"] + FEATURE_COLUMNS]