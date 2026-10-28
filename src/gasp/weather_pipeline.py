from datetime import datetime
import pandas as pd
from gasp import weather_cpc, weather_weekly

def build_weekly_degree_days(start_year: int, end_year: int = None) -> pd.DataFrame:
    #create a blank dataframe to hold the other dataframes
    weekly_dfs = []
    #check if there is a specified end year, if not set it to the current year
    if end_year is None:
        end_year = datetime.now().year;
    #list of the files needed 
    files = [ "UtilityGas.Heating.txt", "Population.Heating.txt", "Population.Cooling.txt"]
    for file in files:
        #daily data frame that holds the fetched data from each file 
        daily_df = weather_cpc.fetch_cpc_daily(file , start_year , end_year)
        #convert the daily data to weekly eia
        weekly_df = weather_weekly.to_eia_weeks(daily_df)
        #add the weekly data frame to the blank dataframe
        weekly_dfs.append(weekly_df)
    #one dataframe to use pandas to combine all the dataframes in weekly_dfs, ignoring index to remove the indexes applied in the previous dataframe
    total_df = pd.concat(weekly_dfs , ignore_index = True)
    #sort by week_end, resetting the index to cleanly have labels after sorting by week
    total_df = total_df.sort_values("week_end").reset_index(drop = True)
    return total_df


