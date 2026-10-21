from __future__ import annotations
import os
import pandas as pd
import requests
from dotenv import load_dotenv


# This function searches for your .env file and loads the variables inside
load_dotenv()

#Created a helper function 
def _get_api_key() -> str:
    api_key = os.getenv("EIA_API_KEY")
    if not api_key:
        raise ValueError("The EIA API key is missing from the .env file!")
    return api_key


def fetch_series(series_id: str, start: str | None = None, end: str | None = None, page_size: int = 5000) -> pd.DataFrame:
    # Looks at the variable name in the .env file
    api_key = _get_api_key()
#URL for EIA API and puts it into all_rows
    url = f"https://api.eia.gov/v2/natural-gas/stor/wkly/data/"
    offset = 0
    all_rows = []

    while True:
        params = {
            "api_key": api_key,
            "frequency": "weekly",
            "data[0]": "value",
            "facets[series][]": series_id,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": page_size
        }
        #Checks if a date is entered so it can be sent to EIA server
        if start is not None:
            params["start"] = start
        if end is not None:
            params["end"] = end

        #This will request a strict timeout to prevent hanging
        response = requests.get(url, params=params, timeout=30)
        #Making an exception if the request fails
        response.raise_for_status()
        #Reads this page's rows and the total number of rows EIA has
        body = response.json().get("response", {})
        rows = body.get("data", [])
        total = int(body.get("total", 0))

        all_rows.extend(rows)
        offset += len(rows)

        #Stops when EIA returns nothing or every row has been collected
        if not rows or offset >= total:
            break
    #Stops the project from crashing by stopping any queries that have no results
    if not all_rows:
        return pd.DataFrame({
            "period": pd.Series(dtype="datetime64[ns]"),
            "series": pd.Series(dtype="object"),
            "value": pd.Series(dtype="float64")
        })
    #Formats data into dataframe by converting it and sorting from oldest to newest based on time period
    df = pd.DataFrame(all_rows)[["period", "series", "value"]]
    df["period"] = pd.to_datetime(df["period"])
    df["series"] = df["series"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").astype(float)
    df = df.sort_values("period").reset_index(drop=True)
    return df
#Gets the most recent file and checks if our data is up to date wthout having to download the entire file
def latest_period(route: str = "natural-gas/stor/wkly") -> pd.Timestamp:
    api_key = _get_api_key()
    url = f"https://api.eia.gov/v2/{route.strip('/')}/"

    response = requests.get(url, params={"api_key": api_key}, timeout=10)
    response.raise_for_status()

    end_period = response.json().get("response", {}).get('endPeriod')
    if not end_period:
        raise ValueError(f"EIA metadata for '{route}' has no endPeriod")
    return pd.Timestamp(end_period)