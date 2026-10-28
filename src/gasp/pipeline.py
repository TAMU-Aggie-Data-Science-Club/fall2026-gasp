from __future__ import annotations
from pathlib import Path
from typing import Iterable
import pandas as pd
from gasp import db, eia, runlog, validate

#Deault series ID for the Lower 48 states
LOWER_48_SERIES = "NW2_EPG0_SWO_R48_BCF"

def _normalize_series_ids(series_ids: str | Iterable[str]) -> list[str]:
    
    """Allows passing either a single series string or a list of series strings."""
    #This is if the user accidently passes a single string, it'll wrap it in a list so our loops don't interate over the individual letters
    if isinstance(series_ids, str):
        return [series_ids]
    return list(series_ids)

def _process_single_series(
        series_id: str,
        start: str | None,
        end: str | None,
        path: str | Path,
) -> int:
    
    """Fetch, validate, store, and log a single EIA series."""
    #First it downloads the data from EIA and then counts how many rows there is
    #If the download itself fails, log a failed receipt before raising the error
    try:
        df = eia.fetch_series(series_id, start=start, end=end)
    except Exception as e:
        runlog.log_run(
            series=series_id,
            start_date=start,
            end_date=end,
            rows_fetched=0,
            new_rows_stored=0,
            problems=[f"fetch failed: {e}"],
            success=False,
            path=path,
        )
        raise
    rows_fetched = len(df)
    
    #This asks validate.py to check if there are any data errors in the table
    problems = validate.validate_storage(df)

    #If the problem list is not empty, it logs a failed receipt and then crashes intentionally
    if problems:
        runlog.log_run(
            series=series_id,
            start_date=start,
            end_date=end,
            rows_fetched=rows_fetched,
            new_rows_stored=0, #it'll be zero since we aborted
            problems=problems,
            success=False, #Marks it as failed in the receipt
            path=path,

        )
        #Makes sure to stop the script so bad data does not get saved
        raise ValueError(f"Validation failed for {series_id}: {problems}")
    
    #Now if there are no problems, then we can save it to the databse and log a success receipt.
    new_rows = db.write_series(df, path=path)
    runlog.log_run(
        series=series_id,
        start_date=start,
        end_date=end,
        rows_fetched=rows_fetched,
        new_rows_stored=new_rows,
        problems=[],
        success=True,
        path=path,
    )
    return new_rows
def backfill_storage(
        series_ids: str | Iterable[str],
        path: str | Path = db.DB_PATH,
) -> int:
    """Pull full history with eia.fetch_series, validate, and store with db.write_series"""
    total_new_rows = 0
    for series_id in _normalize_series_ids(series_ids):

        #By doing start & end equal to non we tell the helper function to download the entire dataset from the EIA. 
        #Additionally the helper function hadles the fetching, validating, and saving.
        total_new_rows += _process_single_series(
            series_id=series_id,
            start=None,
            end=None,
            path=path,
        )
    return total_new_rows
def refresh_storage(
        series_ids: str | Iterable[str],
        lookback_weeks: int=8,
        path: str | Path = db.DB_PATH,
) -> int:
    """Pull only from lookback_weeks before newest stored period up to eia.latest_period()."""
   
    
    #this asks EIA for the newest available report date and format as YYYY-MM-DD
    end_ts = eia.latest_period()
    end_str = end_ts.strftime("%Y-%m-%d")

    total_new_rows = 0
    for series_id in _normalize_series_ids(series_ids):
        
        # Read what is currently in our databse for this series
        existing_df = db.read_series(series_id, path=path)
        if existing_df.empty:
            start_str = None
        else:
            
            #This finds the newest stored date, subtracts 8 weeks, and then formats it into YYYY-MM-DD
            newest_stored = pd.to_datetime(existing_df["period"]).max()
            
            #This finds the most recent date in our databse and rewinds it back 8 weeks of data from the EIA to catch any recent data revisions
            start_ts = newest_stored - pd.Timedelta(weeks=lookback_weeks)
            start_str = start_ts.strftime("%Y-%m-%d")
        
        #Pass the overlapping date window into the helper function
        total_new_rows += _process_single_series(
            series_id=series_id,
            start=start_str,
            end=end_str,
            path=path,
        )
    return total_new_rows

#This is so that when python runs -m gasp.pipeline it will automatically refresh_storage for the Lower 48 series
if __name__ == "__main__":
    refresh_storage([LOWER_48_SERIES], path=db.DB_PATH)