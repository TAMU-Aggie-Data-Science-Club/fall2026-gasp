from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import pandas as pd
from gasp import db

def init_runlog(path: str | Path = db.DB_PATH) -> None:
    """Create the runs table in the SQLite file if it doesn't exist."""

    #Converts path into a Path object and makes sure that the/a parent folder exists
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    #This will connect to SQLite database file and create a table caled "runs" if it does not already exist
    with sqlite3.connect(db_path) as conn:

        #Sets up an empty table with these coloumns
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
            run_time TEXT,
            series TEXT,
            start_date TEXT,
            end_date TEXT,
            rows_fetched INTEGER,
            new_rows_stored INTEGER,
            problems TEXT,
            success INTEGER
            )
            """
        )

#Highlights the exact parameter and tells Python the ones with | None, that it's ok for them to be blank
def log_run(
        series: str,
        start_date: str | None,
        end_date: str | None,
        rows_fetched: int,
        new_rows_stored: int,
        problems: list[str] | str | None,
        success: bool,
        path: str | Path = db.DB_PATH,
) -> None:
    """Write a single run entry (successful or failed) to the runs table."""

    #this makes sure that the run table actually exists before inserting a new row
    init_runlog(path)

    #this records the exact UTC timestamp of this run and then returns it as a string
    run_time = datetime.now(timezone.utc).isoformat()

    #this will convert the problems list into a semicolon-seperated string for SQLite
    if isinstance(problems, list):
        problems_str = "; ".join(str(p) for p in problems)
    elif problems is None:
        problems_str = ""
    else:
        #This makes sure that if a unexpected data type gets passed through the function, the str() changes it into a text so it doesn't crash the database
        problems_str = str(problems)

    #This will open the database and add a row of receipt data using ? as a placeholder
    with sqlite3.connect(path) as conn:

        #SQL command that creates these empty buckets that are filled with ? untill a value is entered and formats it in sequential order
        conn.execute(
            """
            INSERT INTO runs ( run_time, series, start_date, end_date, rows_fetched, new_rows_stored, problems, success
            )
            Values (?,?,?,?,?,?,?,?)
            """,
            (
                run_time,
                series,
                start_date,
                end_date,
                int(rows_fetched),
                int(new_rows_stored),
                problems_str,
                int(bool(success)),
            ),
        )

#sorts the last 10 entries and automatically builds a Pandas DataFrame
def last_runs(n: int = 10, path: str | Path = db.DB_PATH) -> pd.DataFrame:
    
    """Read back the most recent n run log entries as a DataFrame."""
    #This again makes sure that the runs table exists so when it reads a empty DB it'll return an empty table instead of crashing
    init_runlog(path)

    #This query will order the most recent n rows from newest to oldest
    with sqlite3.connect(path) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM runs ORDER BY rowid DESC LIMIT ?",
            conn,
            params=(n,),
        )

    #Convert SQLite's 1 and 0 integers into Python's true/false booleans
    if not df.empty:
        df["success"] = df["success"].astype(bool)
    return df
