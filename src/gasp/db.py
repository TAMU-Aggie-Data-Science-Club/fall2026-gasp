from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "gasp.db"


def init_db(path: str | Path = DB_PATH) -> None:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                series TEXT NOT NULL,
                period TEXT NOT NULL,
                value REAL NOT NULL,
                retrieved_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS degree_days (
                week_end TEXT NOT NULL,
                region TEXT NOT NULL,
                measure TEXT NOT NULL,
                weighting TEXT NOT NULL,
                value REAL NOT NULL,
                retrieved_at TEXT NOT NULL
            )
        """)

        # Degree days have many more rows than storage (several regions, measures,
        # and weightings per week), so index the lookup write_degree_days does per row
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_degree_days_key
            ON degree_days (region, measure, weighting, week_end, retrieved_at)
        """)


def write_series(df: pd.DataFrame, retrieved_at=None, path: str | Path = DB_PATH) -> int:
    init_db(path)
    rows_added = 0

    if retrieved_at is None:
        retrieved_at = pd.Timestamp.now(tz="UTC")

    retrieved_at = pd.Timestamp(retrieved_at)

    if retrieved_at.tzinfo is None:
        retrieved_at = retrieved_at.tz_localize("UTC")
    else:
        retrieved_at = retrieved_at.tz_convert("UTC")

    retrieved_at = retrieved_at.isoformat()

    with sqlite3.connect(path) as conn:
        for row in df.itertuples(index=False):
            series_id = row.series
            period = pd.Timestamp(row.period).isoformat()
            value = float(row.value)

            previous = conn.execute(
                """
                SELECT value
                FROM observations
                WHERE series = ?
                  AND period = ?
                  AND retrieved_at <= ?
                ORDER BY retrieved_at DESC, rowid DESC
                LIMIT 1
                """,
                (series_id, period, retrieved_at)
            ).fetchone()

            if previous is not None and previous[0] == value:
                continue

            conn.execute(
                """
                INSERT INTO observations
                    (series, period, value, retrieved_at)
                VALUES (?, ?, ?, ?)
                """,
                (series_id, period, value, retrieved_at)
            )

            rows_added += 1

    return rows_added


def read_series(series_id: str, as_of=None, path: str | Path = DB_PATH) -> pd.DataFrame:
    init_db(path)

    params = [series_id]
    cutoff_clause = ""

    if as_of is not None:
        cutoff = pd.Timestamp(as_of)

        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")

        cutoff_clause = "AND retrieved_at <= ?"
        params.append(cutoff.isoformat())

    with sqlite3.connect(path) as conn:
        query = f"""
            SELECT period, series, value
            FROM (
                SELECT period, series, value,
                       ROW_NUMBER() OVER (
                           PARTITION BY series, period
                           ORDER BY retrieved_at DESC, rowid DESC
                       ) AS rn
                FROM observations
                WHERE series = ?
                {cutoff_clause}
            )
            WHERE rn = 1
            ORDER BY period
        """

        result = pd.read_sql_query(
            query,
            conn,
            params=params
        )

    result["period"] = pd.to_datetime(result["period"])
    return result[["period", "series", "value"]]


def _to_utc_iso(ts) -> str:
    """Normalize a timestamp to a UTC ISO string, the same way write_series does."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def write_degree_days(df: pd.DataFrame, retrieved_at=None, path: str | Path = DB_PATH) -> int:
    """Store a weekly degree-day frame (CONTRACTS.md, section 2).

    Same rules as write_series: never overwrites, identical data stores
    nothing, and a changed value is stored as a new row with its own
    retrieved_at. Returns the number of new rows stored.
    """
    init_db(path)
    rows_added = 0

    if retrieved_at is None:
        retrieved_at = pd.Timestamp.now(tz="UTC")
    retrieved_at = _to_utc_iso(retrieved_at)

    with sqlite3.connect(path) as conn:
        for row in df.itertuples(index=False):
            week_end = pd.Timestamp(row.week_end).isoformat()
            region = str(row.region)
            measure = str(row.measure)
            weighting = str(row.weighting)
            value = float(row.value)

            # Latest value already stored for this week and combination, as of this write
            previous = conn.execute(
                """
                SELECT value
                FROM degree_days
                WHERE week_end = ?
                  AND region = ?
                  AND measure = ?
                  AND weighting = ?
                  AND retrieved_at <= ?
                ORDER BY retrieved_at DESC, rowid DESC
                LIMIT 1
                """,
                (week_end, region, measure, weighting, retrieved_at),
            ).fetchone()

            # Identical data: store nothing
            if previous is not None and previous[0] == value:
                continue

            conn.execute(
                """
                INSERT INTO degree_days
                    (week_end, region, measure, weighting, value, retrieved_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (week_end, region, measure, weighting, value, retrieved_at),
            )
            rows_added += 1

    return rows_added


def read_degree_days(
    region="US",
    measure=None,
    weighting=None,
    as_of=None,
    path: str | Path = DB_PATH,
) -> pd.DataFrame:
    """Read a weekly degree-day frame (CONTRACTS.md, section 2).

    Filters to whichever of region, measure, and weighting are given
    (None means all). Same as_of logic as read_series: returns the latest
    value for each week and combination stored at or before as_of.
    """
    init_db(path)

    filters = []
    params = []
    for column, wanted in (("region", region), ("measure", measure), ("weighting", weighting)):
        if wanted is not None:
            filters.append(f"AND {column} = ?")
            params.append(str(wanted))

    if as_of is not None:
        filters.append("AND retrieved_at <= ?")
        params.append(_to_utc_iso(as_of))

    with sqlite3.connect(path) as conn:
        query = f"""
            SELECT week_end, region, measure, weighting, value
            FROM (
                SELECT week_end, region, measure, weighting, value,
                       ROW_NUMBER() OVER (
                           PARTITION BY week_end, region, measure, weighting
                           ORDER BY retrieved_at DESC, rowid DESC
                       ) AS rn
                FROM degree_days
                WHERE 1 = 1
                {" ".join(filters)}
            )
            WHERE rn = 1
            ORDER BY week_end, region, measure, weighting
        """
        result = pd.read_sql_query(query, conn, params=params)

    # Match the contract types exactly
    result["week_end"] = pd.to_datetime(result["week_end"])
    for col in ("region", "measure", "weighting"):
        result[col] = result[col].astype(str)
    result["value"] = result["value"].astype(float)
    return result[["week_end", "region", "measure", "weighting", "value"]]