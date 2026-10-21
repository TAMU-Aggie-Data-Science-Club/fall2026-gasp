import sqlite3
from pathlib import Path

import pandas as pd


def init_db(path: str = "data/gasp.db") -> None:
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


def write_series(df: pd.DataFrame, retrieved_at=None) -> int:
    init_db()
    rows_added = 0

    if retrieved_at is None:
        retrieved_at = pd.Timestamp.now(tz="UTC")

    retrieved_at = pd.Timestamp(retrieved_at)

    if retrieved_at.tzinfo is None:
        retrieved_at = retrieved_at.tz_localize("UTC")
    else:
        retrieved_at = retrieved_at.tz_convert("UTC")

    retrieved_at = retrieved_at.isoformat()

    with sqlite3.connect("data/gasp.db") as conn:
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


def read_series(series_id: str, as_of=None) -> pd.DataFrame:
    init_db()

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

    with sqlite3.connect("data/gasp.db") as conn:
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
