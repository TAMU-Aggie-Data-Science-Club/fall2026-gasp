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