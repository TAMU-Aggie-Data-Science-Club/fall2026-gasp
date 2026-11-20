

"""Explain LightGBM predictions using feature contributions."""

import sqlite3
from gasp import db

import pandas as pd


from gasp.feature_table import FEATURE_COLUMNS, FEATURE_LABELS
from gasp.lightgbm_model import train_lightgbm


def explain(
    feature_table,
    target,
    feature_columns=FEATURE_COLUMNS,
    params=None,
) -> pd.DataFrame:
    """Explain a forecast for one target week."""

    target = pd.Timestamp(target)
    columns = list(feature_columns)

    table = feature_table.copy()
    table["period"] = pd.to_datetime(table["period"])

    # Find the target week's row.
    target_row = table.loc[table["period"] == target]

    if len(target_row) != 1:
        raise ValueError(
            f"Expected one row for {target.date()}, "
            f"found {len(target_row)}"
        )

    # Train using only weeks before the target.
    model = train_lightgbm(
        table,
        before=target,
        feature_columns=columns,
        params=params,
    )

    # Get each feature's contribution to the prediction.
    X = target_row[columns].astype(float)
    contributions = model.predict(
        X, pred_contrib=True
    )[0]

    base_value = float(contributions[-1])

    # Create the explanation table required by CONTRACTS.md.
    result = pd.DataFrame({
        "target": target,
        "feature": columns,
        "label": [
            FEATURE_LABELS.get(col, col)
            for col in columns
        ],
        "value": X.iloc[0].to_numpy(),
        "contribution": contributions[:-1],
        "base_value": base_value,
    })

    # Show the largest contributions first.
    return result.sort_values(
        "contribution",
        key=lambda values: values.abs(),
        ascending=False,
    ).reset_index(drop=True)



def explain_history(
    feature_table,
    start,
    end,
    feature_columns=FEATURE_COLUMNS,
    params=None,
) -> pd.DataFrame:
    """Calculate average feature contributions by month."""

    table = feature_table.copy()
    table["period"] = pd.to_datetime(table["period"])

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    targets = table.loc[
        table["period"].between(start, end),
        "period",
    ].sort_values()

    records = []

    for target in targets:
        explanation = explain(
            table,
            target,
            feature_columns=feature_columns,
            params=params,
        )

        for _, row in explanation.iterrows():
            records.append({
                "month": target.month,
                "feature": row["feature"],
                "label": row["label"],
                "contribution": row["contribution"],
                "abs_contribution": abs(row["contribution"]),
            })

    if not records:
        raise ValueError("No target weeks found")

    result = pd.DataFrame(records)

    return (
        result.groupby(
            ["month", "feature", "label"],
            as_index=False,
        )
        .agg(
            mean_abs_contribution=(
                "abs_contribution", "mean"
            ),
            mean_contribution=(
                "contribution", "mean"
            ),
        )
        .sort_values(
            ["month", "mean_abs_contribution"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )


HISTORY_COLUMNS = [
    "month",
    "feature",
    "label",
    "mean_abs_contribution",
    "mean_contribution",
    "start",
    "end",
]


def save_explanation_history(
    history,
    start,
    end,
    path=db.DB_PATH,
):
    """Replace the saved explanation history table."""

    result = history.copy()

    required = HISTORY_COLUMNS[:-2]
    missing = [c for c in required if c not in result.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    result["start"] = pd.Timestamp(start)
    result["end"] = pd.Timestamp(end)

    result = (
        result[HISTORY_COLUMNS]
        .sort_values(
            ["month", "mean_abs_contribution"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )

    with sqlite3.connect(path) as conn:
        result.to_sql(
            "explanation_history",
            conn,
            if_exists="replace",
            index=False,
        )


def load_explanation_history(path=db.DB_PATH) -> pd.DataFrame:
    """Load explanation history, or return an empty table."""

    empty = pd.DataFrame({
        "month": pd.Series(dtype="int64"),
        "feature": pd.Series(dtype="str"),
        "label": pd.Series(dtype="str"),
        "mean_abs_contribution": pd.Series(dtype="float64"),
        "mean_contribution": pd.Series(dtype="float64"),
        "start": pd.Series(dtype="datetime64[ns]"),
        "end": pd.Series(dtype="datetime64[ns]"),
    })

    # Do not create a database just to read from it.
    from pathlib import Path

    if not Path(path).is_file():
        return empty

    with sqlite3.connect(path) as conn:
        exists = conn.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table'
              AND name = 'explanation_history'
            """
        ).fetchone()

        if exists is None:
            return empty

        result = pd.read_sql_query(
            "SELECT * FROM explanation_history",
            conn,
            parse_dates=["start", "end"],
        )

    return (
        result[HISTORY_COLUMNS]
        .sort_values(
            ["month", "mean_abs_contribution"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )

