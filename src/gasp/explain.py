

"""Explain LightGBM predictions using feature contributions."""

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
    """Calculate average feature contributions by calendar month."""

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
                "abs_contribution": abs(row["contribution"]),
            })

    if not records:
        raise ValueError(
            "No target weeks found in the requested range"
        )

    result = pd.DataFrame(records)

    # Find the average absolute contribution
    # for each feature in each month.
    result = (
        result.groupby(
            ["month", "feature", "label"],
            as_index=False,
        )["abs_contribution"]
        .mean()
        .rename(columns={
            "abs_contribution": "mean_abs_contribution"
        })
        .sort_values(
            ["month", "mean_abs_contribution"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )

    return result
