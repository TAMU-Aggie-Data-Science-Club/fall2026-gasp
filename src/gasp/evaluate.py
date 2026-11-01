from contextlib import closing
from pathlib import Path
import sqlite3

import pandas as pd
from gasp import backtest, db
import matplotlib.pyplot as plt


_RESULT_COLUMNS = [
    "model", "mae", "rmse", "bias", "n_weeks", "start", "end",
    "coverage", "avg_width",
]
_PREDICTION_COLUMNS = ["model", "period", "actual", "predicted", "low", "high"]


# Run each model once for the chosen weeks and build both summary scores and
# weekly prediction rows, which compare and save_scoreboard can reuse.
def _run_backtests(df, forecasters, start, end, intervals=None):
    """Build summary scores and weekly predictions from one run per model."""
    intervals = {} if intervals is None else intervals
    results = []
    predictions = []

    # Model names connect each point forecaster to its optional interval function.
    for name, forecaster in forecasters.items():
        interval = intervals.get(name)
        test = backtest.walk_forward(
            df, forecaster, start, end, interval=interval
        )

        results.append({
            "model": name,
            "mae": backtest.mae(test["actual"], test["predicted"]),
            "rmse": backtest.rmse(test["actual"], test["predicted"]),
            "bias": float((test["predicted"] - test["actual"]).mean()),
            "n_weeks": len(test),
            "start": test["period"].min(),
            "end": test["period"].max(),
            "coverage": (
                backtest.coverage(test["actual"], test["low"], test["high"])
                if interval is not None else float("nan")
            ),
            "avg_width": (
                backtest.avg_width(test["low"], test["high"])
                if interval is not None else float("nan")
            ),
        })

        # Keep the same weekly rows for saving; missing bounds become NaN.
        weekly = test.reindex(columns=_PREDICTION_COLUMNS[1:]).copy()
        weekly.insert(0, "model", name)
        predictions.append(weekly)

    scores = pd.DataFrame(results, columns=_RESULT_COLUMNS)
    scores = scores.sort_values("mae").reset_index(drop=True)
    weekly_predictions = (
        pd.concat(predictions, ignore_index=True)
        if predictions else pd.DataFrame(columns=_PREDICTION_COLUMNS)
    )
    weekly_predictions = weekly_predictions.sort_values(
        ["model", "period"]
    ).reset_index(drop=True)
    return scores, weekly_predictions


# Return one score row per model, ordered by MAE, with optional interval scores.
# Models without intervals have NaN in the coverage and average-width columns.
def compare(df, forecasters, start, end, intervals=None) -> pd.DataFrame:
    """Compare model accuracy, with interval scores when supplied."""
    scores, _ = _run_backtests(df, forecasters, start, end, intervals)
    return scores


# Refresh the database's summary and weekly-prediction tables from one backtest
# per model, replacing the previous scoreboard each time this function runs.
def save_scoreboard(
    df, forecasters, start, end, intervals=None, path=db.DB_PATH,
) -> None:
    """Run each model once and replace the saved results and predictions."""
    # Both tables come from the same backtests, so models are not run twice.
    scores, predictions = _run_backtests(df, forecasters, start, end, intervals)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(path)) as connection:
        scores.to_sql("results", connection, if_exists="replace", index=False)
        predictions.to_sql(
            "predictions", connection, if_exists="replace", index=False
        )


# Read the saved model summaries without running models, restore their dates,
# and return the scores with the lowest MAE first.
def load_results(path=db.DB_PATH) -> pd.DataFrame:
    """Load saved model scores, sorted by MAE with the best model first."""
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        results = pd.read_sql_query(
            f"SELECT {', '.join(_RESULT_COLUMNS)} FROM results "
            "ORDER BY mae IS NULL, mae",
            connection,
            parse_dates=["start", "end"],
        )

    # SQL NULLs become floating NaNs, including when every model lacks intervals.
    return results.astype({
        "mae": float, "rmse": float, "bias": float, "n_weeks": "int64",
        "coverage": float, "avg_width": float,
    })


# Read saved actual changes, forecasts, and bounds for all models or one model,
# ordered by model and date, ready for charts or further analysis.
def load_predictions(model=None, path=db.DB_PATH) -> pd.DataFrame:
    """Load weekly predictions for all models, or one named model."""
    query = f"SELECT {', '.join(_PREDICTION_COLUMNS)} FROM predictions"
    params = []
    if model is not None:
        query += " WHERE model = ?"
        params.append(model)
    query += " ORDER BY model, period"

    uri = Path(path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        predictions = pd.read_sql_query(
            query, connection, params=params, parse_dates=["period"]
        )
    return predictions.astype({
        "actual": float, "predicted": float, "low": float, "high": float,
    })


#Compares each model's MAE for every month of the year.
def compare_by_month(df, forecasters, start, end) -> pd.DataFrame:
    results = []

    #Run each forecasting model through the backtest.
    for name, forecaster in forecasters.items():
        test = backtest.walk_forward(df, forecaster, start, end)

        #Get the month for each predicted week.
        test["month"] = pd.to_datetime(test["period"]).dt.month

        #Calculate MAE for each month.
        for month, group in test.groupby("month"):
            monthly_mae = backtest.mae(
                group["actual"], group["predicted"]
            )

            #Save the monthly result for this model.
            results.append({
                "model": name,
                "month": int(month),
                "mae": monthly_mae
            })

    #Return the results sorted by month and model.
    return pd.DataFrame(
        results,
        columns=["model", "month", "mae"]
    ).sort_values(["month", "model"]).reset_index(drop=True)


#Plots actual vs predicted storage changes for selected models.
def plot_comparison(df, forecasters, start, end):
    plt.figure(figsize=(14, 5))

    #Run each model and get its predictions.
    for name, forecaster in forecasters.items():
        test = backtest.walk_forward(df, forecaster, start, end)

        #Plot actual values only once.
        if name == next(iter(forecasters)):
            plt.plot(
                test["period"], test["actual"],
                label="Actual", color="black"
            )

        #Plot predicted values for each model.
        plt.plot(
            test["period"], test["predicted"],
            label=name, alpha=0.8
        )

    #Add labels and formatting.
    plt.title("Actual vs Predicted Weekly Storage Change")
    plt.xlabel("Date")
    plt.ylabel("Weekly Storage Change (Bcf)")
    plt.legend()
    plt.grid(True)

    plt.show()
