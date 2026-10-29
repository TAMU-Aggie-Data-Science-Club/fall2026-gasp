
import pandas as pd
from gasp import backtest
import matplotlib.pyplot as plt


#Compares forecasting models using MAE, RMSE, and bias.
def compare(df, forecasters, start, end) -> pd.DataFrame:
    results = []

    #Run the backtest for each forecasting model.
    for name, forecaster in forecasters.items():
        test = backtest.walk_forward(df, forecaster, start, end)

        #Calculate the model's accuracy.
        model_mae = backtest.mae(test["actual"], test["predicted"])
        model_rmse = backtest.rmse(test["actual"], test["predicted"])

        #Positive bias means predictions are too high.
        model_bias = float(
            (test["predicted"] - test["actual"]).mean()
        )

        #Save the results for this model.
        results.append({
            "model": name,
            "mae": model_mae,
            "rmse": model_rmse,
            "bias": model_bias,
            "n_weeks": len(test),
            "start": test["period"].min(),
            "end": test["period"].max()
        })

    #Create the results table, best MAE first.
    return pd.DataFrame(
        results,
        columns=["model", "mae", "rmse", "bias","n_weeks", "start", "end"]
    ).sort_values("mae").reset_index(drop=True)


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
