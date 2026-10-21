import numpy as np
import pandas as pd

#MAE (Mean Absolute Error) measures how far your predictions are from the actual values, on average.
def mae(actual, predicted) -> float:
    return float(np.mean(np.abs(np.array(actual) - np.array(predicted))))

#RMSE (Root Mean Squared Error) measures how inaccurate your predictions are, but gives larger mistakes more weight.
def rmse(actual, predicted) -> float:
    return float(np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2)))

#Bias tells us whether our forecasting model tends to predict too high or too low. KEEPS negative errors
def bias(actual, predicted) -> float:
    return float(np.mean(np.array(predicted) - np.array(actual)))

def walk_forward(df, forecaster, start, end) -> pd.DataFrame:
    start = pd.Timestamp(start) #Converts start and end into Pandas dates so Python can compare them with the dates in your data.
    end = pd.Timestamp(end)

    df = df.sort_values("period").reset_index(drop=True) #Ensures the oldest storage measurements come first.

    results = []

    #Each iteration selects a week that we might want to predict.
    for i in range(len(df)):
        target = df.loc[i, "period"]

        if target < start or target > end:
            continue

        history = df[df["period"] < target].copy() #Gives the forecaster only the weeks before the target week.

        if i == 0:
            raise ValueError("Not enough history to calculate weekly change")

        predicted = float(forecaster(history, target))

        actual = float(df.loc[i, "value"] - df.loc[i - 1, "value"]) #Weekly change is this week's storage minus last week's storage.

        #It records the target date, actual change, predicted change, and error.
        results.append({
            "period": target,
            "actual": actual,
            "predicted": predicted,
            "error": predicted - actual
        })

    return pd.DataFrame(
        results,
        columns=["period", "actual", "predicted", "error"]
    )