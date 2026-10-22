
import pandas as pd

#Predicts the next weekly change using the most recent weekly change.
def last_change(history, target) -> float:

    #Sort data and only use weeks before the target.
    target = pd.Timestamp(target)
    history = history[history["period"] < target].sort_values("period")

    #Need at least two weeks to calculate a change.
    if len(history) < 2:
        raise ValueError("Not enough history to calculate last change")

    #Get the last two storage values.
    last_value = history.iloc[-1]["value"]
    previous_value = history.iloc[-2]["value"]

    #Return the difference between the last two weeks.
    return float(last_value - previous_value)


#Predicts the change using the same ISO week from last year.
def same_week_last_year(history, target) -> float:
    target = pd.Timestamp(target)
    history = history.sort_values("period").reset_index(drop=True).copy()

    #Only use data from before the target week.
    history = history[history["period"] < target].copy()

    #Get the ISO year and week for each row.
    history["iso_year"] = history["period"].dt.isocalendar().year
    history["iso_week"] = history["period"].dt.isocalendar().week

    #Get the year and week we are predicting.
    target_year = target.isocalendar().year
    target_week = target.isocalendar().week

    #Use week 52 when week 53 is unavailable in the previous year.
    if target_week == 53:
        target_week = 52

    #Find the same week from the previous year.
    matching = history[
        (history["iso_year"] == target_year - 1) &
        (history["iso_week"] == target_week)
    ]

    #Raise an error if that week is missing.
    if matching.empty:
        raise ValueError("Not enough history for same week last year")

    #Find where the matching week is in the data.
    index = matching.index[-1]
    position = history.index.get_loc(index)

    #Make sure the matching week has a previous week.
    if position == 0:
        raise ValueError("Not enough history to calculate weekly change")

    #Get the matching week's storage and the week before it.
    last_value = history.iloc[position]["value"]
    previous_value = history.iloc[position - 1]["value"]

    #Return the weekly change from last year.
    return float(last_value - previous_value)



#Predicts the weekly change using the same week from the past 5 years.
def five_year_average(history, target) -> float:
    target = pd.Timestamp(target)
    history = history.sort_values("period").reset_index(drop=True).copy()

    #Only use data from before the target week.
    history = history[history["period"] < target].copy()

    #Get the ISO year and week for each row.
    history["iso_year"] = history["period"].dt.isocalendar().year
    history["iso_week"] = history["period"].dt.isocalendar().week

    target_year = target.isocalendar().year
    target_week = target.isocalendar().week

    #Use week 52 if the target is week 53.
    if target_week == 53:
        target_week = 52

    changes = []

    #Look at the same week for each of the past 5 years.
    for year in range(target_year - 5, target_year):
        matching = history[
            (history["iso_year"] == year) &
            (history["iso_week"] == target_week)
        ]

        #Raise an error if a year is missing.
        if matching.empty:
            raise ValueError("Not enough history for five year average")

        #Find the matching week and the week before it.
        index = matching.index[-1]
        position = history.index.get_loc(index)

        if position == 0:
            raise ValueError("Not enough history to calculate weekly change")

        last_value = history.iloc[position]["value"]
        previous_value = history.iloc[position - 1]["value"]

        #Save the weekly change for this year.
        changes.append(last_value - previous_value)

    #Return the average change across all 5 years.
    return float(sum(changes) / len(changes))
