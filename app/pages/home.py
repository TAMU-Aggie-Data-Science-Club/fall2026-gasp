from __future__ import annotations

import sqlite3
from pathlib import Path
import pandas as pd
import streamlit as st

from gasp import db, evaluate
from gasp.pipeline import LOWER_48_SERIES
from app.components import (
    format_bcf_change,
    format_model_name,
    render_empty_state,
    render_last_updated,
    term_tip,
)


def fetch_live_forecasts(path: str | Path = db.DB_PATH) -> pd.DataFrame:
    """
    Attempt to load the live forecasts table in two safe ways:
    1. First, try calling Team C's official live.load_live_forecasts() function if they have merged it.
    2. Second, if Team C hasn't merged live.py yet, try reading the 'live_forecasts' table directly
       from SQLite using pandas. This lets us test the Home page using a fake table right now!
    3. If neither exists yet, return an empty DataFrame so the page shows a friendly empty message
       instead of crashing with an error.
    """
    try:
        from gasp import live  # type: ignore
        if hasattr(live, "load_live_forecasts"):
            return live.load_live_forecasts(path=path)
    except Exception:
        # Ignore the error if gasp.live doesn't exist yet and move to the fallback below
        pass

    try:
        with sqlite3.connect(path) as conn:
            return pd.read_sql_query("SELECT * FROM live_forecasts", conn)
    except Exception:
        # If the 'live_forecasts' table hasn't been created in the database yet, it will return an empty table
        return pd.DataFrame()


def format_headline_forecast(
    predicted: float,
    target_period: pd.Timestamp,
    low: float | None = None,
    high: float | None = None,
) -> str:
    """
    Build the exact plain-English headline sentence required by Piece 1:
    'We expect a withdrawal of 85 Bcf for the week ending Nov 6 (likely 62 to 110)'
    """
    # A negative number means gas was pulled out of storage which would be a withdrawal while positive means an injection
    direction = "withdrawal" if predicted < 0 else "injection"
    # Take the absolute value and rounds it to a whole number so '-85.2' becomes '85'
    amt = round(abs(float(predicted)))
    # Format the date as 'Nov 6' (short month name + day number)
    week_str = f"{target_period.strftime('%b')} {target_period.day}"

    # Build the optional (likely 62 to 110) 90% confidence range text if low and high bounds exist
    range_part = ""
    if low is not None and high is not None and pd.notna(low) and pd.notna(high):
        # When both low and high are negative
        # we flip their absolute values so the sentence reads naturally from smaller to larger
        if low <= 0 and high <= 0:
            lo_mag, hi_mag = round(abs(float(high))), round(abs(float(low)))
            range_part = f" (likely {lo_mag} to {hi_mag})"
        # When both low and high are positive (an injection range), keep them in normal order
        elif low >= 0 and high >= 0:
            lo_mag, hi_mag = round(float(low)), round(float(high))
            range_part = f" (likely {lo_mag} to {hi_mag})"
        # If the range crosses zero, it'll show explicit +/- signs
        else:
            range_part = f" (likely {round(float(low)):+d} to {round(float(high)):+d} Bcf)"

    return f"We expect a {direction} of {amt} Bcf for the week ending {week_str}{range_part}"


def render_home(path: str | Path = db.DB_PATH) -> None:
    # Main title and one-line subtitle with hover tooltips on key gas terms
    st.title("U.S. Natural Gas Storage Forecast")
    st.markdown(
        f"Weekly forecasts of {term_tip('Lower 48')} {term_tip('working gas in storage')} changes ahead of the Thursday "
        f"{term_tip('EIA weekly storage report')}.",
        unsafe_allow_html=True,
    )

    # Show a loading spinner while we read the database tables
    with st.spinner("Checking latest forecasts and storage data..."):
        live_df = fetch_live_forecasts(path=path)
        results_df = evaluate.load_results(path=path)
        storage_df = db.read_series(LOWER_48_SERIES, path=path)
        weather_df = db.read_degree_days(path=path)

    # Find the newest storage week and weather week dates to display in the top freshness banner
    storage_week = (
        pd.to_datetime(storage_df["period"]).max().strftime("%Y-%m-%d")
        if not storage_df.empty
        else None
    )
    weather_week = (
        pd.to_datetime(weather_df["week_end"]).max().strftime("%Y-%m-%d")
        if not weather_df.empty
        else None
    )
    render_last_updated(storage_week, weather_week)

    # REQUIREMENT CHECK: If the live_forecasts table is empty (or doesn't exist yet),
    # show the friendly empty-state message and stop here without throwing any errors.
    if live_df.empty:
        render_empty_state("The first live forecast runs Wednesday 11/11.")
        return

    # Copy the dataframe and support either 'period' or 'target_period' as the date column name
    live_df = live_df.copy()
    period_col = "period" if "period" in live_df.columns else "target_period"
    live_df[period_col] = pd.to_datetime(live_df[period_col])

    # Filter down to only the rows for the most recent target forecast week
    target_week = live_df[period_col].max()
    week_forecasts = live_df[live_df[period_col] == target_week].copy()

    # Find which live model has the best (lowest) MAE on the saved backtest scoreboard
    best_model_row = None
    if not results_df.empty and "model" in results_df.columns and "mae" in results_df.columns:
        # Get the list of model names sorted from lowest MAE to highest MAE
        ranked_models = results_df.sort_values("mae", ascending=True)["model"].tolist()
        # Loop through the ranked models and pick the first one that has a live forecast available
        for m in ranked_models:
            match = week_forecasts[week_forecasts["model"] == m]
            if not match.empty:
                best_model_row = match.iloc[0]
                break

    #if the scoreboard table wasn't loaded, just pick the first model row in week_forecasts
    if best_model_row is None:
        best_model_row = week_forecasts.iloc[0]

    # Extract the winning model's predicted Bcf change and its 90% interval bounds, if it has them
    pred_val = float(best_model_row["predicted"])
    low_val = float(best_model_row["low"]) if "low" in best_model_row and pd.notna(best_model_row["low"]) else None
    high_val = float(best_model_row["high"]) if "high" in best_model_row and pd.notna(best_model_row["high"]) else None

    # Calculate the official EIA release date: EIA storage weeks end on a Friday, and the official report
    # is published 6 days later on the following Thursday.
    report_date = target_week + pd.Timedelta(days=6)
    report_str = f"{report_date.strftime('%A, %b')} {report_date.day}, {report_date.year}"
    headline_text = format_headline_forecast(pred_val, target_week, low_val, high_val)

    # Look up that same week's 5-year average forecast so we can compare our lead forecast against normal
    five_yr_row = week_forecasts[week_forecasts["model"] == "five_year_average"]
    if not five_yr_row.empty:
        # Use the five_year_average row from live_forecasts if it's already in the table
        five_yr_val = float(five_yr_row.iloc[0]["predicted"])
    elif not storage_df.empty:
        # Otherwise, compute the 5-year average directly from historical storage data
        from gasp import baselines
        five_yr_val = float(baselines.five_year_average(storage_df[storage_df["period"] < target_week], target_week))
    else:
        five_yr_val = None

    # Build the one-line comparison sentence against the 5-year average
    comparison_line = ""
    if five_yr_val is not None and pd.notna(five_yr_val):
        diff = pred_val - five_yr_val
        comparison_line = (
            f"For comparison, the 5-year average for this week is a **{format_bcf_change(five_yr_val)}** "
            f"({abs(diff):.0f} Bcf {'more' if diff > 0 else 'less'} net addition than our lead forecast)."
        )

    # Draw the main Headline Forecast Card using a single clean border container 
    with st.container(border=True):
        st.caption(
            f"LEAD LIVE FORECAST · {format_model_name(str(best_model_row['model'])).upper()} "
            f"(LOWEST BACKTEST MAE) · EIA REPORT DATE: {report_str.upper()}"
        )
        st.subheader(headline_text)
        st.markdown(
            f"**EIA Release Schedule:** Official data for the week ending **{target_week.strftime('%b %d, %Y')}** "
            f"will be published on **{report_str}**."
        )
        if comparison_line:
            st.markdown(comparison_line)

    #Side-by-Side Comparison of All Live Models
    st.subheader("All Live Model Forecasts Side by Side")
    st.markdown(
        f"How to read this section: Each column shows one model's {term_tip('live forecast')} for the week ending "
        f"**{target_week.strftime('%b %d, %Y')}**, ordered from most accurate to least accurate by historical {term_tip('MAE')}.",
        unsafe_allow_html=True,
    )

    # Sort all live models by their scoreboard MAE so the most accurate models appear on the left
    if not results_df.empty and "model" in results_df.columns:
        mae_map = dict(zip(results_df["model"], results_df["mae"]))
        week_forecasts["sort_mae"] = week_forecasts["model"].map(mae_map).fillna(999.0)
        week_forecasts = week_forecasts.sort_values("sort_mae").reset_index(drop=True)

    # Create up to 4 side-by-side columns and display each model's forecast and 90% range
    cols = st.columns(min(len(week_forecasts), 4))
    for idx, (_, row) in enumerate(week_forecasts.iterrows()):
        col = cols[idx % len(cols)]
        with col:
            m_name = format_model_name(str(row["model"]))
            m_pred = format_bcf_change(float(row["predicted"]))
            if "low" in row and "high" in row and pd.notna(row["low"]) and pd.notna(row["high"]):
                sub_range = f"90% range: {float(row['low']):.0f} to {float(row['high']):.0f} Bcf"
            else:
                sub_range = "Point forecast only"
            #keeps the range text neutral gray instead of bright red/green arrows
            st.metric(label=m_name, value=m_pred, delta=sub_range, delta_color="off")


# Run the Home page function when Streamlit loads this file
render_home()

