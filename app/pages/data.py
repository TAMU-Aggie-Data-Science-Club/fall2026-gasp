# Allow modern type hints (like str | None) to run on Python 3.9+
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gasp import baselines, db, features, runlog
from gasp.pipeline import LOWER_48_SERIES
from app.components import (
    apply_chart_bounds,
    get_model_color,
    render_empty_state,
    render_last_updated,
    term_tip,
)


# Cache all three database reads for 10 minutes (600s) so switching timeframes or pages is instant
@st.cache_data(ttl=600)
def load_storage_data() -> pd.DataFrame:
    # Read the historical Lower 48 working gas storage series from data/gasp.db
    return db.read_series(LOWER_48_SERIES)


@st.cache_data(ttl=600)
def load_weather_data() -> pd.DataFrame:
    # Read regional weekly Heating and Cooling Degree Days from data/gasp.db
    return db.read_degree_days()


@st.cache_data(ttl=600)
def load_run_logs() -> pd.DataFrame:
    # Read the most recent pipeline execution logs from the runs table
    return runlog.last_runs()


st.title("Storage & Weather Data Explorer")

# Show a loading spinner while the cached data frames are fetched
with st.spinner("Loading storage history and pipeline logs..."):
    storage_df = load_storage_data()
    weather_df = load_weather_data()
    runs_df = load_run_logs()

# Show a friendly message if the database hasn't been populated yet
if storage_df.empty:
    render_empty_state("No storage data found in the database. Run `python -m gasp.pipeline` first.")
else:
    # Sort storage data chronologically and compute the 5-year mean, min, and max columns for each week
    storage_df = storage_df.sort_values("period").reset_index(drop=True)
    stats_df = features.add_five_year_stats(storage_df)

    # Find the most recent storage week and most recent weather week for our freshness banner
    latest_row = stats_df.iloc[-1]
    latest_date_str = pd.to_datetime(latest_row["period"]).strftime("%Y-%m-%d")
    weather_date_str = (
        pd.to_datetime(weather_df["week_end"]).max().strftime("%Y-%m-%d")
        if not weather_df.empty
        else None
    )

    # Display the subtle "last updated" status line at the top of the page
    render_last_updated(latest_date_str, weather_date_str)

    # Compute the 5-second plain-English takeaway comparing current storage to the 5-year average
    latest_val = float(latest_row["value"])
    latest_mean = float(latest_row["five_year_mean"]) if pd.notna(latest_row.get("five_year_mean")) else None
    if latest_mean is not None:
        diff = latest_val - latest_mean
        status_word = "surplus" if diff >= 0 else "deficit"
        st.markdown(
            f"**Key Takeaway:** As of **{latest_date_str}**, {term_tip('Lower 48')} {term_tip('working gas in storage')} stands at "
            f"**{latest_val:,.0f} {term_tip('Bcf')}**, which is a **{abs(diff):,.0f} {term_tip('Bcf')} {term_tip('surplus/deficit', status_word)}** "
            f"compared to the {term_tip('five-year average and five-year range', 'five-year average')} ({latest_mean:,.0f} Bcf).",
            unsafe_allow_html=True,
        )

    #Section 1: Storage vs. Five-Year Range
    st.subheader("1. Lower 48 Storage vs. Five-Year Historical Range")
    st.markdown(
        f"How to read this chart: The solid navy line tracks actual {term_tip('working gas in storage')} ({term_tip('Bcf')}), "
        f"the dashed orange line shows the {term_tip('five-year average and five-year range', 'five-year average')}, and the shaded gray band shows the 5-year min-to-max range.",
        unsafe_allow_html=True,
    )

    # Radio button letting the user toggle between the default Last 2 Years view and Full History
    timeframe = st.radio(
        "Timeframe:",
        options=("Last 2 Years", "Full History"),
        horizontal=True,
        key="data_tf_radio",
    )

    # Filter the dataframe based on the user's radio button choice
    if timeframe == "Last 2 Years":
        cutoff = stats_df["period"].max() - pd.DateOffset(years=2)
        plot_df = stats_df[stats_df["period"] >= cutoff].copy()
    else:
        plot_df = stats_df.copy()

    # Initialize the Plotly figure for Storage vs. 5-Year Range
    fig_storage = go.Figure()

    #Shaded 5-year min-to-max band.(Layer 1)
    # Just like on the Models page, we draw an invisible line at 'five_year_max' first,
    # then draw 'five_year_min' with fill='tonexty' to shade the area between min and max.
    band_df = plot_df.dropna(subset=["five_year_min", "five_year_max"])
    if not band_df.empty:
        fig_storage.add_trace(
            go.Scatter(
                x=band_df["period"],
                y=band_df["five_year_max"],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig_storage.add_trace(
            go.Scatter(
                x=band_df["period"],
                y=band_df["five_year_min"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(128, 139, 150, 0.25)",
                name="5-Year Min–Max Range",
                hovertemplate="5-Yr Range: %{y:,.0f} to %{customdata:,.0f} Bcf<extra></extra>",
                customdata=band_df["five_year_max"],
            )
        )

    #Dashed burnt-orange line showing the 5-year historical mean(Layer 2)
    mean_df = plot_df.dropna(subset=["five_year_mean"])
    if not mean_df.empty:
        fig_storage.add_trace(
            go.Scatter(
                x=mean_df["period"],
                y=mean_df["five_year_mean"],
                mode="lines",
                name="5-Year Average",
                line={"color": "#D95F02", "width": 2, "dash": "dash"},
                hovertemplate="5-Yr Average: %{y:,.0f} Bcf<extra></extra>",
            )
        )

    #Solid navy line showing actual working gas in storage(Layer 3)
    fig_storage.add_trace(
        go.Scatter(
            x=plot_df["period"],
            y=plot_df["value"],
            mode="lines",
            name="Working Gas in Storage",
            line={"color": "#1B4F72", "width": 2.6},
            hovertemplate="Actual Storage: %{y:,.0f} Bcf<extra></extra>",
        )
    )

    # Gather all Y values so apply_chart_bounds can lock the zoom-out limit to the exact data fit
    y_vals = pd.concat(
        [plot_df["value"], plot_df["five_year_min"].dropna(), plot_df["five_year_max"].dropna()],
        ignore_index=True,
    )
    apply_chart_bounds(
        fig_storage,
        x_min=plot_df["period"].min(),
        x_max=plot_df["period"].max(),
        x_title="Week",
        y_title="Working Gas in Storage (Bcf)",
        y_min=float(y_vals.min()),
        y_max=float(y_vals.max()),
    )
    fig_storage.update_layout(height=430)
    st.plotly_chart(fig_storage, use_container_width=True)

    #Section 2: Recent Weekly Change
    st.subheader("2. Weekly Storage Change vs. Five-Year Average")
    st.markdown(
        f"How to read this chart: Positive values represent net storage {term_tip('injection', 'injections')} (+{term_tip('Bcf')}) "
        f"and negative values represent net {term_tip('withdrawal', 'withdrawals')} (-{term_tip('Bcf')}) compared against the 5-year average change for the same week.",
        unsafe_allow_html=True,
    )

    # Calculate week-over-week storage change using .diff() and filter to the last 2 years
    full_change_df = storage_df.copy()
    full_change_df["weekly_change"] = full_change_df["value"].diff()
    weekly_cutoff = storage_df["period"].max() - pd.DateOffset(years=2)
    recent_changes = full_change_df[full_change_df["period"] >= weekly_cutoff].copy()

    # For each week in the last 2 years, compute the 5-year average change using only history prior to that week
    recent_changes["avg_weekly_change"] = [
        baselines.five_year_average(storage_df[storage_df["period"] < p], p)
        for p in recent_changes["period"]
    ]

    # Plot actual weekly change (solid black) against the 5-year average weekly change 
    fig_change = go.Figure()
    fig_change.add_trace(
        go.Scatter(
            x=recent_changes["period"],
            y=recent_changes["weekly_change"],
            mode="lines",
            name="Actual Weekly Change",
            line={"color": get_model_color("actual"), "width": 2.3},
            hovertemplate="Actual Change: %{y:.1f} Bcf<extra></extra>",
        )
    )
    fig_change.add_trace(
        go.Scatter(
            x=recent_changes["period"],
            y=recent_changes["avg_weekly_change"],
            mode="lines",
            name="5-Year Average Change",
            line={"color": get_model_color("five_year_average"), "width": 2, "dash": "dash"},
            hovertemplate="5-Yr Avg Change: %{y:.1f} Bcf<extra></extra>",
        )
    )

    # Lock chart bounds so the user can zoom in and double-click to reset, but never zoom out past the data
    chg_vals = pd.concat(
        [recent_changes["weekly_change"].dropna(), recent_changes["avg_weekly_change"].dropna()],
        ignore_index=True,
    )
    apply_chart_bounds(
        fig_change,
        x_min=recent_changes["period"].min(),
        x_max=recent_changes["period"].max(),
        x_title="Week",
        y_title="Weekly Storage Change (Bcf)",
        y_min=float(chg_vals.min()),
        y_max=float(chg_vals.max()),
    )
    fig_change.update_layout(height=390)
    st.plotly_chart(fig_change, use_container_width=True)

    #Section 3: Data Freshness & Pipeline Run Log
    st.subheader("3. Data Freshness & Recent Pipeline Runs")
    # Display the newest storage and weather weeks side-by-side in two equal columns
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Newest Stored Storage Week", latest_date_str)
    with col2:
        st.metric("Newest Stored Weather Week", weather_date_str or "No weather data")

    st.markdown("How to read this table: Each row records an automated data fetch run, how many new rows were stored, and whether validation succeeded.")
    if runs_df.empty:
        st.info("No pipeline runs logged yet.")
    else:
        st.dataframe(runs_df, use_container_width=True, hide_index=True)
