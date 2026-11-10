# Allow modern type hints (like str | None) to run on older Python versions (e.g. 3.9)
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gasp import evaluate
from app.components import (
    apply_chart_bounds,
    format_model_name,
    get_model_color,
    help_text,
    render_empty_state,
    term_tip,
)


# Cache the database queries for 10 minutes (600 seconds) so every click on the webpage
# reuses the data in memory instead of reading the SQLite database file from scratch.
# Each returns an empty table if the scoreboard hasn't been saved yet, so the empty states show.
@st.cache_data(ttl=600)
def load_scoreboard_results() -> pd.DataFrame:
    # Load the summary metrics table (MAE, RMSE, bias, coverage) saved by Team C
    try:
        return evaluate.load_results()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_scoreboard_predictions() -> pd.DataFrame:
    # Load the week-by-week historical backtest predictions and 90% interval bounds
    try:
        return evaluate.load_predictions()
    except Exception:
        return pd.DataFrame()


def _hex_to_rgba(hex_color: str, alpha: float = 0.20) -> str:
    """
    Helper function that converts a solid hex color (like '#008080') into a see-through
    RGBA color string (like 'rgba(0, 128, 128, 0.20)'). We need this so the 90% confidence
    band is translucent and doesn't hide the lines underneath it.
    """
    h = hex_color.lstrip("#")
    if len(h) == 6:
        # Convert the 3 pairs of hex digits into Red, Green, and Blue numbers (0-255)
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"
    return f"rgba(27, 79, 114, {alpha})"


# Page title displayed at the top of the Models page
st.title("Model Performance & Backtest Comparisons")

# Show a loading spinner while the scoreboard and prediction tables load from the database
with st.spinner("Loading saved backtest scoreboard and predictions..."):
    results_df = load_scoreboard_results()
    predictions_df = load_scoreboard_predictions()

# If the scoreboard table is empty, show a friendly message instead of crashing
if results_df.empty:
    render_empty_state("No saved scoreboard results found yet. Run `python -m gasp.scoreboard` to generate them.")
else:
    # Sort models from lowest (best) MAE to highest MAE and reset row index numbers to 0, 1, 2...
    sorted_results = results_df.sort_values("mae", ascending=True).reset_index(drop=True)
    # Grab the top row (index 0), which is our most accurate model overall
    best_row = sorted_results.iloc[0]

    # Display the "readable in 5 seconds" plain-English takeaway at the very top of the page.
    # unsafe_allow_html=True allows our term_tip() function to render hover tooltips.
    st.markdown(
        f"**Key Takeaway:** Over the 2021–2025 walk-forward {term_tip('backtest')}, "
        f"**{format_model_name(best_row['model'])}** achieved the most accurate weekly forecasts with an average miss "
        f"({term_tip('MAE')}) of **{best_row['mae']:.1f} {term_tip('Bcf')}**.",
        unsafe_allow_html=True,
    )

    # --- Section 1: Scoreboard Table ---
    st.subheader("1. Model Scoreboard (2021–2025 Backtest)")
    # One-sentence "how to read this" explanation with hover tooltips on technical terms
    st.markdown(
        f"How to read this table: Models are ranked from most accurate (lowest {term_tip('MAE')}) at the top to least accurate. "
        f"{term_tip('MAE')} measures average miss size in {term_tip('Bcf')}, {term_tip('RMSE')} penalizes large single-week misses more heavily, "
        f"and {term_tip('bias')} shows whether a model tends to over- or under-predict.",
        unsafe_allow_html=True,
    )
    # Render the interactive scoreboard table stretched across the page width
    st.dataframe(sorted_results, use_container_width=True, hide_index=True)

# --- Section 2: Actual vs. Predicted Weekly Changes ---
st.subheader("2. Actual vs. Predicted Weekly Storage Changes")
st.markdown(
    f"How to read this chart: The solid black line shows actual weekly {term_tip('injection', 'injections')} (+) and "
    f"{term_tip('withdrawal', 'withdrawals')} (-) in {term_tip('Bcf')}, while dashed colored lines show each model's forecast "
    f"(and shaded bands show {term_tip('forecast range (90%)', '90% forecast ranges')} when available).",
    unsafe_allow_html=True,
)

if predictions_df.empty:
    render_empty_state("No saved model predictions found in the database.")
else:
    # Make a copy of the dataframe so we don't accidentally modify the cached table in memory,
    # and convert the 'period' column from text strings into proper calendar dates.
    pred_df = predictions_df.copy()
    pred_df["period"] = pd.to_datetime(pred_df["period"])

    # Build the list of models for the dropdown menu, ordered from best MAE to worst MAE
    if not results_df.empty and "model" in results_df.columns:
        available_models = [
            m for m in results_df.sort_values("mae")["model"].tolist()
            if m in pred_df["model"].unique()
        ]
    else:
        available_models = sorted(pred_df["model"].unique().tolist())

    # Create the multi-select dropdown box. Default to selecting the top 2 models so the user
    # immediately sees how two different models compare without having to click first.
    selected_models = st.multiselect(
        "Select one or more models to compare:",
        options=available_models,
        default=available_models[:2] if len(available_models) >= 2 else available_models[:1],
        format_func=format_model_name,
        help=help_text("backtest"),
    )

    if not selected_models:
        st.info("Select at least one model above to compare against actual weekly changes.")
    else:
        # Filter the predictions table down to only the rows for the models the user selected
        filtered_preds = pred_df[pred_df["model"].isin(selected_models)].copy()

        # Extract one single timeline of actual storage changes. Because every model row for a given
        # week has the same 'actual' value, drop_duplicates keeps just one row per week.
        actual_df = (
            filtered_preds[["period", "actual"]]
            .drop_duplicates(subset=["period"])
            .sort_values("period")
        )

        # Create an empty Plotly figure canvas where we will stack our bands and lines
        fig_avp = go.Figure()

        # LAYER 1: Shaded 90% confidence bands (drawn first so they sit behind the lines).
        # We check if 'low' and 'high' columns exist and drop any rows where they are NaN
        # (so models without intervals, like SARIMA, simply skip this step without errors).
        if "low" in filtered_preds.columns and "high" in filtered_preds.columns:
            for model_name in selected_models:
                m_band = (
                    filtered_preds[filtered_preds["model"] == model_name]
                    .dropna(subset=["low", "high"])
                    .sort_values("period")
                )
                if not m_band.empty:
                    color_hex = get_model_color(model_name)
                    # Plotly trick for shaded bands: First draw an invisible line (width=0) at the 'high' bound.
                    fig_avp.add_trace(
                        go.Scatter(
                            x=m_band["period"],
                            y=m_band["high"],
                            mode="lines",
                            line={"width": 0},
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )
                    # Next, draw the 'low' bound line and set fill='tonexty', which tells Plotly to shade
                    # the area between this 'low' line and the 'high' line we just drew right above it.
                    fig_avp.add_trace(
                        go.Scatter(
                            x=m_band["period"],
                            y=m_band["low"],
                            mode="lines",
                            line={"width": 0},
                            fill="tonexty",
                            fillcolor=_hex_to_rgba(color_hex, 0.22),
                            name=f"{format_model_name(model_name)} 90% Range",
                            hovertemplate="90% Range: %{y:.1f} to %{customdata:.1f} Bcf<extra></extra>",
                            customdata=m_band["high"],
                        )
                    )

        # LAYER 2: Draw the actual EIA weekly storage changes as a solid charcoal-black line
        fig_avp.add_trace(
            go.Scatter(
                x=actual_df["period"],
                y=actual_df["actual"],
                mode="lines",
                name="Actual Weekly Change",
                line={"color": get_model_color("actual"), "width": 2.4},
                hovertemplate="Actual: %{y:.1f} Bcf<extra></extra>",
            )
        )

        # LAYER 3: Loop through each selected model and draw its predicted weekly changes as a
        # dashed line using that model's permanent color from MODEL_COLORS.
        for model_name in selected_models:
            m_df = filtered_preds[filtered_preds["model"] == model_name].sort_values("period")
            fig_avp.add_trace(
                go.Scatter(
                    x=m_df["period"],
                    y=m_df["predicted"],
                    mode="lines",
                    name=format_model_name(model_name),
                    line={"color": get_model_color(model_name), "width": 2, "dash": "dash"},
                    hovertemplate=f"{format_model_name(model_name)}: %{{y:.1f}} Bcf<extra></extra>",
                )
            )

        # Collect all Y-axis values (actual, predicted, low, high) so we know the exact minimum
        # and maximum numbers shown on the chart.
        y_cols = [filtered_preds["actual"], filtered_preds["predicted"]]
        if "low" in filtered_preds.columns and "high" in filtered_preds.columns:
            y_cols.extend([filtered_preds["low"].dropna(), filtered_preds["high"].dropna()])
        all_y = pd.concat(y_cols, ignore_index=True).dropna()

        # Lock the chart's X and Y axes to the exact min/max of the data so users can box-zoom in
        # and double-click to reset, but can never zoom out into empty space past the data.
        apply_chart_bounds(
            fig_avp,
            x_min=filtered_preds["period"].min(),
            x_max=filtered_preds["period"].max(),
            x_title="Week",
            y_title="Weekly Storage Change (Bcf)",
            y_min=float(all_y.min()) if not all_y.empty else None,
            y_max=float(all_y.max()) if not all_y.empty else None,
        )
        fig_avp.update_layout(height=440)
        st.plotly_chart(fig_avp, use_container_width=True)

# --- Section 3: MAE by Month ---
st.subheader("3. Model MAE by Calendar Month")
st.markdown(
    f"How to read this chart: Shorter bars mean better accuracy—each bar shows a model's average error ({term_tip('MAE')}, in {term_tip('Bcf')}) "
    f"for that month of the year across the backtest.",
    unsafe_allow_html=True,
    )

if not predictions_df.empty:
    # Copy predictions and calculate the absolute miss |actual - predicted| for every single week
    monthly_df = predictions_df.copy()
    monthly_df["period"] = pd.to_datetime(monthly_df["period"])
    monthly_df["abs_error"] = (monthly_df["actual"] - monthly_df["predicted"]).abs()

    # Extract both the month number (1-12 for chronological sorting) and 3-letter name ("Jan"-"Dec" for labels)
    monthly_df["month_num"] = monthly_df["period"].dt.month
    monthly_df["month_name"] = monthly_df["period"].dt.strftime("%b")

    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Group by model and month, then average the weekly absolute errors to get each model's monthly MAE
    mae_by_month = (
        monthly_df.groupby(["model", "month_num", "month_name"], as_index=False)["abs_error"]
        .mean()
        .rename(columns={"abs_error": "mae"})
        .sort_values("month_num")
    )

    # Order the models from best overall MAE to worst so the grouped bars always appear in rank order
    if not results_df.empty and "model" in results_df.columns:
        model_order = [
            m for m in results_df.sort_values("mae")["model"].tolist()
            if m in mae_by_month["model"].unique()
        ]
    else:
        model_order = sorted(mae_by_month["model"].unique().tolist())

    # Create a grouped bar chart with one colored bar series per model
    fig_month = go.Figure()
    for model_name in model_order:
        m_df = mae_by_month[mae_by_month["model"] == model_name].sort_values("month_num")
        fig_month.add_trace(
            go.Bar(
                x=m_df["month_name"],
                y=m_df["mae"],
                name=format_model_name(model_name),
                marker_color=get_model_color(model_name),
                hovertemplate=f"{format_model_name(model_name)} (%{{x}}): %{{y:.1f}} Bcf<extra></extra>",
            )
        )

    # Find the tallest bar so we can lock the Y-axis ceiling just above it (max_mae * 1.08)
    max_mae = float(mae_by_month["mae"].max()) if not mae_by_month.empty else 50.0
    fig_month.update_layout(
        barmode="group",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font={"family": "sans-serif", "color": "#18222C", "size": 13},
        height=400,
        margin={"l": 16, "r": 16, "t": 32, "b": 16},
        # Lock the 12 calendar months (-0.5 to 11.5 in category index space) so users can't zoom out past Jan-Dec
        xaxis={
            "title": "Month of Year",
            "categoryorder": "array",
            "categoryarray": month_order,
            "range": [-0.5, 11.5],
            "minallowed": -0.5,
            "maxallowed": 11.5,
        },
        yaxis={
            "title": "Mean Absolute Error (Bcf)",
            "range": [0, max_mae * 1.08],
            "minallowed": 0,
            "maxallowed": max_mae * 1.08,
            "gridcolor": "#E5E2DC",
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
    )
    st.plotly_chart(fig_month, use_container_width=True)