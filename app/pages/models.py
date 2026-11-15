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
    """Convert a solid hex color (like '#D95F02') into a see-through RGBA color for range boxes."""
    h = hex_color.lstrip("#")
    if len(h) == 6:
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
st.subheader("2. Actual vs. Predicted Weekly Storage Changes (2025 Backtest)")
st.markdown(
    f"Each shaded box is a model's {term_tip('forecast range (90%)', '90% range')} for that week, and the dotted line "
    f"is its forecast. The dark line with diamonds is what the EIA reported, in {term_tip('Bcf')}. Below 0 means gas "
    f"was pulled out of storage (a {term_tip('withdrawal')}); above 0 means gas was added (an {term_tip('injection')}). "
    f"Drag to zoom in, double-click to reset.",
    unsafe_allow_html=True,
)

if predictions_df.empty:
    render_empty_state("No saved model predictions found in the database.")
else:
    # Copy so we don't modify the cached table, and convert 'period' into calendar dates
    pred_df = predictions_df.copy()
    pred_df["period"] = pd.to_datetime(pred_df["period"])

    # Show only the most recent backtest year
    pred_df = pred_df[pred_df["period"].dt.year == 2025]

    # Models for the picker, ordered from best MAE to worst MAE
    if not results_df.empty and "model" in results_df.columns:
        available_models = [
            m for m in results_df.sort_values("mae")["model"].tolist()
            if m in pred_df["model"].unique()
        ]
    else:
        available_models = sorted(pred_df["model"].unique().tolist())

    # Start with LightGBM only; other models can be added from the picker
    selected_models = st.multiselect(
        "Models to show:",
        options=available_models,
        default=["lightgbm"] if "lightgbm" in available_models else available_models[:1],
        format_func=format_model_name,
        help=help_text("backtest"),
    )

    if not selected_models:
        st.info("Select at least one model above to compare against actual weekly changes.")
    else:
        filtered_preds = pred_df[pred_df["model"].isin(selected_models)].copy()

        # One timeline of actual changes: every model row for a week has the same actual
        actual_df = (
            filtered_preds[["period", "actual"]]
            .drop_duplicates(subset=["period"])
            .sort_values("period")
        )

        fig_avp = go.Figure()
        box_width = pd.Timedelta(days=4).total_seconds() * 1000  # bar widths on a date axis are in ms

        for model_name in selected_models:
            m_df = filtered_preds[filtered_preds["model"] == model_name].sort_values("period")
            color = get_model_color(model_name)
            name = format_model_name(model_name)

            # Range boxes: a floating bar from low to high for each week that has a range
            if "low" in m_df.columns and "high" in m_df.columns:
                ranged = m_df.dropna(subset=["low", "high"])
                if not ranged.empty:
                    fig_avp.add_trace(
                        go.Bar(
                            x=ranged["period"],
                            base=ranged["low"],
                            y=ranged["high"] - ranged["low"],
                            width=box_width,
                            name=f"{name} 90% range",
                            marker={
                                "color": _hex_to_rgba(color, 0.18),
                                "line": {"color": _hex_to_rgba(color, 0.6), "width": 1},
                            },
                            customdata=ranged[["low", "high"]].to_numpy(),
                            hovertemplate=f"{name} range: %{{customdata[0]:.0f}} to %{{customdata[1]:.0f}} Bcf<extra></extra>",
                        )
                    )

            # The model's forecast: dotted line with dots
            fig_avp.add_trace(
                go.Scatter(
                    x=m_df["period"],
                    y=m_df["predicted"],
                    mode="lines+markers",
                    name=f"{name} forecast",
                    line={"color": color, "width": 2, "dash": "dot"},
                    marker={"size": 5, "color": color},
                    hovertemplate=f"{name} forecast: %{{y:.0f}} Bcf<extra></extra>",
                )
            )

        # What the EIA reported: solid line with diamonds
        fig_avp.add_trace(
            go.Scatter(
                x=actual_df["period"],
                y=actual_df["actual"],
                mode="lines+markers",
                name="Actual (EIA)",
                line={"color": get_model_color("actual"), "width": 2.5},
                marker={"size": 6, "symbol": "diamond", "color": get_model_color("actual")},
                hovertemplate="Actual: %{y:.0f} Bcf<extra></extra>",
            )
        )

        # Fit the axes to all plotted values, with room on each side for the boxes
        y_cols = [filtered_preds["actual"], filtered_preds["predicted"]]
        if "low" in filtered_preds.columns and "high" in filtered_preds.columns:
            y_cols.extend([filtered_preds["low"].dropna(), filtered_preds["high"].dropna()])
        all_y = pd.concat(y_cols, ignore_index=True).dropna()
        pad = pd.Timedelta(days=5)

        apply_chart_bounds(
            fig_avp,
            x_min=(filtered_preds["period"].min() - pad).to_pydatetime(),
            x_max=(filtered_preds["period"].max() + pad).to_pydatetime(),
            x_title="Week ending (Friday)",
            y_title="Weekly storage change (Bcf)",
            y_min=float(all_y.min()) if not all_y.empty else None,
            y_max=float(all_y.max()) if not all_y.empty else None,
        )
        # Boxes from different models overlap at the same week instead of sitting side by side
        fig_avp.update_layout(height=480, barmode="overlay")
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