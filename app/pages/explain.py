from __future__ import annotations

from datetime import datetime
import os
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Only import db at the top level.Keeps `live` and `explain` inside functions
# below so Python doesn't try to load LightGBM/libomp on macOS just to render UI charts.
from gasp import db
from app.components import (
    apply_chart_bounds,
    format_bcf_change,
    render_empty_state,
    term_tip,
)

# Editorial colors matching our light theme:
# Burnt orange for inputs pushing toward withdrawal and deep navy for injection
WITHDRAWAL_COLOR = "#D95F02"
INJECTION_COLOR = "#1B4F72"

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _active_db_path() -> str:
    """Return GASP_DB_PATH if set in the terminal (e.g., data/deploy_copy.db), else default db.DB_PATH."""
    return os.environ.get("GASP_DB_PATH") or str(db.DB_PATH)


def load_latest_explanations() -> pd.DataFrame:
    """
    Loads the live_explanations table.
    Tries live.load_explanations() first, then falls back to reading SQLite directly
    so macOS environments without libomp still display the live forecast breakdown.
    """
    active_path = _active_db_path()
    try:
        from gasp import live

        if hasattr(live, "load_explanations"):
            df = live.load_explanations(path=active_path)
            if not df.empty:
                return df
    except TypeError:
        try:
            from gasp import live

            df = live.load_explanations()
            if not df.empty:
                return df
        except Exception:
            pass
    except Exception:
        pass

    try:
        with sqlite3.connect(active_path) as conn:
            return pd.read_sql_query("SELECT * FROM live_explanations", conn)
    except Exception:
        return pd.DataFrame()


def load_seasonal_history() -> pd.DataFrame:
    """
    Loads the explanation_history table.
    Team B's Piece 2 adds explain.load_explanation_history(). Until that merges,
    we also check SQLite directly and return an empty DataFrame if absent so the page
    renders a friendly empty state instead of an error.
    """
    active_path = _active_db_path()
    try:
        from gasp import explain

        if hasattr(explain, "load_explanation_history"):
            df = explain.load_explanation_history(path=active_path)
            if not df.empty:
                return df
    except TypeError:
        try:
            from gasp import explain

            df = explain.load_explanation_history()
            if not df.empty:
                return df
        except Exception:
            pass
    except Exception:
        pass

    try:
        with sqlite3.connect(active_path) as conn:
            return pd.read_sql_query("SELECT * FROM explanation_history", conn)
    except Exception:
        return pd.DataFrame()


def build_top_reason_sentence(top_row: pd.Series, final_forecast: float) -> str:
    """Write a plain-English headline sentence naming the single biggest driver of this week's forecast."""
    label = str(top_row["label"])
    contrib = float(top_row["contribution"])
    overall_dir = "withdrawal" if final_forecast < 0 else "injection"
    push_dir = "withdrawal" if contrib < 0 else "injection"

    # Keeps acronyms (like HDD, CDD, EIA) uppercase while formatting normal labels cleanly
    if len(label) > 1 and not label[:2].isupper():
        clean_label = label[0].upper() + label[1:]
    else:
        clean_label = label

    if push_dir == overall_dir:
        return (
            f"**{clean_label}** ({contrib:+.1f} Bcf) is the biggest reason "
            f"we expect a **{overall_dir}** this week."
        )
    return (
        f"**{clean_label}** ({contrib:+.1f} Bcf toward {push_dir}) is the single largest "
        f"adjustment in this week's **{overall_dir}** forecast."
    )


st.title("What Is Driving the Forecast")

with st.spinner("Loading forecast feature breakdowns..."):
    exp_df = load_latest_explanations()
    history_df = load_seasonal_history()



# PART 1: THIS WEEK'S LIVE FORECAST BREAKDOWN

st.subheader("1. This Week's Live Forecast, Broken Down")

if exp_df.empty:
    render_empty_state(
        "No live forecast breakdown is stored in this database yet. "
        "(Run with GASP_DB_PATH=data/deploy_copy.db to view the live deployed forecast.)"
    )
else:
    exp_df = exp_df.copy()
    exp_df["target"] = pd.to_datetime(exp_df["target"])

    # Filters to the most recent target Friday in live_explanations
    latest_target = exp_df["target"].max()
    week_df = exp_df[exp_df["target"] == latest_target].copy()

    # Verifys how the pieces add up: base_value ("an average week") + all contributions = LightGBM's forecast
    base_val = float(week_df["base_value"].iloc[0])
    total_contrib = float(week_df["contribution"].sum())
    lightgbm_forecast = base_val + total_contrib

    # Sorts all inputs by absolute size to grab the 8 largest contributions
    week_df["abs_contrib"] = week_df["contribution"].abs()
    top8_df = week_df.sort_values("abs_contrib", ascending=False).head(8).copy()

    # Simple sentence above the chart naming the biggest driver
    st.markdown(build_top_reason_sentence(top8_df.iloc[0], lightgbm_forecast))

    # Shows exact arithmetic so anyone can verify base_value + contributions = LightGBM forecast
    st.markdown(
        f"**How the pieces add up for the week ending {latest_target.strftime('%b %d, %Y')}:** "
        f"Starting from an average week (**{base_val:+.1f} {term_tip('Bcf')}**), "
        f"adding all feature contributions (**{total_contrib:+.1f} {term_tip('Bcf')}**) "
        f"equals {term_tip('LightGBM')}'s forecast of **{lightgbm_forecast:+.1f} {term_tip('Bcf')}** "
        f"({format_bcf_change(lightgbm_forecast, precision=1)}).",
        unsafe_allow_html=True,
    )

    # One-sentence chart explanation without a "How to read this:" prefix
    st.markdown(
        f"Each bar shows how many {term_tip('Bcf')} one input pushed this week's forecast "
        f"toward a {term_tip('withdrawal')} (left, orange) or toward an {term_tip('injection')} (right, navy).",
        unsafe_allow_html=True,
    )

    # Reverses order so the #1 largest bar sits at the very top of the horizontal chart
    plot_top8 = top8_df.iloc[::-1].copy()
    bar_colors = [
        WITHDRAWAL_COLOR if c < 0 else INJECTION_COLOR
        for c in plot_top8["contribution"]
    ]
    direction_words = [
        "Toward withdrawal" if c < 0 else "Toward injection"
        for c in plot_top8["contribution"]
    ]

    fig_week = go.Figure()
    fig_week.add_trace(
        go.Bar(
            x=plot_top8["contribution"],
            y=plot_top8["label"],
            orientation="h",
            marker_color=bar_colors,
            customdata=list(zip(plot_top8["value"], direction_words)),
            text=[f"{c:+.1f} Bcf" for c in plot_top8["contribution"]],
            textposition="auto",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Impact: %{x:+.1f} Bcf (%{customdata[1]})<br>"
                "This week's value: %{customdata[0]:.2f}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    # Fits X-axis bounds to the data with breathing room for bar labels and lock zoom-out limits
    max_span = float(plot_top8["abs_contrib"].max()) if not plot_top8.empty else 10.0
    x_pad = max(max_span * 0.22, 1.5)
    x_lo = min(float(plot_top8["contribution"].min()) - x_pad, -1.0)
    x_hi = max(float(plot_top8["contribution"].max()) + x_pad, 1.0)

    apply_chart_bounds(
        fig_week,
        x_min=x_lo,
        x_max=x_hi,
        x_title="Contribution to Weekly Storage Change Forecast (Bcf)",
        y_title="Model Input (Top 8)",
    )
    fig_week.update_layout(
        height=390,
        xaxis={
            "title": "Contribution to Weekly Storage Change Forecast (Bcf)",
            "range": [x_lo, x_hi],
            "minallowed": x_lo,
            "maxallowed": x_hi,
            "zeroline": True,
            "zerolinecolor": "#18222C",
            "zerolinewidth": 1.5,
            "gridcolor": "#E5E2DC",
        },
        yaxis={
            "title": "",
            "range": [-0.5, len(plot_top8) - 0.5],
            "minallowed": -0.5,
            "maxallowed": len(plot_top8) - 0.5,
        },
        hovermode="closest",
    )
    st.plotly_chart(fig_week, use_container_width=True)



# PART 2: WHAT DRIVES THE FORECAST BY SEASON

st.subheader("2. What Drives the Forecast by Season")

if history_df.empty:
    render_empty_state(
        "Seasonal driver history will appear here once Team B's historical explanation table merges."
    )
else:
    history_df = history_df.copy()

    # Default the dropdown selector to the current calendar month
    current_month_num = datetime.now().month
    selected_month_name = st.selectbox(
        "Select a calendar month:",
        options=MONTH_NAMES,
        index=current_month_num - 1,
    )
    selected_month_num = MONTH_NAMES.index(selected_month_name) + 1

    # Filters to the chosen month and show its top 5 inputs by mean_abs_contribution
    month_rows = (
        history_df[history_df["month"] == selected_month_num]
        .sort_values("mean_abs_contribution", ascending=False)
        .head(5)
        .copy()
    )

    if month_rows.empty:
        render_empty_state(f"No historical explanation rows found for {selected_month_name}.")
    else:
        st.markdown(
            f"Bar length shows each input's average importance in **{selected_month_name}** "
            f"(mean absolute contribution in {term_tip('Bcf')}), while color shows whether it typically "
            f"pushes the forecast toward a {term_tip('withdrawal')} (orange) or {term_tip('injection')} (navy).",
            unsafe_allow_html=True,
        )

        # Reverses order so the #1 most important seasonal input sits at the top
        plot_season = month_rows.iloc[::-1].copy()
        season_colors = [
            WITHDRAWAL_COLOR if m < 0 else INJECTION_COLOR
            for m in plot_season["mean_contribution"]
        ]
        season_dirs = [
            f"Typically pushes toward withdrawal ({m:+.1f} Bcf avg)"
            if m < 0
            else f"Typically pushes toward injection ({m:+.1f} Bcf avg)"
            for m in plot_season["mean_contribution"]
        ]

        fig_season = go.Figure()
        fig_season.add_trace(
            go.Bar(
                x=plot_season["mean_abs_contribution"],
                y=plot_season["label"],
                orientation="h",
                marker_color=season_colors,
                customdata=season_dirs,
                text=[
                    f"{abs_c:.1f} Bcf ({'withdrawal' if signed_c < 0 else 'injection'})"
                    for abs_c, signed_c in zip(
                        plot_season["mean_abs_contribution"],
                        plot_season["mean_contribution"],
                    )
                ],
                textposition="auto",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Average importance: %{x:.1f} Bcf<br>"
                    "%{customdata}<extra></extra>"
                ),
                showlegend=False,
            )
        )

        max_abs = float(plot_season["mean_abs_contribution"].max()) if not plot_season.empty else 10.0
        s_hi = max(max_abs * 1.22, 1.0)

        apply_chart_bounds(
            fig_season,
            x_min=0.0,
            x_max=s_hi,
            x_title="Average Absolute Contribution (Bcf)",
            y_title="Top 5 Seasonal Inputs",
        )
        fig_season.update_layout(
            height=320,
            xaxis={
                "title": "Average Absolute Contribution (Bcf)",
                "range": [0.0, s_hi],
                "minallowed": 0.0,
                "maxallowed": s_hi,
                "gridcolor": "#E5E2DC",
            },
            yaxis={
                "title": "",
                "range": [-0.5, len(plot_season) - 0.5],
                "minallowed": -0.5,
                "maxallowed": len(plot_season) - 0.5,
            },
            hovermode="closest",
        )
        st.plotly_chart(fig_season, use_container_width=True)
