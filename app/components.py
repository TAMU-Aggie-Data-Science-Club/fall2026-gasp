# Allow modern type hints to work on Python 3.9+
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

#FIXED MODEL COLOR PALETTE. Every model gets one distinct color across the entire app 
MODEL_COLORS: dict[str, str] = {
    "actual": "#18222C",              # Charcoal black for actual EIA storage data
    "ensemble": "#1B4F72",            # Deep Navy
    "lightgbm": "#D95F02",            # Burnt Orange
    "sarimax": "#008080",             # Deep Teal
    "sarima": "#C0392B",              # Crimson Red
    "five_year_average": "#6C3483",   # Deep Plum
    "last_change": "#8C510A",         # Warm Ochre Brown
    "same_week_last_year": "#2E8B57", # Forest Green
}

# Clean readable names for each model ID
MODEL_LABELS: dict[str, str] = {
    "ensemble": "Ensemble",
    "lightgbm": "LightGBM",
    "sarimax": "SARIMAX",
    "sarima": "SARIMA",
    "five_year_average": "5-Year Average",
    "last_change": "Last Week's Change",
    "same_week_last_year": "Same Week Last Year",
}

# COMPLETE GLOSSARY: Written for someone who has never traded natural gas (about 1-2 sentences per term).
GLOSSARY: dict[str, str] = {
    "Bcf": "Billion cubic feet, the standard volume unit used to measure natural gas in underground storage. One Bcf is enough gas to heat roughly 10,000 U.S. homes for a year.",
    "working gas in storage": "The volume of natural gas in underground caverns and depleted reservoirs that can actually be pumped out and delivered to consumers. It excludes permanent 'base gas' needed to maintain underground pressure.",
    "injection": "A positive weekly storage change (+Bcf), meaning more gas was produced than the country burned and the extra was pumped into underground storage. Injections usually happen during mild spring through fall weather.",
    "withdrawal": "A negative weekly storage change (-Bcf), meaning the country burned more gas than was produced and pulled reserves out of storage. Withdrawals mainly happen in winter when heating demand spikes.",
    "weekly change": "The net amount of natural gas added to (injection) or removed from (withdrawal) storage from one Friday to the next, measured in Bcf.",
    "EIA weekly storage report": "The official U.S. Energy Information Administration survey released every Thursday at 10:30 AM ET reporting total working gas in storage as of the prior Friday.",
    "Lower 48": "The 48 contiguous U.S. states (excluding Alaska and Hawaii) whose pipelines form the main interconnected natural gas grid.",
    "five-year average and five-year range": "The historical mean, minimum, and maximum storage levels recorded on the same calendar week over the previous five years. Analysts use this benchmark to judge whether current supplies are tight or comfortable.",
    "surplus/deficit": "The difference in Bcf between current storage levels and the five-year average for the same week of the year. A positive number is a surplus (more gas than normal); a negative number is a deficit.",
    "HDD": "Heating Degree Days measure how cold a week was below 65°F. More HDDs mean colder weather, higher furnace usage, and larger storage withdrawals.",
    "CDD": "Cooling Degree Days measure how hot a week was above 65°F. More CDDs mean hotter weather, heavier air-conditioning electricity demand, and smaller gas injections because power plants burn gas to generate electricity.",
    "gas-weighted vs population-weighted": "Gas-weighted weather gives more weight to states with heavy residential gas heating (like the Midwest and Northeast), making it best for predicting winter heating demand. Population-weighted weather weights states by total residents, which tracks summer air-conditioning demand.",
    "vs normal": "The difference between a week's actual degree days and the long-term climate normal for that calendar week. Positive values mean the week was colder (for HDD) or hotter (for CDD) than typical for that time of year.",
    "forecast range (90%)": "The shaded low-to-high interval around a forecast designed to contain the true weekly storage change 90% of the time (9 out of 10 weeks).",
    "coverage": "The percentage of historical backtest weeks where the actual storage change landed inside the model's 90% forecast range. A well-calibrated 90% interval has coverage near 0.90 (90%).",
    "MAE": "Mean Absolute Error is the average size of a model's miss in Bcf, ignoring whether it guessed too high or too low. Lower MAE means a more accurate model.",
    "RMSE": "Root Mean Squared Error squares each weekly error before averaging and taking the square root, which penalizes big one-week misses much more heavily than small ones.",
    "bias": "The average signed error across all backtest weeks. A bias near zero means the model does not systematically over-predict or under-predict.",
    "backtest": "A walk-forward historical simulation from 2021 to 2025 where each model predicts one week ahead using only the data that was actually available at that point in time.",
    "live forecast": "The real-time prediction for the upcoming Thursday EIA storage report, generated before the official number is released.",
    "SARIMA": "Seasonal Autoregressive Integrated Moving Average, a statistical time-series model that predicts storage changes purely from past storage patterns and annual seasonality without looking at weather.",
    "SARIMAX": "An extension of SARIMA that adds external weather variables (Heating and Cooling Degree Days) to adjust the storage forecast for unusually hot or cold weeks.",
    "LightGBM": "A gradient-boosted decision tree machine learning model that learns nonlinear relationships between storage history, seasonal cycles, and regional weather.",
    "ensemble": "A combined forecast that averages the predictions of top statistical and machine learning models to reduce individual model blind spots.",
}


def get_model_color(model: str) -> str:
    """Return the fixed distinct hex color for a given model name."""
    return MODEL_COLORS.get(str(model), "#4A6572")


def format_model_name(model: str) -> str:
    """Return a clean human-readable display name for a model key."""
    return MODEL_LABELS.get(str(model), str(model))


def format_bcf_change(value: float | int, precision: int = 0) -> str:
    """
    Turn a signed number into plain English:
      -85 -> '85 Bcf withdrawal'
      +40 -> '40 Bcf injection'
    """
    if pd.isna(value):
        return "N/A"
    amt = abs(float(value))
    num_str = f"{amt:.{precision}f}" if precision > 0 else f"{round(amt):.0f}"
    if value < 0:
        return f"{num_str} Bcf withdrawal"
    if value > 0:
        return f"{num_str} Bcf injection"
    return "0 Bcf change"


def help_text(term: str) -> str:
    """Return the plain-English glossary definition for a given term."""
    return GLOSSARY.get(term, "")


def term_tip(term: str, label: str | None = None) -> str:
    """
    Wrap a term in an HTML <abbr> tag so hovering over it shows its plain-English
    glossary definition as a tooltip.
    """
    display = label or term
    definition = GLOSSARY.get(term, "").replace('"', "&quot;")
    if not definition:
        return display
    return (
        f'<abbr title="{definition}" '
        f'style="text-decoration: underline dotted #1B4F72; cursor: help;">{display}</abbr>'
    )


def render_last_updated(storage_week: str | None = None, weather_week: str | None = None) -> None:
    """Render a clean one-line banner showing when storage and weather data were last updated."""
    parts = []
    if storage_week:
        parts.append(f"Newest stored storage week: **{storage_week}**")
    if weather_week:
        parts.append(f"Newest stored weather week: **{weather_week}**")
    if parts:
        st.caption(" · ".join(parts))


def render_empty_state(message: str = "The first live forecast runs Wednesday 11/11.") -> None:
    """Show a friendly plain-English message instead of an error or blank chart when data isn't ready."""
    st.info(message)


def apply_chart_bounds(
    fig: go.Figure,
    x_min,
    x_max,
    x_title: str,
    y_title: str,
    y_min: float | None = None,
    y_max: float | None = None,
) -> go.Figure:
    """
    Apply our zooming standard to any Plotly chart:
    1. Every chart opens fitted to all of its data.
    2. Locking minallowed/maxallowed makes that full fit the furthest anyone can zoom out.
    3. Users can box-zoom in, and double-clicking resets to the full fit.
    4. Every axis shows explicit units.
    """
    xaxis_cfg: dict = {
        "title": {"text": x_title, "font": {"size": 13, "color": "#18222C"}},
        "range": [x_min, x_max],
        "minallowed": x_min,
        "maxallowed": x_max,
        "showgrid": True,
        "gridcolor": "#E5E2DC",
        "zeroline": False,
    }
    yaxis_cfg: dict = {
        "title": {"text": y_title, "font": {"size": 13, "color": "#18222C"}},
        "showgrid": True,
        "gridcolor": "#E5E2DC",
        "zeroline": True,
        "zerolinecolor": "#B0ADA6",
    }

    # Adds a small 6% visual breathing room to the top/bottom of the Y-axis and lock zoom-out limit
    if y_min is not None and y_max is not None and pd.notna(y_min) and pd.notna(y_max):
        span = float(y_max) - float(y_min)
        pad = max(span * 0.06, 1.0)
        lo, hi = float(y_min) - pad, float(y_max) + pad
        yaxis_cfg["range"] = [lo, hi]
        yaxis_cfg["minallowed"] = lo
        yaxis_cfg["maxallowed"] = hi

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font={"family": "sans-serif", "color": "#18222C", "size": 13},
        margin={"l": 16, "r": 16, "t": 32, "b": 16},
        xaxis=xaxis_cfg,
        yaxis=yaxis_cfg,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "title_text": "",
        },
        hovermode="x unified",
    )
    return fig