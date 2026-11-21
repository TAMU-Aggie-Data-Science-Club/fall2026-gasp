# Allow modern type hints to work on Python 3.9+
from __future__ import annotations
from datetime import datetime, time
from zoneinfo import ZoneInfo

import html
import re

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
    "contribution": "How many Bcf one input (like colder-than-normal weather) added to or subtracted from a model's forecast. Start from the model's average-week forecast, add up all the contributions, and you get its forecast for the week.",
    "scenario": "One possible way the future could play out, such as storage through the winter if the weather repeated a past year's. Looking at many scenarios shows the range of outcomes instead of a single guess.",
    "what-if": "A test that changes one thing, such as making this week's weather 20% colder, and shows how the forecast would change. It shows how much the forecast depends on the weather; it is not a prediction that the weather will change.",
    "regional weather": "Heating and cooling degree days measured for separate parts of the country, such as the Midwest or the Northeast, instead of one national number, because gas demand depends on where it is cold or hot.",
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


# id for a term's spot in the glossary, e.g. "surplus/deficit" -> "term-surplus-deficit"
def term_id(term: str) -> str:
    return "term-" + re.sub(r"[^a-z0-9]+", "-", term.lower()).strip("-")


# styles for the term tooltips (loaded once per page from dashboard.py)
TERM_TIP_CSS = """
.term-tip { position: relative; display: inline; border-bottom: 1px dotted currentColor; cursor: help; }
.term-tip .term-link { color: inherit; text-decoration: none; }
.term-tip .tip-body {
    display: none; position: absolute; left: 0; bottom: 100%; padding-bottom: 6px;
    width: max-content; max-width: min(20rem, 80vw); z-index: 1000;
    font-size: 0.875rem; font-weight: 400; line-height: 1.4; text-align: left; white-space: normal;
}
.term-tip .tip-box {
    display: block; padding: 0.6rem 0.75rem; border-radius: 6px;
    background: #18222C; color: #FFFFFF; border: 1px solid rgba(255, 255, 255, 0.25);
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
}
.term-tip .tip-more { display: block; margin-top: 0.4rem; color: #9FD0FF; text-decoration: underline; }
.term-tip:hover .tip-body, .term-tip:focus .tip-body, .term-tip:focus-within .tip-body { display: block; }
/* on touch screens a tap shows the definition; the link to the glossary is inside it */
@media (hover: none) { .term-tip .term-link { pointer-events: none; } }
"""


def inject_term_tip_css() -> None:
    """Load the tooltip styles. Call once near the top of dashboard.py, before the page runs."""
    css = f"<style>{TERM_TIP_CSS}</style>"
    if hasattr(st, "html"):
        st.html(css)
    else:
        st.markdown(css, unsafe_allow_html=True)


def term_tip(term: str, label: str | None = None) -> str:
    """
    Wrap a term so its definition pops up on hover or tap, and clicking it opens the glossary
    on the About page. Use inside st.markdown(..., unsafe_allow_html=True).
    """
    display = html.escape(label or term)
    definition = GLOSSARY.get(term, "")
    if not definition:
        return display
    link = f"about?term={term_id(term)}#{term_id(term)}"
    return (
        '<span class="term-tip" tabindex="0">'
        f'<a class="term-link" href="{link}" target="_self">{display}</a>'
        '<span class="tip-body"><span class="tip-box">'
        f"{html.escape(definition)}"
        f'<a class="tip-more" href="{link}" target="_self">See the glossary</a>'
        "</span></span></span>"
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


def render_empty_state(message: str = "This will appear here after the next weekly forecast.") -> None:
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

# Forecasts appear on the site from the Sunday after their week ends, 5:30 PM Central
PUBLISH_TZ = ZoneInfo("America/Chicago")


def is_published(target) -> bool:
    """True once a target week's forecast may be shown on the site."""
    day = (pd.Timestamp(target) + pd.Timedelta(days=2)).date()
    return datetime.now(PUBLISH_TZ) >= datetime.combine(day, time(17, 30), tzinfo=PUBLISH_TZ)