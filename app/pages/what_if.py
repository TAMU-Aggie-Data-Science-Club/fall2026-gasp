# What-If page: shows how the forecast changes if this week's weather had been different
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components import (
    apply_chart_bounds,
    format_model_name,
    get_model_color,
    is_published,
    render_empty_state,
    term_tip,
)
from gasp import db

# one tab per model, LightGBM opens by default 
MODELS = ["lightgbm", "sarimax"]

REQUIRED_COLUMNS = ["model", "target", "weather_change_pct", "predicted", "low", "high"]


#work on data 


# load the what-if grid, empty table if it doesn't exist yet
def load_whatif(path: str | Path = db.DB_PATH) -> pd.DataFrame:
    # use Team C's function once it's merged
    try:
        from gasp import live  # type: ignore

        if hasattr(live, "load_whatif"):
            return live.load_whatif(path=path)
    except Exception:
        pass

    # until then, read the table straight from the database
    try:
        with sqlite3.connect(path) as conn:
            return pd.read_sql_query("SELECT * FROM whatif", conn)
    except Exception:
        return pd.DataFrame()


# clean up the types and keep only the latest published target week
def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty or any(c not in raw for c in ("model", "target", "weather_change_pct", "predicted")):
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    df = raw.copy()
    for col in ("low", "high"):
        if col not in df:
            df[col] = float("nan")
    df["target"] = pd.to_datetime(df["target"], errors="coerce")
    for col in ("weather_change_pct", "predicted", "low", "high"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["target", "weather_change_pct", "predicted"])

    # Only weeks whose forecast is already shown on the site (Sunday 5:30 PM Central)
    df = df[df["target"].apply(is_published)]
    if df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = df[df["target"] == df["target"].max()]
    return df[REQUIRED_COLUMNS].sort_values(["model", "weather_change_pct"]).reset_index(drop=True)


#wording for the tab


# "a withdrawal of 85 Bcf" / "an injection of 40 Bcf"
def bcf_phrase(value: float) -> str:
    if value < 0:
        return f"a withdrawal of {abs(value):.0f} Bcf"
    if value > 0:
        return f"an injection of {value:.0f} Bcf"
    return "no change in storage"


# short version for the big number boxes: "Withdrawal: 85 Bcf"
def bcf_short(value: float) -> str:
    if value < 0:
        return f"Withdrawal: {abs(value):.0f} Bcf"
    if value > 0:
        return f"Injection: {value:.0f} Bcf"
    return "No change"


# the likely range in plain english, blank if the model has no range
def range_phrase(low: float, high: float) -> str:
    if pd.isna(low) or pd.isna(high):
        return ""
    if high <= 0:
        return f"likely a withdrawal of {abs(high):.0f} to {abs(low):.0f} Bcf"
    if low >= 0:
        return f"likely an injection of {low:.0f} to {high:.0f} Bcf"
    return f"likely anywhere from a withdrawal of {abs(low):.0f} Bcf to an injection of {high:.0f} Bcf"


# "had been 20% colder": colder/warmer in winter, hotter/cooler in summer, plain wording otherwise
def weather_clause(pct: int, target: pd.Timestamp) -> str:
    amount = abs(int(pct))
    if target.month in (11, 12, 1, 2, 3):
        return f"had been {amount}% {'colder' if pct > 0 else 'warmer'}"
    if target.month in (6, 7, 8, 9):
        return f"had been {amount}% {'hotter' if pct > 0 else 'cooler'}"
    return f"had had {amount}% {'more' if pct > 0 else 'fewer'} heating and cooling degree days"


# building the actual chart 


# forecast and range across the whole slider, with the chosen step marked
def build_chart(sub: pd.DataFrame, model: str, pct: int) -> go.Figure:
    color = get_model_color(model)
    name = format_model_name(model)
    fig = go.Figure()
    values: list[float] = []

    # range band: top edge, then the bottom edge filled up to it
    if sub["low"].notna().any() and sub["high"].notna().any():
        fig.add_trace(
            go.Scatter(x=sub["weather_change_pct"], y=sub["high"], mode="lines", line={"width": 0},
                       showlegend=False, hoverinfo="skip")
        )
        fig.add_trace(
            go.Scatter(x=sub["weather_change_pct"], y=sub["low"], mode="lines", line={"width": 0},
                       fill="tonexty", fillcolor="rgba(128,128,128,0.18)",
                       showlegend=False, hoverinfo="skip")
        )
        values += sub["low"].dropna().tolist() + sub["high"].dropna().tolist()

    # the forecast line
    fig.add_trace(
        go.Scatter(x=sub["weather_change_pct"], y=sub["predicted"], mode="lines+markers",
                   name=f"{name} forecast", line={"color": color, "width": 2}, marker={"size": 6})
    )
    values += sub["predicted"].tolist()

    # big marker on the step the slider is on
    chosen = sub[sub["weather_change_pct"] == pct]
    if not chosen.empty:
        fig.add_trace(
            go.Scatter(x=chosen["weather_change_pct"], y=chosen["predicted"], mode="markers",
                       name="Your choice", marker={"size": 13, "color": color, "symbol": "diamond",
                                                    "line": {"width": 2, "color": "#18222C"}})
        )

    # fit the axes to the data, a little room on each side
    x_min = float(sub["weather_change_pct"].min()) - 1.5
    x_max = float(sub["weather_change_pct"].max()) + 1.5
    fig = apply_chart_bounds(
        fig, x_min, x_max,
        x_title="Change in heating and cooling degree days (%)",
        y_title="Forecast weekly storage change (Bcf; below 0 = withdrawal)",
        y_min=min(values), y_max=max(values),
    )
    fig.update_xaxes(dtick=10, ticksuffix="%")
    return fig


#one tab


def render_model(df: pd.DataFrame, model: str) -> None:
    name = format_model_name(model)
    sub = df[df["model"] == model]
    if sub.empty:
        render_empty_state(f"The what-if results for {name} will appear here once the weekly forecast has run.")
        return

    target = pd.Timestamp(sub["target"].iloc[0])

    # the slider, in steps of 5
    pct = st.slider(
        "Change in heating and cooling degree days",
        min_value=-30, max_value=30, value=0, step=5, format="%d%%",
        key=f"whatif_pct_{model}",
    )

    row = sub[sub["weather_change_pct"] == pct]
    base = sub[sub["weather_change_pct"] == 0]
    if row.empty or base.empty:
        render_empty_state("This step isn't available yet.")
        return
    new, old = float(row["predicted"].iloc[0]), float(base["predicted"].iloc[0])
    label = "0%" if pct == 0 else f"{pct:+d}%"

    # key sentence first
    if pct == 0:
        st.subheader(f"With this week's actual weather, {name} expects {bcf_phrase(old)}.")
    else:
        # skip repeating the type if both are the same direction ("instead of 85")
        same_direction = (new < 0) == (old < 0) and new != 0 and old != 0
        before = f"{abs(old):.0f}" if same_direction else bcf_phrase(old)
        st.subheader(
            f"If this week {weather_clause(pct, target)}, {name} expects {bcf_phrase(new)} instead of {before}."
        )
    rng = range_phrase(float(row["low"].iloc[0]), float(row["high"].iloc[0]))
    if rng:
        st.write(f"At {label} the forecast is {rng}.")

    # side by side: the real forecast and the chosen step
    left, right = st.columns(2)
    left.metric("Actual forecast (0%)", bcf_short(old))
    right.metric(
        f"At {label}",
        bcf_short(new),
        delta=None if pct == 0 else f"{new - old:+.0f} Bcf vs actual",
        delta_color="off",
    )

    # chart across the whole slider
    st.plotly_chart(build_chart(sub, model, pct))
    st.markdown(
        f"Each point is {name}'s forecast if the week's weather had been that much different; "
        "the gray band is its likely range, and the diamond is the step you picked. "
        "Drag to zoom in, double-click to reset.",
    )
    st.caption(
        "For unusually extreme weeks, the forecast can stop changing at the far ends of the slider: "
        "the model can't predict beyond the most extreme weather it has seen."
    )


#the actual page 


def render() -> None:
    st.title("Weather What-If Simulator")

    # load the data with a spinner
    with st.spinner("Loading what-if results..."):
        df = prepare(load_whatif())

    # nothing yet: friendly message instead of an empty page
    if df.empty:
        render_empty_state("What-if results will appear here once the weekly forecast has run.")
        return

    target = pd.Timestamp(df["target"].iloc[0])
    st.write(f"Week ending {target:%b} {target.day}. Slide to see how each model's forecast would change.")

    # what the slider means, in plain words
    st.markdown(
        f"+20% means 20% more heating and cooling degree days ({term_tip('HDD')} and {term_tip('CDD')}) "
        "than this week actually had (colder in winter, hotter in summer). "
        "Negative numbers mean fewer, so milder weather.",
        unsafe_allow_html=True,
    )

    # one model per tab, nothing combined
    tabs = st.tabs([format_model_name(m) for m in MODELS])
    for tab, model in zip(tabs, MODELS):
        with tab:
            render_model(df, model)


render()