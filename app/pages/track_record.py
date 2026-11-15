# Track Record page: how each live forecast did once the EIA report came out
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components import (
    MODEL_COLORS,
    apply_chart_bounds,
    format_bcf_change,
    format_model_name,
    get_model_color,
    help_text,
    render_empty_state,
    term_tip,
)
from gasp import db

# the columns we use from live_forecasts
REQUIRED_COLUMNS = ["model", "target", "predicted", "low", "high", "actual"]


#work on data


# load the live forecasts, empty table if they don't exist yet
def load_live_forecasts(path: str | Path = db.DB_PATH) -> pd.DataFrame:
    # use Team C's function once it's merged
    try:
        from gasp import live  # type: ignore

        if hasattr(live, "load_live_forecasts"):
            return live.load_live_forecasts(path=path)
    except Exception:
        pass

    # until then, read the table straight from the database
    try:
        with sqlite3.connect(path) as conn:
            return pd.read_sql_query("SELECT * FROM live_forecasts", conn)
    except Exception:
        return pd.DataFrame()


# clean up the types and sort by week, then model
def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty or "model" not in raw or "target" not in raw or "predicted" not in raw:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    df = raw.copy()

    # fill in any missing columns so the rest of the page doesn't break
    for col in ("low", "high", "actual"):
        if col not in df:
            df[col] = float("nan")

    df["target"] = pd.to_datetime(df["target"], errors="coerce")
    for col in ("predicted", "low", "high", "actual"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["target", "predicted"])
    return df[REQUIRED_COLUMNS].sort_values(["target", "model"]).reset_index(drop=True)


# add error, in-range check and running MAE for every row
def score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["model", "target"]).copy()

    # error = forecast minus actual (positive means the forecast was too high)
    out["error"] = out["predicted"] - out["actual"]
    out["abs_error"] = out["error"].abs()

    # did the actual land inside the range?
    released = out["actual"].notna()
    has_range = out["low"].notna() & out["high"].notna()
    inside = released & has_range & (out["actual"] >= out["low"]) & (out["actual"] <= out["high"])
    out["range_result"] = "Waiting for result"
    out.loc[released & ~has_range, "range_result"] = "No range"
    out.loc[released & has_range, "range_result"] = "No"
    out.loc[inside, "range_result"] = "Yes"

    # running MAE = average miss so far, per model (blank until the actual is out)
    out["running_mae"] = out.groupby("model")["abs_error"].transform(lambda s: s.expanding().mean())
    out.loc[~released, "running_mae"] = float("nan")
    return out.sort_values(["target", "model"]).reset_index(drop=True)


# one row per model across all weeks that have an actual
def summarize(scored: pd.DataFrame) -> pd.DataFrame:
    released = scored[scored["actual"].notna()]
    if released.empty:
        return pd.DataFrame()
    rows = []
    for model, g in released.groupby("model"):
        # only count weeks where the model gave a range
        ranged = g[g["range_result"].isin(["Yes", "No"])]
        rows.append(
            {
                "model": model,
                "weeks": len(g),
                "mae": g["abs_error"].mean(),
                "bias": g["error"].mean(),
                "in_range": (ranged["range_result"] == "Yes").mean() if len(ranged) else float("nan"),
                "ranged_weeks": len(ranged),
            }
        )
    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)


#work on display helpers


# "Nov 6" style date
def short_date(ts: pd.Timestamp) -> str:
    return f"{ts:%b} {ts.day}"


# number with a + or - in front, dash if it's missing
def signed(value: float, digits: int = 0) -> str:
    return "—" if pd.isna(value) else f"{value:+.{digits}f}"


# turn a hex color into a see-through one for the range boxes
def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# the table for one week: every model's forecast, range, actual, error
def week_table(g: pd.DataFrame, released: bool) -> pd.DataFrame:
    # best model first once the result is out, otherwise alphabetical
    g = g.sort_values("abs_error") if released else g.sort_values("model")
    return pd.DataFrame(
        {
            "Model": [format_model_name(m) for m in g["model"]],
            "Forecast (Bcf)": [signed(v) for v in g["predicted"]],
            "90% range (Bcf)": [
                "—" if pd.isna(lo) or pd.isna(hi) else f"{lo:+.0f} to {hi:+.0f}"
                for lo, hi in zip(g["low"], g["high"])
            ],
            "Actual (Bcf)": [signed(v) for v in g["actual"]],
            "Error (Bcf)": [signed(v) for v in g["error"]],
            "Actual inside range?": list(g["range_result"]),
            "Running MAE (Bcf)": ["—" if pd.isna(v) else f"{v:.1f}" for v in g["running_mae"]],
        }
    )


# forecasts vs actuals: each week's 90% range as a shaded box, forecasts as a dotted line,
# and the EIA actual as a solid line with diamonds
def build_chart(scored: pd.DataFrame, models: list[str]) -> go.Figure:
    fig = go.Figure()
    values: list[float] = []  # every y value we plot, used to fit the axis
    box_width = pd.Timedelta(days=4).total_seconds() * 1000  # bar widths on a date axis are in ms

    for model in models:
        sub = scored[scored["model"] == model].sort_values("target")
        color = get_model_color(model)
        name = format_model_name(model)

        # range boxes: a floating bar from low to high for each week that has a range
        ranged = sub.dropna(subset=["low", "high"])
        if not ranged.empty:
            fig.add_trace(
                go.Bar(
                    x=ranged["target"], base=ranged["low"], y=ranged["high"] - ranged["low"],
                    width=box_width, name=f"{name} 90% range",
                    marker={"color": rgba(color, 0.18), "line": {"color": rgba(color, 0.6), "width": 1}},
                    customdata=ranged[["low", "high"]].to_numpy(),
                    hovertemplate=f"{name} range: %{{customdata[0]:.0f}} to %{{customdata[1]:.0f}} Bcf<extra></extra>",
                )
            )
            values += ranged["low"].tolist() + ranged["high"].tolist()

        # the model's forecast: dotted line with markers
        fig.add_trace(
            go.Scatter(
                x=sub["target"], y=sub["predicted"], mode="lines+markers", name=f"{name} forecast",
                line={"color": color, "width": 2, "dash": "dot"},
                marker={"size": 9, "color": color},
                hovertemplate=f"{name} forecast: %{{y:.0f}} Bcf<extra></extra>",
            )
        )
        values += sub["predicted"].tolist()

    # the actual EIA numbers (only for weeks that have one): solid line with diamonds
    actual = scored.dropna(subset=["actual"]).groupby("target", as_index=False)["actual"].first()
    if not actual.empty:
        fig.add_trace(
            go.Scatter(
                x=actual["target"], y=actual["actual"], mode="lines+markers", name="Actual (EIA)",
                line={"color": MODEL_COLORS["actual"], "width": 3},
                marker={"size": 11, "symbol": "diamond", "color": MODEL_COLORS["actual"]},
                hovertemplate="Actual: %{y:.0f} Bcf<extra></extra>",
            )
        )
        values += actual["actual"].tolist()

    # fit the axes to the data each time the page loads, with room on each side for the boxes
    pad = pd.Timedelta(days=5)
    x_min = (scored["target"].min() - pad).to_pydatetime()
    x_max = (scored["target"].max() + pad).to_pydatetime()
    fig = apply_chart_bounds(
        fig, x_min, x_max,
        x_title="Week ending (Friday)",
        y_title="Weekly storage change (Bcf; below 0 = withdrawal)",
        y_min=min(values), y_max=max(values),
    )

    # boxes from different models overlap at the same week instead of sitting side by side
    fig.update_layout(barmode="overlay")

    # one tick per forecast week
    fig.update_xaxes(tickmode="array", tickvals=sorted(scored["target"].unique()), tickformat="%b %d")
    return fig


#work on the page


def render() -> None:
    st.title("Live Forecast Track Record")

    # load the data with a spinner
    with st.spinner("Loading live forecasts..."):
        df = prepare(load_live_forecasts())

    # nothing yet: friendly message instead of an empty page
    if df.empty:
        render_empty_state("The first live forecast runs Wednesday 11/11.")
        return

    scored = score(df)
    summary = summarize(scored)

    # top of the page: the key number in plain english
    if summary.empty:
        # forecasts exist but no report has come out yet
        latest = scored["target"].max()
        st.subheader("Waiting for the first official result")
        st.write(
            f"{scored['model'].nunique()} models have forecast the week ending {short_date(latest)}. "
            f"The EIA report comes out Thursday {short_date(latest + pd.Timedelta(days=6))} at 10:30 AM ET, "
            "and results will appear here."
        )
    else:
        best = summary.iloc[0]
        weeks = int(best["weeks"])
        st.subheader(
            f"So far, {format_model_name(best['model'])} has missed by the least: "
            f"{best['mae']:.0f} Bcf on average over {weeks} live week{'s' if weeks != 1 else ''}"
        )
        st.markdown(
            f"That average miss is the {term_tip('MAE', 'MAE')}. Lower is better.",
            unsafe_allow_html=True,
        )

        # running MAE per model across all live weeks
        table = pd.DataFrame(
            {
                "Model": [format_model_name(m) for m in summary["model"]],
                "Weeks scored": summary["weeks"],
                "Running MAE (Bcf)": summary["mae"].round(1),
                "Bias (Bcf)": summary["bias"].round(1),
                "Actual landed inside range": [
                    "—" if pd.isna(v) else f"{v:.0%} of {int(n)}"
                    for v, n in zip(summary["in_range"], summary["ranged_weeks"])
                ],
            }
        )
        # hover over a column name to see what it means
        st.dataframe(
            table,
            hide_index=True,
            column_config={
                "Running MAE (Bcf)": st.column_config.Column(help=help_text("MAE")),
                "Bias (Bcf)": st.column_config.Column(help=help_text("bias")),
                "Actual landed inside range": st.column_config.Column(help=help_text("forecast range (90%)")),
            },
        )
        st.markdown(
            f"Running {term_tip('MAE')} is each model's average miss across every live week with a result, "
            f"in {term_tip('Bcf')}. {term_tip('bias', 'Bias')} above 0 means the model has tended to forecast too high.",
            unsafe_allow_html=True,
        )

    # chart: LightGBM to start; other models can be added from the picker
    st.subheader("Forecasts vs actuals")
    all_models = sorted(scored["model"].unique())
    ranked = list(summary["model"]) if not summary.empty else all_models
    chosen = st.multiselect(
        "Models to show", all_models,
        default=["lightgbm"] if "lightgbm" in all_models else ranked[:1],
        format_func=format_model_name,
    )
    if chosen:
        st.plotly_chart(build_chart(scored, chosen))
        st.markdown(
            f"Each shaded box is a model's {term_tip('forecast range (90%)', '90% range')} for that week, "
            f"and the dotted line is its forecast. The dark line with diamonds is what the EIA reported, in {term_tip('Bcf')}. "
            f"Below 0 means gas was pulled out of storage (a {term_tip('withdrawal')}); above 0 means gas was added "
            f"(an {term_tip('injection')}). Drag to zoom in, double-click to reset.",
            unsafe_allow_html=True,
        )
    else:
        st.info("Pick at least one model to see the chart.")

    # week by week: newest first, the latest one starts open
    st.subheader("Week by week")
    weeks_desc = sorted(scored["target"].unique(), reverse=True)
    for i, target in enumerate(weeks_desc):
        g = scored[scored["target"] == target]
        target = pd.Timestamp(target)
        actual = g["actual"].dropna()
        released = not actual.empty

        # title shows the actual, or when the report is due
        status = (
            f"actual: {format_bcf_change(actual.iloc[0])}"
            if released
            else f"result due Thursday {short_date(target + pd.Timedelta(days=6))}"
        )
        with st.expander(f"Week ending {short_date(target)} ({status})", expanded=(i == 0)):
            st.dataframe(
                week_table(g, released),
                hide_index=True,
                column_config={
                    "Forecast (Bcf)": st.column_config.Column(help=help_text("Bcf")),
                    "90% range (Bcf)": st.column_config.Column(help=help_text("forecast range (90%)")),
                    "Running MAE (Bcf)": st.column_config.Column(help=help_text("MAE")),
                },
            )
    st.markdown(
        "Error is forecast minus actual, so a positive error means the forecast was too high. "
        f"Negative numbers are {term_tip('withdrawal', 'withdrawals')} and positive numbers are "
        f"{term_tip('injection', 'injections')}.",
        unsafe_allow_html=True,
    )


render()