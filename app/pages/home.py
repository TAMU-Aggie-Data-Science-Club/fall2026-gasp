from __future__ import annotations

import sqlite3
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st

from gasp import baselines, db, evaluate
from gasp.pipeline import LOWER_48_SERIES
from app.components import (
    format_bcf_change,
    format_model_name,
    render_empty_state,
    render_last_updated,
    term_tip,
)

# A target week ends on Friday. Its forecast runs Sunday afternoon and is
# shown on this page from Sunday 5:30 PM Central, two days after the week ends.
FORECAST_DAYS_AFTER_WEEK_END = 2
PUBLISH_TIME = time(17, 30)
PUBLISH_TZ = ZoneInfo("America/Chicago")


def publish_time(target: pd.Timestamp) -> datetime:
    """When the forecast for a target week appears on the page."""
    day = (target + pd.Timedelta(days=FORECAST_DAYS_AFTER_WEEK_END)).date()
    return datetime.combine(day, PUBLISH_TIME, tzinfo=PUBLISH_TZ)


def fetch_live_forecasts(path: str | Path = db.DB_PATH) -> pd.DataFrame:
    """Load the live forecasts, or an empty table if they don't exist yet."""
    try:
        from gasp import live  # type: ignore
        if hasattr(live, "load_live_forecasts"):
            return live.load_live_forecasts(path=path)
    except Exception:
        pass

    try:
        with sqlite3.connect(path) as conn:
            return pd.read_sql_query("SELECT * FROM live_forecasts", conn)
    except Exception:
        return pd.DataFrame()


def fetch_results(path: str | Path = db.DB_PATH) -> pd.DataFrame:
    """Load the saved scoreboard, or an empty table if the scoreboard hasn't been run yet."""
    try:
        return evaluate.load_results(path=path)
    except Exception:
        return pd.DataFrame()


def short_date(ts: pd.Timestamp) -> str:
    """'Nov 6' style date."""
    return f"{ts.strftime('%b')} {ts.day}"


def long_date(ts: pd.Timestamp) -> str:
    """'Thursday, Nov 12, 2026' style date."""
    return f"{ts.strftime('%A, %b')} {ts.day}, {ts.year}"


def format_range(low: float | None, high: float | None) -> str:
    """' (likely 62 to 110)' for a range, or '' if there isn't one."""
    if low is None or high is None or pd.isna(low) or pd.isna(high):
        return ""
    # Both negative (a withdrawal range): show sizes from smaller to larger
    if low <= 0 and high <= 0:
        return f" (likely {round(abs(float(high)))} to {round(abs(float(low)))})"
    # Both positive (an injection range): keep normal order
    if low >= 0 and high >= 0:
        return f" (likely {round(float(low))} to {round(float(high))})"
    # The range crosses zero: show explicit signs
    return f" (likely {round(float(low)):+d} to {round(float(high)):+d} Bcf)"


def format_headline_forecast(
    predicted: float,
    target_period: pd.Timestamp,
    low: float | None = None,
    high: float | None = None,
) -> str:
    """'We expect a withdrawal of 85 Bcf for the week ending Nov 6 (likely 62 to 110)'"""
    direction = "withdrawal" if predicted < 0 else "injection"
    amt = round(abs(float(predicted)))
    return (
        f"We expect a {direction} of {amt} Bcf for the week ending "
        f"{short_date(target_period)}{format_range(low, high)}"
    )


def rank_rows(rows: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    """Order a week's model rows from best to worst backtest MAE."""
    if not results_df.empty and "model" in results_df.columns and "mae" in results_df.columns:
        mae_map = dict(zip(results_df["model"], results_df["mae"]))
        rows = rows.assign(sort_mae=rows["model"].map(mae_map).fillna(999.0))
        return rows.sort_values("sort_mae").reset_index(drop=True)
    return rows.reset_index(drop=True)


def value_or_none(row: pd.Series, column: str) -> float | None:
    return float(row[column]) if column in row and pd.notna(row[column]) else None


def five_year_sentence(
    storage_df: pd.DataFrame, week_forecasts: pd.DataFrame, target: pd.Timestamp, predicted: float | None
) -> str:
    """The 5-year average for the week, plus how our forecast compares when there is one."""
    five_yr_row = week_forecasts[week_forecasts["model"] == "five_year_average"]
    if not five_yr_row.empty:
        five_yr_val = float(five_yr_row.iloc[0]["predicted"])
    elif not storage_df.empty:
        try:
            five_yr_val = float(baselines.five_year_average(storage_df[storage_df["period"] < target], target))
        except Exception:
            return ""
    else:
        return ""

    sentence = f"The 5-year average for this week is a **{format_bcf_change(five_yr_val)}**"
    if predicted is None:
        return sentence + "."
    diff = predicted - five_yr_val
    return sentence + f"; our forecast is **{abs(diff):.0f} Bcf {'higher' if diff > 0 else 'lower'}**."


def render_forecast_card(
    week_forecasts: pd.DataFrame, results_df: pd.DataFrame, storage_df: pd.DataFrame, target: pd.Timestamp
) -> None:
    """The main box while a forecast is out and its report hasn't been released."""
    lead = rank_rows(week_forecasts, results_df).iloc[0]
    predicted = float(lead["predicted"])
    report_date = target + pd.Timedelta(days=6)

    with st.container(border=True):
        st.caption(
            f"LEAD LIVE FORECAST · {format_model_name(str(lead['model'])).upper()} "
            f"· EIA REPORT DATE: {long_date(report_date).upper()}"
        )
        st.subheader(
            format_headline_forecast(predicted, target, value_or_none(lead, "low"), value_or_none(lead, "high"))
        )
        st.markdown(
            f"**EIA Release Schedule:** Official data for the week ending **{target.strftime('%b %d, %Y')}** "
            f"will be published on **{long_date(report_date)}**."
        )
        sentence = five_year_sentence(storage_df, week_forecasts, target, predicted)
        if sentence:
            st.markdown(sentence)


def render_next_forecast_card(storage_df: pd.DataFrame, next_target: pd.Timestamp) -> None:
    """The main box while the forecast for next_target isn't published yet."""
    published_at = publish_time(next_target)
    report_date = next_target + pd.Timedelta(days=6)

    with st.container(border=True):
        st.caption(f"NEXT LIVE FORECAST · EIA REPORT DATE: {long_date(report_date).upper()}")
        if datetime.now(PUBLISH_TZ) < published_at:
            st.subheader(
                f"Our next forecast, for the week ending {short_date(next_target)}, "
                f"comes out {long_date(pd.Timestamp(published_at.date()))} at 5:30 PM CT"
            )
        else:
            st.subheader(
                f"Our forecast for the week ending {short_date(next_target)} "
                "is running late and will appear here shortly"
            )
        st.markdown(
            f"**EIA Release Schedule:** Official data for the week ending **{next_target.strftime('%b %d, %Y')}** "
            f"will be published on **{long_date(report_date)}**."
        )
        sentence = five_year_sentence(storage_df, pd.DataFrame(columns=["model"]), next_target, None)
        if sentence:
            st.markdown(sentence)


def render_last_result(week_rows: pd.DataFrame, results_df: pd.DataFrame, target: pd.Timestamp) -> None:
    """How our forecast for the most recently reported week compared to EIA's actual."""
    rows = rank_rows(week_rows, results_df)
    lead = rows.iloc[0]
    actual = float(lead["actual"])
    predicted = float(lead["predicted"])
    low, high = value_or_none(lead, "low"), value_or_none(lead, "high")

    in_range = ""
    if low is not None and high is not None:
        in_range = ", inside our range" if low <= actual <= high else ", outside our range"

    st.subheader(f"Last Week's Result: Week Ending {short_date(target)}")
    st.markdown(
        f"We forecast a **{format_bcf_change(predicted)}**{format_range(low, high)}. "
        f"EIA reported a **{format_bcf_change(actual)}**, so we were off by "
        f"**{abs(predicted - actual):.0f} Bcf**{in_range}."
    )

    cols = st.columns(len(rows))
    for col, (_, row) in zip(cols, rows.iterrows()):
        with col:
            st.caption(format_model_name(str(row["model"])))
            st.markdown(f"Forecast: **{format_bcf_change(float(row['predicted']))}**")
            st.caption(f"Off by {abs(float(row['predicted']) - actual):.0f} Bcf")


def render_side_by_side(week_forecasts: pd.DataFrame, results_df: pd.DataFrame, target: pd.Timestamp) -> None:
    """Every live model's forecast for the current week, kept low-key below the main box."""
    st.markdown("#### All Live Model Forecasts")
    st.markdown(
        f"Each column shows one model's {term_tip('live forecast')} for the week ending "
        f"**{target.strftime('%b %d, %Y')}**, ordered from most accurate to least accurate by historical {term_tip('MAE')}.",
        unsafe_allow_html=True,
    )

    rows = rank_rows(week_forecasts, results_df)
    cols = st.columns(min(len(rows), 4))
    for idx, (_, row) in enumerate(rows.iterrows()):
        with cols[idx % len(cols)]:
            st.caption(format_model_name(str(row["model"])))
            st.markdown(f"**{format_bcf_change(float(row['predicted']))}**")
            low, high = value_or_none(row, "low"), value_or_none(row, "high")
            if low is not None and high is not None:
                st.caption(f"90% range: {low:.0f} to {high:.0f} Bcf")
            else:
                st.caption("Point forecast only")


def render_home(path: str | Path = db.DB_PATH) -> None:
    st.title("U.S. Natural Gas Storage Forecast")
    st.markdown(
        f"Weekly forecasts of {term_tip('Lower 48')} {term_tip('working gas in storage')} changes ahead of the Thursday "
        f"{term_tip('EIA weekly storage report')}.",
        unsafe_allow_html=True,
    )

    with st.spinner("Checking latest forecasts and storage data..."):
        live_df = fetch_live_forecasts(path=path)
        results_df = fetch_results(path=path)
        storage_df = db.read_series(LOWER_48_SERIES, path=path)
        weather_df = db.read_degree_days(path=path)

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

    if live_df.empty:
        render_empty_state("The first live forecast runs Wednesday 11/11.")
        return

    live_df = live_df.copy()
    live_df["target"] = pd.to_datetime(live_df["target"])

    latest_target = live_df["target"].max()
    latest_rows = live_df[live_df["target"] == latest_target]
    latest_released = latest_rows["actual"].notna().any()

    # The latest forecast is shown from its publish time until its report comes out
    forecast_visible = not latest_released and datetime.now(PUBLISH_TZ) >= publish_time(latest_target)

    # Main box: the current forecast while it's shown, otherwise when the next one comes out
    if forecast_visible:
        render_forecast_card(latest_rows, results_df, storage_df, latest_target)
    elif latest_released:
        render_next_forecast_card(storage_df, latest_target + pd.Timedelta(days=7))
    else:
        render_next_forecast_card(storage_df, latest_target)

    # Last week's result: the most recent week EIA has reported on
    scored = live_df[live_df["actual"].notna()]
    if not scored.empty:
        result_target = scored["target"].max()
        render_last_result(scored[scored["target"] == result_target], results_df, result_target)

    # Every model's current forecast, only while a forecast is shown
    if forecast_visible:
        render_side_by_side(latest_rows, results_df, latest_target)


render_home()