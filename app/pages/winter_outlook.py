from __future__ import annotations
import os
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Only import db and features at the top level. We keep `outlook` inside try/except
# blocks below because src/gasp/outlook.py uses Python 3.10+
from gasp import db, features
from gasp.pipeline import LOWER_48_SERIES
from app.components import (
    apply_chart_bounds,
    render_empty_state,
    term_tip,
)


def _active_db_path() -> str:
    """Returns GASP_DB_PATH if set in the terminal, else default db.DB_PATH."""
    return os.environ.get("GASP_DB_PATH") or str(db.DB_PATH)


def format_winter_name(scenario_year: int) -> str:
    """
    Formats a scenario integer like 2001 into its winter season label '2000–01'.
    """
    yr = int(scenario_year)
    return f"{yr - 1}–{yr % 100:02d}"


def load_winter_outlook_df() -> pd.DataFrame:
    """
    Loads the winter_outlook table from the active database.
    Tries outlook.load_outlook() first, then falls back to reading SQLite directly.
    """
    active_path = _active_db_path()
    try:
        from gasp import outlook

        df = outlook.load_outlook(path=active_path)
        if not df.empty:
            return df
    except TypeError:
        try:
            from gasp import outlook

            df = outlook.load_outlook()
            if not df.empty:
                return df
        except Exception:
            pass
    except Exception:
        pass

    try:
        with sqlite3.connect(active_path) as conn:
            return pd.read_sql_query("SELECT * FROM winter_outlook", conn)
    except Exception:
        return pd.DataFrame()


def summarize_winter_outlook(outlook_df: pd.DataFrame) -> dict | str:
    """
    Gets summary stats from outlook.summarize_outlook() when supported, or compute the
    exact end-of-March median, coldest winter, and warmest winter directly from outlook_df.
    """
    try:
        from gasp import outlook

        return outlook.summarize_outlook(outlook_df)
    except Exception:
        return {}


def get_five_year_march_end_level() -> float | None:
    """
    Computes the historical five-year average working gas storage level at the end of March
    so we can place a reference marker on the final March date of the fan chart.
    """
    active_path = _active_db_path()
    try:
        storage_df = db.read_series(LOWER_48_SERIES, path=active_path)
    except TypeError:
        try:
            storage_df = db.read_series(LOWER_48_SERIES)
        except Exception:
            return None
    except Exception:
        return None

    try:
        if storage_df.empty:
            return None
        storage_df = storage_df.sort_values("period").reset_index(drop=True)
        stats_df = features.add_five_year_stats(storage_df)
        stats_df["period"] = pd.to_datetime(stats_df["period"])

        # Looks at historical late-March weeks (March 24–31) to get the end-of-March 5-year mean
        late_march = stats_df[
            (stats_df["period"].dt.month == 3) & (stats_df["period"].dt.day >= 24)
        ].dropna(subset=["five_year_mean"])

        if not late_march.empty:
            return float(late_march.iloc[-1]["five_year_mean"])
    except Exception:
        pass
    return None


st.title("Seasonal Winter Storage Outlook")

with st.spinner("Loading 26-winter replay scenarios..."):
    outlook_df = load_winter_outlook_df()

if outlook_df.empty:
    render_empty_state(
        "No winter outlook scenarios are stored in this database yet. "
        "(Run with GASP_DB_PATH=data/deploy_copy.db to view the live deployed outlook.)"
    )
else:
    outlook_df = outlook_df.copy()
    outlook_df["period"] = pd.to_datetime(outlook_df["period"])
    outlook_df = outlook_df.sort_values(["scenario", "period"]).reset_index(drop=True)

    summary = summarize_winter_outlook(outlook_df)
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        sum_info = summary.iloc[0].to_dict()
    elif isinstance(summary, pd.Series):
        sum_info = summary.to_dict()
    elif isinstance(summary, dict):
        sum_info = summary
    else:
        sum_info = {}

    # Computes the weekly 10th percentile, median, and 90th percentile across all 26 replayed winters
    weekly_stats = (
        outlook_df.groupby("period", as_index=False)["storage"]
        .agg(
            p10=lambda s: s.quantile(0.10),
            median="median",
            p90=lambda s: s.quantile(0.90),
        )
        .sort_values("period")
    )

    # Identify's the final end-of-March week to find the coldest (lowest storage)
    # and warmest (highest storage) replayed winters
    final_period = outlook_df["period"].max()
    end_rows = outlook_df[outlook_df["period"] == final_period].sort_values("storage")

    coldest_scenario = int(sum_info.get("coldest_scenario", end_rows.iloc[0]["scenario"]))
    coldest_storage = float(
        sum_info.get("coldest_storage", sum_info.get("min_storage", end_rows.iloc[0]["storage"]))
    )
    warmest_scenario = int(sum_info.get("warmest_scenario", end_rows.iloc[-1]["scenario"]))
    warmest_storage = float(
        sum_info.get("warmest_storage", sum_info.get("max_storage", end_rows.iloc[-1]["storage"]))
    )
    median_end_storage = float(sum_info.get("median_storage", weekly_stats.iloc[-1]["median"]))

    coldest_label = str(sum_info.get("coldest_label", format_winter_name(coldest_scenario)))
    warmest_label = str(sum_info.get("warmest_label", format_winter_name(warmest_scenario)))

    # Simple 5-second headline summary sentence
    if isinstance(summary, str) and summary.strip():
        st.markdown(f"**{summary}**")
    elif "summary" in sum_info and isinstance(sum_info["summary"], str):
        st.markdown(f"**{sum_info['summary']}**")
    else:
        st.markdown(
            f"**In a typical winter, storage ends March near {round(median_end_storage / 10) * 10:,.0f} {term_tip('Bcf')}; "
            f"in a cold winter like {coldest_label}, near {round(coldest_storage / 10) * 10:,.0f} {term_tip('Bcf')} "
            f"(and in a mild winter like {warmest_label}, near {round(warmest_storage / 10) * 10:,.0f} {term_tip('Bcf')}).**",
            unsafe_allow_html=True,
        )

    # One-sentence explanation directly above the fan chart
    st.markdown(
        f"Each thin gray line replays actual weekly weather from one of the past 26 winters (2000–2026) "
        f"through our {term_tip('SARIMAX')} model starting from current {term_tip('working gas in storage')}.",
        unsafe_allow_html=True,
    )

    fig_fan = go.Figure()

    # LAYER 1: Every scenario as a thin gray line
    scenarios = sorted(outlook_df["scenario"].unique())
    first_gray = True
    for scen in scenarios:
        s_df = outlook_df[outlook_df["scenario"] == scen]
        w_name = format_winter_name(int(scen))
        fig_fan.add_trace(
            go.Scatter(
                x=s_df["period"],
                y=s_df["storage"],
                mode="lines",
                line={"color": "rgba(140, 148, 158, 0.35)", "width": 1.1},
                name="26 Replayed Winters (2000–2026)" if first_gray else w_name,
                showlegend=first_gray,
                hovertemplate=f"Winter {w_name}: %{{y:,.0f}} Bcf<extra></extra>",
            )
        )
        first_gray = False

    # LAYER 2: 10th to 90th percentile shaded band
    fig_fan.add_trace(
        go.Scatter(
            x=weekly_stats["period"],
            y=weekly_stats["p90"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig_fan.add_trace(
        go.Scatter(
            x=weekly_stats["period"],
            y=weekly_stats["p10"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(27, 79, 114, 0.18)",
            name="10th–90th Percentile Band",
            customdata=weekly_stats["p90"],
            hovertemplate="10th–90th Percentile: %{y:,.0f} to %{customdata:,.0f} Bcf<extra></extra>",
        )
    )

    # LAYER 3: Warmest winter highlighted and named
    warm_df = outlook_df[outlook_df["scenario"] == warmest_scenario]
    fig_fan.add_trace(
        go.Scatter(
            x=warm_df["period"],
            y=warm_df["storage"],
            mode="lines",
            line={"color": "#008080", "width": 2.4},
            name=f"Warmest Winter ({warmest_label})",
            hovertemplate=f"Warmest ({warmest_label}): %{{y:,.0f}} Bcf<extra></extra>",
        )
    )

    # LAYER 4: Coldest winter highlighted and named
    cold_df = outlook_df[outlook_df["scenario"] == coldest_scenario]
    fig_fan.add_trace(
        go.Scatter(
            x=cold_df["period"],
            y=cold_df["storage"],
            mode="lines",
            line={"color": "#D95F02", "width": 2.4},
            name=f"Coldest Winter ({coldest_label})",
            hovertemplate=f"Coldest ({coldest_label}): %{{y:,.0f}} Bcf<extra></extra>",
        )
    )

    # LAYER 5: Median across all 26 scenarios as a bold navy line
    fig_fan.add_trace(
        go.Scatter(
            x=weekly_stats["period"],
            y=weekly_stats["median"],
            mode="lines",
            line={"color": "#1B4F72", "width": 3.4},
            name="Median Path",
            hovertemplate="Median Trajectory: %{y:,.0f} Bcf<extra></extra>",
        )
    )

    # LAYER 6: Marker for the five-year average end-of-March storage level
    five_yr_march = sum_info.get("five_year_average_march", get_five_year_march_end_level())
    y_values_list = [outlook_df["storage"]]
    if five_yr_march is not None and pd.notna(five_yr_march):
        five_yr_val = float(five_yr_march)
        y_values_list.append(pd.Series([five_yr_val]))
        fig_fan.add_trace(
            go.Scatter(
                x=[final_period],
                y=[five_yr_val],
                mode="markers+text",
                marker={"color": "#6C3483", "size": 11, "symbol": "diamond"},
                text=[f"  5-Yr Avg ({five_yr_val:,.0f} Bcf)"],
                textposition="middle left",
                name="5-Year Avg End-of-March",
                hovertemplate=f"5-Year Avg End-of-March: {five_yr_val:,.0f} Bcf<extra></extra>",
            )
        )

    all_y = pd.concat(y_values_list, ignore_index=True).dropna()
    apply_chart_bounds(
        fig_fan,
        x_min=outlook_df["period"].min(),
        x_max=outlook_df["period"].max(),
        x_title="Week",
        y_title="Projected Working Gas in Storage (Bcf)",
        y_min=float(all_y.min()),
        y_max=float(all_y.max()),
    )
    fig_fan.update_layout(height=460, hovermode="x unified")
    st.plotly_chart(fig_fan, use_container_width=True)

    # The two required plain-English notes directly beneath the chart
    st.markdown(
        f"1. **Why the median ends below the {term_tip('five-year average and five-year range', 'five-year average')}:** "
        f"Many of the 26 replayed winters (2000–2026) were colder than the five most recent winters, and baseline U.S. natural gas demand is higher today.\n"
        f"2. **What these trajectories capture:** "
        f"These 26 paths show how weather alone changes end-of-winter storage; because supply, exports, and power generation also fluctuate, the real uncertainty range is somewhat wider.",
        unsafe_allow_html=True,
    )
