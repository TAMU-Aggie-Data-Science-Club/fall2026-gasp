from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gasp import db, features
from gasp.pipeline import LOWER_48_SERIES
from app.components import (
    apply_chart_bounds,
    render_empty_state,
    term_tip,
)


def format_winter_name(scenario_year: int) -> str:
    """Formats a scenario integer like 2001 into its winter season label '2000–01'."""
    yr = int(scenario_year)
    return f"{yr - 1}–{yr % 100:02d}"


def load_winter_outlook_df() -> pd.DataFrame:
    """Load the saved winter outlook (CONTRACTS.md, section 12), or an empty table."""
    try:
        from gasp import outlook

        return outlook.load_outlook()
    except Exception:
        return pd.DataFrame()


def get_five_year_march_end_level() -> float | None:
    """
    The historical five-year average storage level at the end of March, for a reference
    marker on the final March date of the fan chart.
    """
    try:
        storage_df = db.read_series(LOWER_48_SERIES)
        if storage_df.empty:
            return None
        storage_df = storage_df.sort_values("period").reset_index(drop=True)
        stats_df = features.add_five_year_stats(storage_df)
        stats_df["period"] = pd.to_datetime(stats_df["period"])

        # Late-March weeks (March 24–31) give the end-of-March 5-year mean
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
    render_empty_state("The winter outlook appears here after the next weekly forecast run.")
else:
    outlook_df = outlook_df.copy()
    outlook_df["period"] = pd.to_datetime(outlook_df["period"])
    outlook_df = outlook_df.sort_values(["scenario", "period"]).reset_index(drop=True)

    # Weekly 10th percentile, median, and 90th percentile across all replayed winters
    weekly_stats = (
        outlook_df.groupby("period", as_index=False)["storage"]
        .agg(
            p10=lambda s: s.quantile(0.10),
            median="median",
            p90=lambda s: s.quantile(0.90),
        )
        .sort_values("period")
    )

    # The final end-of-March week: coldest winter = lowest storage, warmest = highest
    final_period = outlook_df["period"].max()
    end_rows = outlook_df[outlook_df["period"] == final_period].sort_values("storage")

    coldest_scenario = int(end_rows.iloc[0]["scenario"])
    coldest_storage = float(end_rows.iloc[0]["storage"])
    warmest_scenario = int(end_rows.iloc[-1]["scenario"])
    warmest_storage = float(end_rows.iloc[-1]["storage"])
    median_end_storage = float(weekly_stats.iloc[-1]["median"])

    coldest_label = format_winter_name(coldest_scenario)
    warmest_label = format_winter_name(warmest_scenario)
    n_winters = outlook_df["scenario"].nunique()

    # Simple 5-second headline summary sentence
    st.markdown(
        f"**In a typical winter, storage ends March near {round(median_end_storage / 10) * 10:,.0f} {term_tip('Bcf')}; "
        f"in a cold winter like {coldest_label}, near {round(coldest_storage / 10) * 10:,.0f} {term_tip('Bcf')} "
        f"(and in a mild winter like {warmest_label}, near {round(warmest_storage / 10) * 10:,.0f} {term_tip('Bcf')}).**",
        unsafe_allow_html=True,
    )

    # One-sentence explanation directly above the fan chart
    st.markdown(
        f"Each thin gray line replays actual weekly weather from one of the past {n_winters} winters (2000–2026) "
        f"through our {term_tip('SARIMAX')} model starting from current {term_tip('working gas in storage')}.",
        unsafe_allow_html=True,
    )

    fig_fan = go.Figure()

    # LAYER 1: Every scenario as a thin gray line (no hover, so the hover box stays readable)
    scenarios = sorted(outlook_df["scenario"].unique())
    first_gray = True
    for scen in scenarios:
        s_df = outlook_df[outlook_df["scenario"] == scen]
        fig_fan.add_trace(
            go.Scatter(
                x=s_df["period"],
                y=s_df["storage"],
                mode="lines",
                line={"color": "rgba(140, 148, 158, 0.35)", "width": 1.1},
                name=f"{n_winters} Replayed Winters (2000–2026)",
                showlegend=first_gray,
                hoverinfo="skip",
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

    # LAYER 5: Median across all scenarios as a bold navy line
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
    five_yr_march = get_five_year_march_end_level()
    y_values_list = [outlook_df["storage"]]
    if five_yr_march is not None:
        y_values_list.append(pd.Series([five_yr_march]))
        fig_fan.add_trace(
            go.Scatter(
                x=[final_period],
                y=[five_yr_march],
                mode="markers+text",
                marker={"color": "#6C3483", "size": 11, "symbol": "diamond"},
                text=[f"  5-Yr Avg ({five_yr_march:,.0f} Bcf)"],
                textposition="middle left",
                name="5-Year Avg End-of-March",
                hovertemplate=f"5-Year Avg End-of-March: {five_yr_march:,.0f} Bcf<extra></extra>",
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
        f"Many of the replayed winters (2000–2026) were colder than the five most recent winters, and baseline U.S. natural gas demand is higher today.\n"
        f"2. **What these trajectories capture:** "
        f"These paths show how weather alone changes end-of-winter storage; because supply, exports, and power generation also fluctuate, the real uncertainty range is somewhat wider.",
        unsafe_allow_html=True,
    )