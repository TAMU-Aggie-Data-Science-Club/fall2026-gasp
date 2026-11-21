# Allow modern type hints to run cleanly on Python 3.9+
from __future__ import annotations

import html

import streamlit as st

# the glossary and the id helper from the shared components toolbox
from app.components import GLOSSARY, term_id

# Page title for the About & Glossary page
st.title("About This Project & Glossary")

# Simple explanation of what the GASP project is
st.subheader("What This Project Is")
st.markdown(
    "**Gas Analysis and Storage Prediction (G.A.S.P.)** is an end-to-end data pipeline and forecasting dashboard "
    "built by the Texas A&M Aggie Data Science Club. Every week, it tracks U.S. Lower 48 underground natural gas "
    "inventories reported by the Energy Information Administration (EIA) alongside temperatures from the "
    "NOAA Climate Prediction Center (CPC) to forecast the upcoming Thursday storage report."
)

# Three-step summary of how the data pipeline, models, and backtest work together
st.subheader("How the Forecast Works")
st.markdown(
    "1. **Automated Data Pipeline:** Each week, our SQLite database ingests the latest EIA working gas storage series "
    "and NOAA Heating and Cooling Degree Days, both as national averages (weighted by gas-heated homes and "
    "population) and for separate regions of the country.\n"
    "2. **Statistical & Machine Learning Models:** We compare simple historical benchmarks (like the 5-year seasonal average) "
    "against time-series models (SARIMA and weather-adjusted SARIMAX) and gradient-boosted decision trees (LightGBM).\n"
    "3. **Walk-Forward Backtesting & Calibrated Ranges:** Every model is evaluated on a strict 2021–2025 walk-forward "
    "backtest where it can only see data available prior to each forecast week, producing both a point forecast and a "
    "calibrated 90% confidence range."
)

# Glossary of every term used on the site
st.subheader("Glossary of Natural Gas & Forecasting Terms")

# Table styles: text wraps onto new lines, so nothing is cut off and nothing scrolls sideways, even on a phone
st.markdown(
    """
    <style>
    table.glossary { width: 100%; border-collapse: collapse; table-layout: fixed; }
    table.glossary th, table.glossary td {
        text-align: left; vertical-align: top; padding: 0.6rem 0.75rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.3); overflow-wrap: anywhere;
    }
    table.glossary th { font-size: 0.85rem; }
    table.glossary td.term { width: 32%; font-weight: 600; }
    table.glossary tr.picked td { background: rgba(27, 79, 114, 0.14); }
    @media (max-width: 640px) {
        table.glossary th, table.glossary td { padding: 0.5rem 0.4rem; font-size: 0.9rem; }
        table.glossary td.term { width: 36%; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create a text box so users can type a keyword (like 'HDD' or 'Bcf') to filter the glossary table live
search_query = st.text_input("Filter glossary terms:", placeholder="e.g., Bcf, withdrawal, HDD, SARIMAX...")

# Keep any term whose name or definition contains the search word (case-insensitive); blank search keeps all
rows = [
    (term, definition)
    for term, definition in GLOSSARY.items()
    if not search_query
    or search_query.lower() in term.lower()
    or search_query.lower() in definition.lower()
]

# A term clicked somewhere else on the site arrives as ?term=..., so show that one first and highlight it
picked = st.query_params.get("term", "")
if picked:
    rows.sort(key=lambda row: term_id(row[0]) != picked)  # sort is stable, so the rest keep their order
    if rows and term_id(rows[0][0]) == picked:
        st.markdown(
            f"Showing **{html.escape(rows[0][0])}** first. "
            '<a href="about" target="_self">Show the glossary in its usual order</a>',
            unsafe_allow_html=True,
        )
    else:
        picked = ""

# Build the table as one line of html (a blank line inside would break how markdown reads it)
if rows:
    body = "".join(
        f'<tr id="{term_id(term)}"{" class=picked" if term_id(term) == picked else ""}>'
        f'<td class="term">{html.escape(term)}</td><td>{html.escape(definition)}</td></tr>'
        for term, definition in rows
    )
    st.markdown(
        f'<table class="glossary"><thead><tr><th>Term</th><th>Definition</th></tr></thead><tbody>{body}</tbody></table>',
        unsafe_allow_html=True,
    )
else:
    st.info("No glossary terms matched your search.")