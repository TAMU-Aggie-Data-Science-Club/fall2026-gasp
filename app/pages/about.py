# Allow modern type hints to run cleanly on Python 3.9+
from __future__ import annotations

import pandas as pd
import streamlit as st

# Import the complete 24-term dictionary from the shared components toolbox
from app.components import GLOSSARY

# Page title for the About & Glossary page
st.title("About This Project & Glossary")

#Simple explanation of what the GASP project is
st.subheader("What This Project Is")
st.markdown(
    "The **Gas Analytics & Storage Predictor (GASP)** is an end-to-end data pipeline and forecasting dashboard "
    "built by the Texas A&M Aggie Data Science Club. Every week, it tracks U.S. Lower 48 underground natural gas "
    "inventories reported by the Energy Information Administration (EIA) alongside regional temperatures from the "
    "NOAA Climate Prediction Center (CPC) to forecast the upcoming Thursday storage report."
)

#Three-step summary of how the data pipeline, models, and backtest work together
st.subheader("How the Forecast Works")
st.markdown(
    "1. **Automated Data Pipeline:** Each week, our SQLite database ingests the latest EIA working gas storage series "
    "and NOAA regional Heating and Cooling Degree Days (weighted by gas-heated homes and population).\n"
    "2. **Statistical & Machine Learning Models:** We compare simple historical benchmarks (like the 5-year seasonal average) "
    "against time-series models (SARIMA and weather-adjusted SARIMAX) and gradient-boosted decision trees (LightGBM).\n"
    "3. **Walk-Forward Backtesting & Calibrated Ranges:** Every model is evaluated on a strict 2021–2025 walk-forward "
    "backtest where it can only see data available prior to each forecast week, producing both a point forecast and a "
    "calibrated 90% confidence range."
)

#Interactive Glossary table containing all required natural gas and modeling terms
st.subheader("Glossary of Natural Gas & Forecasting Terms")
st.markdown(
    "Written in plain English for readers who have never traded natural gas. "
    "Use the search box below to filter terms quickly."
)

# Create a text box so users can type a keyword (like 'HDD' or 'Bcf') to filter the glossary table live
search_query = st.text_input("Filter glossary terms:", placeholder="e.g., Bcf, withdrawal, HDD, SARIMAX...")

# Build a list of matching rows from GLOSSARY: if search_query is blank, keep all terms;
# otherwise, keep any term whose name or definition contains the search word (case-insensitive).
glossary_rows = [
    {"Term": term, "Plain-English Definition": definition}
    for term, definition in GLOSSARY.items()
    if not search_query
    or search_query.lower() in term.lower()
    or search_query.lower() in definition.lower()
]

# Display the matching terms as a clean full-width table, or show a note if nothing matched the search
if glossary_rows:
    st.dataframe(pd.DataFrame(glossary_rows), use_container_width=True, hide_index=True)
else:
    st.info("No glossary terms matched your search.")
