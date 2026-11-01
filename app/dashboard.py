import streamlit as st
import pandas as pd
import altair as alt
from gasp import db, runlog, features, baselines
from gasp.pipeline import LOWER_48_SERIES

st.title("Natural Gas Storage Dashboard")

# runs st.cache_data so as to not requery the db on every use, re-reading every 10 minutes so new data shows up
@st.cache_data(ttl=600)
def load_storage_data():
    return db.read_series(LOWER_48_SERIES)

@st.cache_data(ttl=600)
def load_weather_data():
    return db.read_degree_days()

@st.cache_data(ttl=600)
def load_run_logs():
    return runlog.last_runs()

# load data
storage_df = load_storage_data()
weather_df = load_weather_data()
runs_df = load_run_logs()

# data freshness
st.header("1. Data Freshness")

col1, col2 = st.columns(2)
with col1:
    #check if the data frame is empty
    if not storage_df.empty:
        #find the most recent date for storage
        newest_storage = storage_df['period'].max()
        st.write(f"**Newest Stored Storage Week:** {newest_storage.date()}")
    else:
        st.warning("No storage data found in database.")
with col2:
    #check if the data frame is empty
    if not weather_df.empty:
         #find the most recent date for storage
        newest_weather = weather_df['week_end'].max()
        st.metric("Newest Weather Week", str(newest_weather.date()))
    else:
        st.warning("No weather data found.")

st.subheader("Recent Pipeline Runs")
st.dataframe(runs_df, use_container_width=True)

# --- 2. Storage vs. Five-Year Range Section ---
st.header("2. Storage vs. Five-Year Range")

#user option interface
timeframe = st.radio("Select Timeframe", ("Last 2 Years", "Full History"), key="tf_radio")

# comptue the five year stats
full_stats_df = features.add_five_year_stats(storage_df.copy())

#apply user choice to the dataframe
if timeframe == "Last 2 Years":
    cutoff_date = full_stats_df['period'].max() - pd.DateOffset(years=2)
    plot_df = full_stats_df[full_stats_df['period'] >= cutoff_date].copy()
else:
    plot_df = full_stats_df.copy()

#build chart
base = alt.Chart(plot_df).encode(
    x=alt.X('period:T', title='Date')
)

#build min max band
band = base.mark_area(opacity=0.25, color='gray').encode(
    y=alt.Y('five_year_min:Q', title='Working Gas Storage (BCF)'),
    y2='five_year_max:Q'
)

#five year mean as a dashed line
mean_line = base.mark_line(color='orange', strokeDash=[4, 4], strokeWidth=2).encode(
    y='five_year_mean:Q'
)

#actual storage as a solid line
actual_line = base.mark_line(color='blue', strokeWidth=2).encode(
    y='value:Q'
)

#combine all the layers and print
chart = (band + mean_line + actual_line).properties(
    height=450,
    title="Lower 48 Storage vs. 5-Year Historical Range"
).interactive()

st.altair_chart(chart, use_container_width=True)

st.header("3. Weekly Change")

#calculate weekly change on full data set
full_change_df = storage_df.copy()
full_change_df['weekly_change'] = full_change_df['value'].diff()

#filter recent changes for the last 2 years
weekly_cutoff = storage_df['period'].max() - pd.DateOffset(years=2)
recent_changes = full_change_df[full_change_df['period'] >= weekly_cutoff].copy()

#five-year average change for each week, the same calculation as the baseline
recent_changes['avg_weekly_change'] = [
    baselines.five_year_average(storage_df[storage_df['period'] < p], p)
    for p in recent_changes['period']
]

#prepare to plot average alongside actual
chart_change_df = recent_changes.set_index('period')[['weekly_change', 'avg_weekly_change']]

st.subheader("Recent Weekly Change vs. 5-Year Average Change")
st.line_chart(chart_change_df)