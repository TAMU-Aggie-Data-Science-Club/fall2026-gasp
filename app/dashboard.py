import streamlit as st
import pandas as pd
import altair as alt
from gasp import db, runlog, features, baselines, evaluate
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
#Piece 2: Dashboard, model pages

#Cache scoreboard results and predictions for 10 minutes; an empty table if the scoreboard hasn't been saved yet
@st.cache_data(ttl=600)
def load_scoreboard_results():
    try:
        return evaluate.load_results()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def load_scoreboard_predictions():
    try:
        return evaluate.load_predictions()
    except Exception:
        return pd.DataFrame()

#Fetches both of the cached dataframes for use in the dashboard
results_df = load_scoreboard_results()
predictions_df = load_scoreboard_predictions()

#Explains what MAE, RMSE, and bias measure
st.header("4. Model Scoreboard")
st.caption("**MAE** (Mean Absolute Error) measures average error magnitude in Bcf, "
           "**RMSE** (Root Mean Squared Error) penalizes large misses more heavily, and "
           "**bias** (mean signed error) shows whether a model tends to over- or under-predict." 
)
if results_df.empty:
    st.warning("No saved scoreboard results found. Run 'python -m gasp.scoreboard' first.")
else:
    #Sorts the results by MAE in ascending order and resets the index for display
    sorted_results = results_df.sort_values("mae", ascending=True).reset_index(drop=True)
    st.dataframe(sorted_results, use_container_width=True)

#Actual Vs. Predicted Section
st.header("5. Actual vs. Predicted Weekly Changes")

if predictions_df.empty:
    st.warning("No saved predictions found in database.")
else:
    #Ensures period is a datetime object for Altair time-series plotting and copies predictions 
    pred_df = predictions_df.copy()
    pred_df["period"] = pd.to_datetime(pred_df["period"])

    #If there's scoreboard results then the model's dropdown list will be ordered by MAE 
    if not results_df.empty and "model" in results_df.columns:
        available_models = [
            m for m in results_df.sort_values("mae")["model"].tolist() if m in pred_df["model"].unique()
        ]
    else:
        available_models = sorted(pred_df["model"].unique().tolist())

    #This allows the use to pick one or more models to compare in the chart, defaulting to the best model if available
    selected_models = st.multiselect("Select one or more models to compare:",options=available_models, default=available_models[:1])

    if selected_models:
        #Filters predictions to only that were slected by the user
        filtered_preds = pred_df[pred_df["model"].isin(selected_models)].copy()
        #Extracts a clean series of actual weekly storage changes
        actual_df = (filtered_preds[["period", "actual"]].drop_duplicates(subset=["period"]).sort_values("period")
        )
        #This builds a black line for actual weekly storage changes, with a tooltip for the week and actual value
        actual_line = (alt.Chart(actual_df).mark_line(color="black", strokeWidth=2.5)
                       .encode(
                           x=alt.X("period:T", title="Week"),
                            y=alt.Y("actual:Q", title="Weekly Storage Change (Bcf)"), 
                            tooltip=[
                                alt.Tooltip("period:T", title="Week"),
                                  alt.Tooltip("actual:Q", title="Actual", format=".2f")
                                  ]
                                )
                            )
        #This builds a dashed line for each model's predictions that are selected
        pred_lines = (
            alt.Chart(filtered_preds).mark_line(strokeDash=[4,2], strokeWidth=2).encode(
                x=alt.X("period:T", title="week"),
                y=alt.Y("predicted:Q", title="Weekly Storage Change (Bcf)"),
                color=alt.Color("model:N", title="Model"),
                tooltip=[
                    alt.Tooltip("period:T", title="Week"),
                    alt.Tooltip("model:N", title="Model"),
                    alt.Tooltip("predicted:Q", title="Predicted", format=".2f"),
                    alt.Tooltip("actual:Q", title="Actual", format=".2f"),
                ]
            )
        )
        layers = [actual_line, pred_lines]
        #This will check if the selected model has valid high and low predictions like SARIMAX
        if "low" in filtered_preds.columns and "high" in filtered_preds.columns:
            #Drops the rows where the lows and highes are NAN
            band_df = filtered_preds.dropna(subset=["low", "high"]).copy()
            if not band_df.empty:
                #Shades confidence band between low and high bounds  
                interval_band = (
                    alt.Chart(band_df).mark_area(opacity=.25).encode(
                        x=alt.X("period:T", title="Week"),
                        y=alt.Y("low:Q", title="Weekly Storage Change (Bcf)"),
                        y2="high:Q",
                        color=alt.Color("model:N", title="Model"),
                    )
                )
                layers.insert(0, interval_band)
        #This combines all active layers and displays the interactive chart
        avp_chart = (alt.layer(*layers).properties(height=450, 
                                                   title="Actual vs. Predicted Weekly Storage Changes (Black = Actual)")
                                                   .interactive())
        st.altair_chart(avp_chart, use_container_width=True)
    else:
        st.info("Select at least one model above to view predictions.")

#MAE by Month Section
st.header("6. MAE by Month")

if not predictions_df.empty:
    #Copies predictions and converts period to datetime
    monthly_df = predictions_df.copy()
    monthly_df["period"] = pd.to_datetime(monthly_df["period"])
    
    #Weekly absolute error 
    monthly_df["abs_error"] = (monthly_df["actual"] - monthly_df["predicted"]).abs()

    #Calculates the month number and month name for labels
    monthly_df["month_num"] = monthly_df["period"].dt.month
    monthly_df["month_name"] = monthly_df["period"].dt.strftime("%b")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    #Calculates the absolute errors by model and month to get Monthly MAE
    mae_by_month = (
        monthly_df.groupby(["model", "month_num", "month_name"], as_index=False)["abs_error"]
        .mean().rename(columns={"abs_error": "mae"}).sort_values("month_num")
    )

    month_chart = (
        alt.Chart(mae_by_month).mark_bar().encode(
            x=alt.X("month_name:N", sort=month_order, title="Month"),
            xOffset="model:N",
            y=alt.Y("mae:Q", title="Mean Absolute Error (Bcf)"),
            color=alt.Color("model:N", title="Model"),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("month_name:N", title="Month"),
                alt.Tooltip("mae:Q", title="MAE (Bcf)", format=".2f"),
            ],
        )
        .properties(height=400, title="Model Mean Absolute Error by Month")
    )
    st.altair_chart(month_chart,use_container_width=True)