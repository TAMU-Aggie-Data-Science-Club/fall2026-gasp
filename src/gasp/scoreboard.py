"""Refresh the saved scoreboard with python -m gasp.scoreboard."""

from gasp import baselines, db, evaluate, pipeline, sarima, weather_features
from gasp.sarimax import make_sarimax


# Build the 2021-2025 scoreboard from stored storage and weather data, save the
# available models' scores and weekly predictions, and print the saved summary.
def main() -> None:
    """Backtest the available models and save their results to the database."""
    # Keep the earlier history for training; the dates below select target weeks.
    path = db.DB_PATH
    storage = db.read_series(pipeline.LOWER_48_SERIES, path=path)
    weekly_weather = db.read_degree_days(region="US", path=path)

    if storage.empty or weekly_weather.empty:
        raise ValueError(
            "Load storage and weather with pipeline.backfill_storage and "
            "pipeline.refresh_weather before running the scoreboard."
        )
    storage_with_weather = weather_features.add_weather(storage, weekly_weather)

    # Use weather deviations for SARIMAX
    sarimax_weather = storage_with_weather.dropna(
        subset=["hdd_utility_gas_dev", "cdd_population_dev"]
    ).copy()

    sarimax_forecaster, sarimax_interval = make_sarimax(
        weather=sarimax_weather,
        columns=("hdd_utility_gas_dev", "cdd_population_dev"),
        order=(1, 0, 0),
        refit_every=4,
        use_fourier=True,
    )

    # Register all models and their prediction intervals.
    forecasters = {
        "last_change": baselines.last_change,
        "same_week_last_year": baselines.same_week_last_year,
        "five_year_average": baselines.five_year_average,
        "sarima": sarima.make_sarima(),
        "sarimax": sarimax_forecaster,
    }

    intervals = {
        "sarimax": sarimax_interval,
    }

    # Saving performs one backtest per model and writes both scoreboard tables.
    evaluate.save_scoreboard(
        storage_with_weather,
        forecasters,
        start="2021-01-01",
        end="2025-12-31",
        intervals=intervals,
        path=path,
    )
    print(f"Saved the 2021-2025 scoreboard to {path}")
    print(evaluate.load_results(path=path).to_string(index=False))


if __name__ == "__main__":
    main()
