"""Refresh the saved scoreboard with python -m gasp.scoreboard."""

from gasp import baselines, db, evaluate, pipeline, sarima, weather_features


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

    # Add SARIMAX and its interval function after Piece 1 merges.
    forecasters = {
        "last_change": baselines.last_change,
        "same_week_last_year": baselines.same_week_last_year,
        "five_year_average": baselines.five_year_average,
        "sarima": sarima.make_sarima(),
    }

    # Saving performs one backtest per model and writes both scoreboard tables.
    evaluate.save_scoreboard(
        storage_with_weather,
        forecasters,
        start="2021-01-01",
        end="2025-12-31",
        path=path,
    )
    print(f"Saved the 2021-2025 scoreboard to {path}")
    print(evaluate.load_results(path=path).to_string(index=False))


if __name__ == "__main__":
    main()
