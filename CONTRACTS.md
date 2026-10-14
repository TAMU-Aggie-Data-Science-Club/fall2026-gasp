# Contracts

Shapes that modules pass between each other. Build to these exactly.

## 1. Storage frame

Produced by `eia.py`, stored by `db.py`, checked by `validate.py`, used by `features.py`, `baselines.py`, `backtest.py`.

| column | type | notes |
|---|---|---|
| `period` | datetime | the Friday that ends the EIA reporting week |
| `series` | str | EIA series ID, e.g. `NW2_EPG0_SWO_R48_BCF` |
| `value` | float | working gas in storage, Bcf |

Sorted by `period`, oldest first.

## 2. Degree-day frame

Produced by `weather_cpc.py` and `weather_noaa.py`.

| column | type | notes |
|---|---|---|
| `week_end` | datetime | last day of the source's week |
| `region` | str | `"US"` for national |
| `measure` | str | `"hdd"` or `"cdd"` |
| `value` | float | degree days for that week |

## 3. Forecaster

Written in `baselines.py` (and every model after), called by `backtest.py`.

    def forecast(history: pd.DataFrame, target: pd.Timestamp) -> float

- `history` is a storage frame for one series, containing **only rows before `target`**
- returns the predicted weekly change in storage (Bcf) for the week ending `target`
- models that need other inputs, like degree days, receive them when the forecaster is created, not through this signature. The signature never changes, so `backtest.py` works with every model

## 4. Daily temperature frame

Input to `weather_noaa.py`. Whatever fetches daily temperatures later must return this shape.

| column | type | notes |
|---|---|---|
| `date` | datetime | one row per city per day |
| `city` | str | city name, e.g. `"Chicago"` |
| `tavg` | float | daily average temperature, °F |