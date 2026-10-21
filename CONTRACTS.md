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

## 2. Weekly degree-day frame

Produced by `weather_weekly.py` from a daily degree-day frame (section 4).

| column | type | notes |
|---|---|---|
| `week_end` | datetime | the Friday that ends the EIA week (Saturday through Friday) |
| `region` | str | `"US"` for the continental US; Census division number otherwise, e.g. `"3"` |
| `measure` | str | `"hdd"` or `"cdd"` |
| `weighting` | str | `"population"` or `"utility_gas"` |
| `value` | float | degree days summed over the 7 days |

Complete weeks only. Sorted by `week_end`, oldest first.

## 3. Forecaster

Written in `baselines.py` (and every model after), called by `backtest.py`.

    def forecast(history: pd.DataFrame, target: pd.Timestamp) -> float

- `history` is a storage frame for one series, containing **only rows before `target`**
- returns the predicted weekly change in storage (Bcf) for the week ending `target`
- models that need other inputs, like degree days, receive them when the forecaster is created, not through this signature. The signature never changes, so `backtest.py` works with every model

## 4. Daily degree-day frame

Produced by `weather_cpc.py`, input to `weather_weekly.py`.

| column | type | notes |
|---|---|---|
| `date` | datetime | one row per region, measure, and weighting per day |
| `region` | str | same as section 2 |
| `measure` | str | same as section 2 |
| `weighting` | str | same as section 2 |
| `value` | float | degree days for that day |

Sorted by `date`, oldest first.

## 5. Weather columns

Added to a storage frame by `weather_features.py`. Used by SARIMAX, LightGBM, and every model after.

| column | type | notes |
|---|---|---|
| `hdd_utility_gas` | float | weekly HDD, weighted by gas-heating households, US |
| `hdd_population` | float | weekly HDD, population weighted, US |
| `cdd_population` | float | weekly CDD, population weighted, US |
| `hdd_utility_gas_dev` | float | `hdd_utility_gas` minus its normal |
| `hdd_population_dev` | float | `hdd_population` minus its normal |
| `cdd_population_dev` | float | `cdd_population` minus its normal |

Joined on `period` = `week_end`. The storage frame's own columns are unchanged. A week with no weather data gets NaN; it is never dropped.

Normals come from `weather_weekly.add_normal`. ISO week 53 rarely has enough prior years for a normal, so week 53 rows use week 52's normal instead of NaN.

## 6. Results table

Produced by `evaluate.py`. Used by the dashboard.

| column | type | notes |
|---|---|---|
| `model` | str | name of the forecaster, e.g. `"five_year_average"` |
| `mae` | float | Bcf |
| `rmse` | float | Bcf |
| `bias` | float | Bcf; average of predicted minus actual, so positive means forecasts run too high |
| `n_weeks` | int | number of weeks tested |
| `start` | datetime | first target week |
| `end` | datetime | last target week |

One row per model, sorted by `mae`, best first.