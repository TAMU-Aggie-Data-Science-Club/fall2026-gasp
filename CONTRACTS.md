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

Models with an interval forecaster (section 7) also get:

| column | type | notes |
|---|---|---|
| `coverage` | float | share of weeks inside the 90% interval; target 0.90 |
| `avg_width` | float | Bcf; average interval width |

These are NaN for models without one.

`evaluate.compare` takes an optional `intervals` dict of name to interval forecaster, using the same names as `forecasters`.

One row per model, sorted by `mae`, best first.

## 7. Interval forecaster

Optional. Written alongside a model's forecaster (section 3) when the model can produce a range. Called by `backtest.py`.

    def forecast_interval(history: pd.DataFrame, target: pd.Timestamp) -> tuple[float, float]

- same `history` and `target` rules as section 3: a storage frame for one series, **only rows before `target`**
- returns `(low, high)`: a **90% interval** for the weekly change in storage (Bcf) for the week ending `target`, i.e. the 5th and 95th percentiles
- `low <= forecast <= high` every week, where `forecast` is the section 3 forecaster's value for the same `history` and `target`. Models whose values can cross (e.g. separate quantile models) sort all three before returning
- section 3 is unchanged. Models without intervals, like the baselines, simply don't have one

Scoring, in `backtest.py`:
- `walk_forward` takes an optional `interval` argument; when given, its output gains `low` and `high` columns
- `coverage(actual, low, high) -> float`: share of weeks where `low <= actual <= high`. A well-calibrated 90% interval scores close to 0.90
- `avg_width(low, high) -> float`: average of `high - low`, in Bcf. Narrower is better at the same coverage

## 8. Saved scoreboard

Written by `evaluate.save_scoreboard` into the SQLite database. Read by the dashboard. Replaced in full every time the scoreboard runs, since backtest results are recomputed rather than point-in-time.

`results` table: the section 6 columns, one row per model.

`predictions` table:

| column | type | notes |
|---|---|---|
| `model` | str | same names as `results` |
| `period` | datetime | the target week |
| `actual` | float | Bcf |
| `predicted` | float | Bcf |
| `low` | float | NaN for models without an interval forecaster |
| `high` | float | NaN for models without an interval forecaster |

One row per model per tested week. Sorted by `model`, then `period`.

## 9. Live forecasts

Written by `live.py` each week before the EIA release (Thursday 10:30 ET). Read by the dashboard.

`live_forecasts` table:

| column | type | notes |
|---|---|---|
| `model` | str | same names as the scoreboard |
| `target` | datetime | the Friday ending the week the upcoming report covers |
| `predicted` | float | Bcf |
| `low` | float | NaN for models without an interval forecaster |
| `high` | float | NaN for models without an interval forecaster |
| `made_at` | str | UTC ISO timestamp; always before that report's release |
| `actual` | float | NaN until the report is released, then filled in |

One row per model per target. A row is never changed after it's written, except to fill `actual`. A forecast is never made or replaced after its report is released.

## 10. Explanations

Written by `live.py` for each LightGBM live forecast. Read by the dashboard.

`live_explanations` table:

| column | type | notes |
|---|---|---|
| `target` | datetime | same as section 9 |
| `feature` | str | a name from `FEATURE_COLUMNS` |
| `label` | str | plain-English name for the dashboard, e.g. "Gas-weighted HDD vs normal" |
| `value` | float | the feature's value that week |
| `contribution` | float | Bcf this feature added to or subtracted from the forecast |
| `base_value` | float | the model's average forecast before any feature; same on every row for a target |

`base_value` plus the sum of `contribution` equals LightGBM's `predicted` for that target in section 9.

## 11. What-if grid

Written by `live.py` for the current target. Read by the dashboard's slider.

`whatif` table:

| column | type | notes |
|---|---|---|
| `model` | str | models that use weather |
| `target` | datetime | same as section 9 |
| `weather_change_pct` | int | from -30 to 30 in steps of 5; the target week's HDD and CDD are scaled by this percent, and each `_dev` column is recomputed as the scaled value minus its normal |
| `predicted` | float | Bcf |
| `low` | float | NaN for models without an interval forecaster |
| `high` | float | NaN for models without an interval forecaster |

`weather_change_pct = 0` matches that model's row in section 9.

## 12. Winter outlook

Written by `live.py` each week. Read by the dashboard.

`winter_outlook` table:

| column | type | notes |
|---|---|---|
| `scenario` | int | the past winter whose weather is replayed, e.g. 2014 means weather from the 2013–14 winter |
| `period` | datetime | each future Friday from the latest stored week through the last Friday of March |
| `storage` | float | projected working gas in storage, Bcf |

Projected with SARIMAX, one week at a time: each future week uses the same ISO week's actual weather from the scenario's winter, and each projected change is added to the previous week's storage. Every scenario starts from the latest actual storage. One row per scenario per week.

## 13. Explanation history

Written by `explain.save_explanation_history`, run from `scoreboard.py` each week. Read by the dashboard's Explain page. Replaced in full every run, like the scoreboard (section 8), since it is recomputed from the backtest period.

`explanation_history` table:

| column | type | notes |
|---|---|---|
| `month` | int | calendar month, 1 to 12 |
| `feature` | str | a name from `FEATURE_COLUMNS` |
| `label` | str | the feature's plain-English name from `FEATURE_LABELS` |
| `mean_abs_contribution` | float | Bcf; average size of this feature's contribution in that month, ignoring direction. How much it matters |
| `mean_contribution` | float | Bcf; average contribution with direction. Negative pushes toward withdrawal, positive toward injection |
| `start` | datetime | first target week in the range (same as the scoreboard's) |
| `end` | datetime | last target week in the range |

One row per month per feature. Sorted by `month`, then `mean_abs_contribution`, largest first.