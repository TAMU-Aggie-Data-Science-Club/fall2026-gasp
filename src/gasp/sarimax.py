"""Weather-augmented weekly storage-change forecasts and 90% intervals.

Reuses the existing SARIMA Fourier feature generator and refit conventions.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from gasp import sarima


def make_sarimax(
    weather,
    columns=("hdd_utility_gas_dev", "cdd_population_dev"),
    order=(1, 0, 0),
    refit_every=4,
    use_fourier=True,
):
    """Return (forecast, forecast_interval), both using the same fitted state."""
    if refit_every < 1:
        raise ValueError("refit_every must be at least 1")
    columns = tuple(columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("columns must be nonempty and unique")
    required = {"period", *columns}
    missing = required.difference(weather.columns)
    if missing:
        raise ValueError(f"Weather frame is missing columns: {sorted(missing)}")

    w = weather.loc[:, ["period", *columns]].copy()
    w["period"] = pd.to_datetime(w["period"])
    if w["period"].isna().any() or w["period"].duplicated().any():
        raise ValueError("Weather periods must be valid and unique")
    w = w.set_index("period").sort_index()

    state = {"fit": None, "periods": (), "changes": np.array([]),
             "exog": None, "since_refit": 0, "cached_key": None,
             "cached_result": None}

    def features(periods, start):
        dates = pd.DatetimeIndex(pd.to_datetime(periods))
        absent = dates.difference(w.index)
        if len(absent):
            raise ValueError(f"Missing weather for week(s): {list(absent.strftime('%Y-%m-%d'))}")
        values = w.loc[dates, list(columns)]
        bad = ~np.isfinite(values.to_numpy(dtype=float)).all(axis=1)
        if bad.any():
            raise ValueError(f"Missing or invalid weather values for week(s): "
                             f"{list(dates[bad].strftime('%Y-%m-%d'))}")
        matrix = values.to_numpy(dtype=float)
        if use_fourier:
            # The existing SARIMA module is the authoritative Fourier implementation.
            seasonal = np.asarray(sarima.make_fourier(len(dates), start=start), dtype=float)
            if seasonal.ndim == 1:
                seasonal = seasonal.reshape(-1, 1)
            if seasonal.shape[0] != len(dates):
                raise ValueError("sarima.make_fourier returned an unexpected row count")
            matrix = np.column_stack((matrix, seasonal))
        return matrix

    def compute(history, target):
        target = pd.Timestamp(target)
        h = history.loc[:, ["period", "value"]].copy()
        h["period"] = pd.to_datetime(h["period"])
        h = h.sort_values("period")
        if h["period"].isna().any() or h["period"].duplicated().any():
            raise ValueError("Storage history periods must be valid and unique")
        if len(h) < 11:
            raise ValueError("Not enough historical data (need at least 10 weekly changes)")
        if (h["period"] >= target).any():
            raise ValueError("History must contain only weeks before target")
        vals = h["value"].to_numpy(dtype=float)
        if not np.isfinite(vals).all():
            raise ValueError("Storage history contains missing or invalid values")
        dates = pd.DatetimeIndex(h["period"].iloc[1:])
        changes = np.diff(vals)

        key = (
            tuple(pd.DatetimeIndex(h["period"]).asi8),
            tuple(vals),
            target.value
        )

        if key == state["cached_key"]:
            return state["cached_result"]

        x = features(dates, start=0)
        target_x = features([target], start=len(changes))

        fit = state["fit"]
        old_dates = state["periods"]
        # Append only a genuinely new contiguous suffix; refit periodically.
        prefix_matches = (fit is not None and len(dates) >= len(old_dates)
                          and tuple(dates[:len(old_dates)]) == old_dates
                          and np.array_equal(changes[:len(old_dates)], state["changes"])
                          and np.array_equal(x[:len(old_dates)], state["exog"]))
        new_count = len(dates) - len(old_dates) if prefix_matches else len(dates)
        if not prefix_matches or state["since_refit"] + new_count >= refit_every:
            fit = SARIMAX(
                changes, exog=x, order=order, trend="c",
                enforce_stationarity=False, enforce_invertibility=False
            ).fit(disp=False, maxiter=200)
            since_refit = 0
        elif new_count:
            fit = fit.append(changes[-new_count:], exog=x[-new_count:], refit=False)
            since_refit = state["since_refit"] + new_count
        else:
            since_refit = state["since_refit"]

        prediction = fit.get_forecast(steps=1, exog=target_x)
        point = float(np.asarray(prediction.predicted_mean).ravel()[0])
        bounds = np.asarray(prediction.conf_int(alpha=0.10), dtype=float).reshape(-1, 2)[0]
        low, high = map(float, bounds)
        if not (np.isfinite([low, point, high]).all() and low <= point <= high):
            raise ValueError("SARIMAX produced invalid 90% prediction bounds")
        state.update(fit=fit, periods=tuple(dates), changes=changes.copy(),
                     exog=x.copy(), since_refit=since_refit,
                     cached_key=key, cached_result=(point, low, high))
        return point, low, high

    def forecaster(history: pd.DataFrame, target: pd.Timestamp) -> float:
        return compute(history, target)[0]

    def forecast_interval(history: pd.DataFrame, target: pd.Timestamp) -> tuple[float, float]:
        _, low, high = compute(history, target)
        return low, high

    return forecaster, forecast_interval