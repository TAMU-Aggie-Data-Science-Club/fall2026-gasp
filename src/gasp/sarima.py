
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def make_fourier(n, start=0, period=365.25 / 7, K=3):
    """Create seasonal sine and cosine features."""
    t = np.arange(start, start + n)
    features = {}

    for k in range(1, K + 1):
        features[f"sin_{k}"] = np.sin(
            2 * np.pi * k * t / period
        )
        features[f"cos_{k}"] = np.cos(
            2 * np.pi * k * t / period
        )

    return pd.DataFrame(features)


def make_sarima(order=(1, 0, 0), refit_every=4, use_fourier=True):
    """Create a forecaster for weekly storage changes."""

    if refit_every < 1:
        raise ValueError("refit_every must be at least 1")

    fitted_result = None
    previous_history = None
    forecasts_since_refit = 0

    def forecaster(history, target):
        nonlocal fitted_result
        nonlocal previous_history
        nonlocal forecasts_since_refit

        # Convert storage levels into weekly changes
        y = (
            history.sort_values("period")["value"]
            .diff()
            .dropna()
            .to_numpy()
        )

        if len(y) < 10:
            raise ValueError("Not enough historical data")

        # Create Fourier seasonal features
        if use_fourier:
            X = make_fourier(len(y)).to_numpy()
            X_next = make_fourier(
                1, start=len(y)
            ).to_numpy()
        else:
            X = None
            X_next = None

        # Check whether one new observation was added
        can_update = (
            previous_history is not None
            and len(y) == len(previous_history) + 1
            and np.array_equal(y[:-1], previous_history)
        )

        should_refit = (
            fitted_result is None
            or not can_update
            or forecasts_since_refit >= refit_every
        )

        if should_refit:
            model = SARIMAX(
                y,
                exog=X,
                order=order,
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            fitted_result = model.fit(disp=False)
            forecasts_since_refit = 0

        else:
            # Update observations without refitting
            new_y = y[-1:]
            new_X = X[-1:] if use_fourier else None

            fitted_result = fitted_result.append(
                new_y,
                exog=new_X,
                refit=False
            )

        # Forecast next week's storage change
        forecast = fitted_result.forecast(
            steps=1,
            exog=X_next
        )

        previous_history = y.copy()
        forecasts_since_refit += 1

        return float(np.asarray(forecast)[0])

    return forecaster
