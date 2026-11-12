import math

def make_ensemble(forecasters, weights=None):
    """Combine multiple forecasters using a weighted average."""

    #Support a list or dictionary of forecasters.
    if isinstance(forecasters, dict):
        names = list(forecasters)
        models = list(forecasters.values())
    else:
        names = None
        models = list(forecasters)

    if not models or not all(callable(model) for model in models):
        raise ValueError("At least one valid forecaster is required")

    #Use equal weights by default.
    if weights is None:
        weights = [1.0] * len(models)
    elif isinstance(weights, dict):
        if names is None or set(weights) != set(names):
            raise ValueError("Weights must match forecaster names")
        weights = [weights[name] for name in names]
    else:
        weights = list(weights)

    if len(weights) != len(models):
        raise ValueError("Each forecaster needs a weight")

    weights = [float(weight) for weight in weights]

    if any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("Weights must be finite and nonnegative")

    total_weight = sum(weights)

    if total_weight <= 0:
        raise ValueError("Total weight must be positive")

    #Predict the weighted average of all models.
    def forecast(history, target):
        predictions = [
            float(model(history, target))
            for model in models
        ]

        return float(
            sum(w * p for w, p in zip(weights, predictions))
            / total_weight
        )

    return forecast
