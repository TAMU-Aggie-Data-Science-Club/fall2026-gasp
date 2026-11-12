
from gasp.live import load_live_forecasts, load_explanations

test_path = "data/test.db"

forecasts = load_live_forecasts(path=test_path)
explanations = load_explanations(path=test_path)

print("LIVE FORECASTS:")
print(forecasts.to_string(index=False))

print("\nLIGHTGBM EXPLANATIONS:")
print(explanations.to_string(index=False))

assert list(forecasts.columns) == [
    "model",
    "target",
    "predicted",
    "low",
    "high",
    "made_at",
    "actual",
]

assert list(explanations.columns) == [
    "target",
    "feature",
    "label",
    "value",
    "contribution",
    "base_value",
]

print("\nAll dashboard loading tests passed!")
