
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from gasp import db, pipeline, feature_table
from gasp.explain import explain
from gasp.lightgbm_model import make_lightgbm
from gasp.live import (
    add_upcoming_week,
    save_live_explanations,
    load_explanations,
)

# Read historical data without refreshing or modifying it.
storage = db.read_series(
    pipeline.LOWER_48_SERIES,
    path=db.DB_PATH,
)

weather = db.read_degree_days(
    region="US",
    path=db.DB_PATH,
)

# Use the next week after the latest stored observation.
target = pd.Timestamp(storage["period"].max()) + pd.Timedelta(days=7)

# Add the unknown target week.
storage_extended = add_upcoming_week(storage)

# Build the same feature table used in live.py.
features = feature_table.build_feature_table(
    storage_extended,
    weather,
)

# Generate the LightGBM forecast.
forecaster = make_lightgbm(
    features,
    refit_every=4,
)

prediction = forecaster(storage, target)

# Calculate the feature contributions.
explanation = explain(features, target)

# Reconstruct the prediction from the contributions.
reconstructed = (
    explanation["base_value"].iloc[0]
    + explanation["contribution"].sum()
)

print("Target:", target.date())
print(f"LightGBM prediction: {prediction:.6f} Bcf")
print(f"Reconstructed prediction: {reconstructed:.6f} Bcf")

print("\nFeature explanations:")
print(
    explanation[
        ["feature", "value", "contribution"]
    ].to_string(index=False)
)

assert len(explanation) == len(feature_table.FEATURE_COLUMNS)

assert np.isclose(
    prediction,
    reconstructed,
    rtol=1e-6,
    atol=1e-6,
)

print("\nAll explanation tests passed!")


# Use a future test target because historical releases cannot
# be saved as new live explanations.
test_explanation = explanation.copy()
test_explanation["target"] = pd.Timestamp("2027-01-08")


with tempfile.TemporaryDirectory() as folder:
    test_path = Path(folder) / "test_explanations.db"

    first_save = save_live_explanations(
        test_explanation,
        path=test_path,
    )

    second_save = save_live_explanations(
        test_explanation,
        path=test_path,
    )

    saved = load_explanations(path=test_path)

    saved_target = saved[
        saved["target"] == pd.Timestamp("2027-01-08")
    ]

    print("\nFirst save:", first_save)
    print("Second save:", second_save)
    print("Saved explanation rows:", len(saved_target))

    assert first_save == len(feature_table.FEATURE_COLUMNS)
    assert second_save == 0
    assert len(saved_target) == len(feature_table.FEATURE_COLUMNS)

    print("\nAll explanation database tests passed!")
