from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "modelling_table.csv"
CONFIG_PATH = PROJECT_ROOT / "outputs" / "part_c_final_config.csv"
MEDIANS_PATH = PROJECT_ROOT / "models" / "imputation_medians.json"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Keep this feature order exactly aligned with Part A and Part C.
FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]


# ============================================================
# STEP 1: LOAD DATA AND FINAL CONFIGURATION
# ============================================================

print("Loading modelling data and final configuration...")

df = pd.read_csv(DATA_PATH)
config_df = pd.read_csv(CONFIG_PATH)

if len(config_df) != 1:
    raise ValueError("Expected exactly one row in part_c_final_config.csv.")

config = config_df.iloc[0]

selector_k = int(config["selector_k"])
n_estimators = int(config["n_estimators"])
max_depth_value = config["max_depth"]
min_samples_leaf = int(config["min_samples_leaf"])
threshold = float(config["threshold"])

# Convert the CSV representation "unlimited" back to None.
max_depth = (
    None
    if str(max_depth_value).strip().lower() == "unlimited"
    else int(max_depth_value)
)

print("Selected configuration:")
print(config.to_string())


# ============================================================
# STEP 2: REBUILD PART C STRATIFIED SPLIT
# ============================================================

X = df[FEATURES]
y = df["is_delayed"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

print("\nSplit sizes:")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))


# ============================================================
# STEP 3: REBUILD AND TRAIN FINAL PIPELINE
# ============================================================

final_model = Pipeline([
    (
        "selector",
        SelectKBest(
            score_func=f_classif,
            k=selector_k
        )
    ),
    (
        "model",
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
    )
])

print("\nTraining final classification pipeline...")

final_model.fit(X_train, y_train)

print("Model training completed.")


# ============================================================
# STEP 4: SAVE MODEL AND SUPPORTING FILES
# ============================================================

model_path = MODELS_DIR / "delay_model.pkl"

joblib.dump(final_model, model_path)

# Threshold file
pd.DataFrame({
    "threshold": [threshold]
}).to_csv(
    MODELS_DIR / "delay_threshold.csv",
    index=False
)

# Feature order file
pd.DataFrame({
    "feature": FEATURES
}).to_csv(
    MODELS_DIR / "feature_order.csv",
    index=False
)

# Imputation medians
import json

with open(MEDIANS_PATH, "r", encoding="utf-8") as file:
    medians = json.load(file)

impute_values = pd.DataFrame([
    {"column": column, "median": median}
    for column, median in medians.items()
])

impute_values.to_csv(
    MODELS_DIR / "impute_values.csv",
    index=False
)

print("\nSaved model artefacts:")
print(model_path)
print(MODELS_DIR / "delay_threshold.csv")
print(MODELS_DIR / "feature_order.csv")
print(MODELS_DIR / "impute_values.csv")


# ============================================================
# STEP 5: RELOAD MODEL AND VERIFY PROBABILITIES
# ============================================================

print("\nReloading saved model...")

reloaded_model = joblib.load(model_path)

original_probabilities = final_model.predict_proba(X_test)[:, 1]
reloaded_probabilities = reloaded_model.predict_proba(X_test)[:, 1]

max_probability_diff = np.max(
    np.abs(original_probabilities - reloaded_probabilities)
)

reload_check = pd.DataFrame([
    {
        "metric": "test_rows_scored",
        "value": len(X_test)
    },
    {
        "metric": "reload_max_abs_probability_diff",
        "value": max_probability_diff
    },
    {
        "metric": "threshold",
        "value": threshold
    }
])

reload_check.to_csv(
    OUTPUT_DIR / "part_d_reload_check.csv",
    index=False
)

print("\nReload verification:")
print(reload_check.to_string(index=False))

assert len(X_test) == 180
assert max_probability_diff <= 1e-9

print("\nPart D - Step 5 completed successfully.")
