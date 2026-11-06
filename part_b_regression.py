
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.shipment_prep import FEATURES

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "modelling_table.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD MODELLING DATA
# --------------------------------------------------

print("Loading modelling table...")

df = pd.read_csv(DATA_PATH)

X = df[FEATURES]
y = df["delivery_hours"]

print(f"Total shipments: {len(df)}")
print(f"Number of features: {len(FEATURES)}")


# --------------------------------------------------
# 3. TRAIN / VALIDATION / TEST SPLIT
# --------------------------------------------------

print("\nSplitting dataset...")

# First split: 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# Second split: 50% validation, 50% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42
)

print(f"Training rows:   {len(X_train)}")
print(f"Validation rows: {len(X_val)}")
print(f"Test rows:       {len(X_test)}")


# --------------------------------------------------
# 4. SCALING WITHOUT DATA LEAKAGE
# --------------------------------------------------

print("\nScaling features...")

scaler = StandardScaler()

# Fit ONLY on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using the already-fitted scaler
X_test_scaled = scaler.transform(X_test)

train_scaled_max_abs_mean = np.abs(
    X_train_scaled.mean(axis=0)
).max()

test_scaled_max_abs_mean = np.abs(
    X_test_scaled.mean(axis=0)
).max()


# --------------------------------------------------
# 5. SAVE SPLIT AND SCALING CHECKS
# --------------------------------------------------

checks = pd.DataFrame([
    {"metric": "train_rows", "value": len(X_train)},
    {"metric": "val_rows", "value": len(X_val)},
    {"metric": "test_rows", "value": len(X_test)},
    {
        "metric": "train_scaled_max_abs_mean",
        "value": train_scaled_max_abs_mean
    },
    {
        "metric": "test_scaled_max_abs_mean",
        "value": test_scaled_max_abs_mean
    }
])

checks.to_csv(
    OUTPUT_DIR / "part_b_checks.csv",
    index=False
)

print("\nScaling checks:")
print(checks.to_string(index=False))

print("\nPart B - Step 1 completed.")


# --------------------------------------------------
# 6. MEAN BASELINE
# --------------------------------------------------

print("\nEvaluating mean baseline...")

# Calculate the average delivery time from training data only
baseline_prediction = np.full(
    shape=len(y_val),
    fill_value=y_train.mean()
)

baseline_mae = mean_absolute_error(
    y_val,
    baseline_prediction
)

baseline_mse = mean_squared_error(
    y_val,
    baseline_prediction
)

baseline_rmse = np.sqrt(baseline_mse)

baseline_r2 = r2_score(
    y_val,
    baseline_prediction
)

print(f"Training mean: {y_train.mean():.4f} hours")
print(f"Baseline MAE:  {baseline_mae:.4f}")
print(f"Baseline MSE:  {baseline_mse:.4f}")
print(f"Baseline RMSE: {baseline_rmse:.4f}")
print(f"Baseline R2:   {baseline_r2:.4f}")


# --------------------------------------------------
# 7. DECISION TREE DEPTH DIAGNOSTIC
# --------------------------------------------------

print("\nEvaluating decision tree depths...")

tree_depths = [1, 2, 3, 4, 6, 8, 12, None]

tree_results = []

for depth in tree_depths:

    tree = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)

    train_prediction = tree.predict(X_train)
    val_prediction = tree.predict(X_val)

    train_rmse = np.sqrt(
        mean_squared_error(y_train, train_prediction)
    )

    val_rmse = np.sqrt(
        mean_squared_error(y_val, val_prediction)
    )

    tree_results.append({
        "max_depth": "unlimited" if depth is None else depth,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse
    })

tree_depth_df = pd.DataFrame(tree_results)

tree_depth_df.to_csv(
    OUTPUT_DIR / "part_b_tree_depth.csv",
    index=False
)

print("\nDecision tree depth results:")
print(tree_depth_df.to_string(index=False))

# Select the depth with lowest validation RMSE.
# Python's min() retains the first item in a tie.
best_tree_row = tree_depth_df.loc[
    tree_depth_df["val_rmse"].idxmin()
]

best_tree_depth = best_tree_row["max_depth"]

if best_tree_depth == "unlimited":
    best_tree_depth = None
else:
    best_tree_depth = int(best_tree_depth)

print(f"\nBest decision tree depth: {best_tree_depth}")
print(f"Best validation RMSE: {best_tree_row['val_rmse']:.4f}")
