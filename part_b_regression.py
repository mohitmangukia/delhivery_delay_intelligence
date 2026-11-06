
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from src.shipment_prep import FEATURES

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


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



# --------------------------------------------------
# 8. FOUR CANDIDATE MODELS
# --------------------------------------------------

print("\nTraining candidate regression models...")

models = {
    "linear_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "decision_tree": DecisionTreeRegressor(
        max_depth=best_tree_depth,
        random_state=42
    ),

    "random_forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
}


# --------------------------------------------------
# 9. VALIDATION METRICS
# --------------------------------------------------

def calculate_metrics(y_true, predictions):
    mse = mean_squared_error(y_true, predictions)

    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, predictions)
    }


metrics_results = []

# Add the mean baseline result
metrics_results.append({
    "model": "mean_baseline",
    "split": "val",
    **calculate_metrics(y_val, baseline_prediction)
})

# Fit and evaluate each candidate
for model_name, model in models.items():

    print(f"Training {model_name}...")

    model.fit(X_train, y_train)

    val_prediction = model.predict(X_val)

    metrics = calculate_metrics(y_val, val_prediction)

    metrics_results.append({
        "model": model_name,
        "split": "val",
        **metrics
    })


metrics_df = pd.DataFrame(
    metrics_results,
    columns=["model", "split", "MAE", "MSE", "RMSE", "R2"]
)

metrics_df.to_csv(
    OUTPUT_DIR / "part_b_metrics.csv",
    index=False
)

print("\nValidation model comparison:")
print(metrics_df.to_string(index=False))


# --------------------------------------------------
# 10. GRADIENT BOOSTING LEARNING-RATE EXPERIMENT
# --------------------------------------------------

print("\nEvaluating Gradient Boosting learning rates...")

learning_rates = [0.01, 0.1, 1.0]

learning_rate_results = []

for rate in learning_rates:

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=rate,
        random_state=42
    )

    model.fit(X_train, y_train)

    val_prediction = model.predict(X_val)

    learning_rate_results.append({
        "learning_rate": rate,
        "val_rmse": np.sqrt(
            mean_squared_error(y_val, val_prediction)
        ),
        "val_r2": r2_score(y_val, val_prediction)
    })


learning_rate_df = pd.DataFrame(learning_rate_results)

learning_rate_df.to_csv(
    OUTPUT_DIR / "part_b_gb_learning_rate.csv",
    index=False
)

print("\nGradient Boosting learning-rate comparison:")
print(learning_rate_df.to_string(index=False))

# --------------------------------------------------
# 11. CROSS-VALIDATION OF TOP TWO MODELS
# --------------------------------------------------

print("\nRunning 5-fold cross-validation...")

# Rank candidate models using validation RMSE.
# Exclude the mean baseline.
candidate_ranking = (
    metrics_df[metrics_df["model"] != "mean_baseline"]
    .sort_values("RMSE")
)

top_two_models = candidate_ranking["model"].head(2).tolist()

print(f"Top two models: {top_two_models}")

# Combine training and validation data.
# Keep the test set completely separate.
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

# Use the same KFold configuration for both models.
kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = []

for model_name in top_two_models:

    model = models[model_name]

    print(f"Cross-validating {model_name}...")

    fold_scores = cross_val_score(
        model,
        X_train_val,
        y_train_val,
        cv=kfold
    )

    cv_results.append({
        "model": model_name,
        "mean_r2": fold_scores.mean(),
        "std_r2": fold_scores.std()
    })

cv_df = pd.DataFrame(cv_results)

cv_df.to_csv(
    OUTPUT_DIR / "part_b_cv.csv",
    index=False
)

print("\nCross-validation results:")
print(cv_df.to_string(index=False))


# --------------------------------------------------
# 12. FINAL MODEL SELECTION AND TEST EVALUATION
# --------------------------------------------------

print("\nSelecting final model...")

# Select the candidate with the lowest validation RMSE.
final_model_name = (
    metrics_df[metrics_df["model"] != "mean_baseline"]
    .sort_values("RMSE")
    .iloc[0]["model"]
)

final_model = models[final_model_name]

print(f"Final selected model: {final_model_name}")

# Evaluate the already-trained model on the untouched test set.
test_prediction = final_model.predict(X_test)

test_metrics = calculate_metrics(y_test, test_prediction)

# Add the final test result to the metrics file.
test_result = {
    "model": final_model_name,
    "split": "test",
    **test_metrics
}

metrics_df = pd.concat(
    [metrics_df, pd.DataFrame([test_result])],
    ignore_index=True
)

metrics_df.to_csv(
    OUTPUT_DIR / "part_b_metrics.csv",
    index=False
)

print("\nFinal test-set performance:")

for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nUpdated Part B metrics saved.")