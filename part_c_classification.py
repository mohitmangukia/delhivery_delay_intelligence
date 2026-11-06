from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.shipment_prep import FEATURES

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "modelling_table.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STEP 1: LOAD DATA AND STRATIFIED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("Loading modelling table...")

df = pd.read_csv(DATA_PATH)

X = df[FEATURES]
y = df["is_delayed"]

print(f"Total shipments: {len(df)}")
print(f"Number of features: {len(FEATURES)}")

print("\nTarget distribution:")
print(y.value_counts().sort_index())


# ------------------------------------------------------------
# 1.1 Stratified split
# ------------------------------------------------------------

print("\nSplitting dataset using stratification...")

# First split: 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Second split: 50% validation, 50% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)


# ------------------------------------------------------------
# 1.2 Verify split sizes and class distribution
# ------------------------------------------------------------

split_results = []

for split_name, X_split, y_split in [
    ("train", X_train, y_train),
    ("validation", X_val, y_val),
    ("test", X_test, y_test)
]:
    split_results.append({
        "split": split_name,
        "total_rows": len(X_split),
        "not_delayed": int((y_split == 0).sum()),
        "delayed": int((y_split == 1).sum())
    })

split_df = pd.DataFrame(split_results)

split_df.to_csv(
    OUTPUT_DIR / "part_c_checks.csv",
    index=False
)

print("\nStratified split results:")
print(split_df.to_string(index=False))

print("\nPart C - Step 1 completed.")


# ============================================================
# STEP 2: BASELINE AND INITIAL CLASSIFIERS
# ============================================================

def evaluate_classifier(model_name, y_true, y_pred):
    """
    Calculate classification metrics and confusion matrix values.

    Positive class (1) represents a delayed shipment.
    """

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "model": model_name,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        )
    }


classification_results = []


# ------------------------------------------------------------
# 2.1 Blind baseline: predict every shipment as on-time
# ------------------------------------------------------------

print("\nEvaluating blind baseline...")

baseline_predictions = np.zeros(
    len(y_val),
    dtype=int
)

baseline_result = evaluate_classifier(
    "blind_all_on_time",
    y_val,
    baseline_predictions
)

classification_results.append(baseline_result)


# ------------------------------------------------------------
# 2.2 Logistic Regression
# ------------------------------------------------------------

print("\nTraining Logistic Regression...")

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_val)

logistic_result = evaluate_classifier(
    "logistic_regression",
    y_val,
    logistic_predictions
)

classification_results.append(logistic_result)


# ------------------------------------------------------------
# 2.3 Decision Tree: compare different depths
# ------------------------------------------------------------

print("\nTesting Decision Tree depths...")

tree_depth_results = []

for depth in [2, 3, 4, 5, 6, 8, 10]:

    tree_model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42
    )

    tree_model.fit(X_train, y_train)

    tree_predictions = tree_model.predict(X_val)

    tree_result = evaluate_classifier(
        f"decision_tree_depth_{depth}",
        y_val,
        tree_predictions
    )

    tree_depth_results.append(tree_result)


# Highest validation F1 wins.
# If F1 is tied, choose the smaller tree depth.
best_tree_result = max(
    tree_depth_results,
    key=lambda result: (
        result["f1"],
        -int(result["model"].split("_")[-1])
    )
)

classification_results.append(best_tree_result)


# Display tree results in the same selection order.
tree_depth_df = pd.DataFrame(tree_depth_results)

tree_depth_df = tree_depth_df.sort_values(
    by="f1",
    ascending=False,
    kind="stable"
)

# Explicitly apply smaller-depth tie-breaking for display.
tree_depth_df["_depth"] = (
    tree_depth_df["model"]
    .str.extract(r"(\d+)$")
    .astype(int)
)

tree_depth_df = tree_depth_df.sort_values(
    by=["f1", "_depth"],
    ascending=[False, True],
    kind="stable"
).drop(columns="_depth")

print("\nDecision Tree depth results:")

print(
    tree_depth_df[
        ["model", "accuracy", "precision", "recall", "f1"]
    ].to_string(index=False)
)

print(f"\nSelected tree: {best_tree_result['model']}")


# ------------------------------------------------------------
# 2.4 Random Forest
# ------------------------------------------------------------

print("\nTraining Random Forest...")

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(X_train, y_train)

rf_predictions = random_forest_model.predict(X_val)

rf_result = evaluate_classifier(
    "random_forest_default",
    y_val,
    rf_predictions
)

classification_results.append(rf_result)


# ------------------------------------------------------------
# 2.5 Save and display classification results
# ------------------------------------------------------------

part_c_models_df = pd.DataFrame(classification_results)

part_c_models_df = part_c_models_df[
    [
        "model",
        "TN",
        "FP",
        "FN",
        "TP",
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]
]

part_c_models_df.to_csv(
    OUTPUT_DIR / "part_c_models.csv",
    index=False
)

print("\nInitial classification results:")

print(
    part_c_models_df.to_string(index=False)
)

print("\nPart C - Step 2 completed.")


# ============================================================
# STEP 3: FEATURE IMPORTANCE AND FEATURE SELECTION
# ============================================================

# These five features are expected to be noise features.
NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no"
]


# ------------------------------------------------------------
# 3.1 Random Forest feature importance using all 17 features
# ------------------------------------------------------------

print("\nCalculating Random Forest feature importance...")

feature_rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

feature_rf_model.fit(X_train, y_train)

rf_importances = feature_rf_model.feature_importances_


# ------------------------------------------------------------
# 3.2 ANOVA F-scores using all 17 features
# ------------------------------------------------------------

print("\nCalculating ANOVA F-scores...")

f_selector = SelectKBest(
    score_func=f_classif,
    k="all"
)

f_selector.fit(X_train, y_train)

f_scores = f_selector.scores_


# ------------------------------------------------------------
# 3.3 Combine feature scores and rankings
# ------------------------------------------------------------

feature_scores_df = pd.DataFrame({
    "feature": FEATURES,
    "f_score": f_scores,
    "rf_importance": rf_importances,
    "is_noise": [
        feature in NOISE_COLUMNS
        for feature in FEATURES
    ]
})

# Rank 1 means the highest F-score.
# Tied scores receive the same rank.
feature_scores_df["f_rank"] = (
    feature_scores_df["f_score"]
    .rank(method="min", ascending=False)
    .astype(int)
)

feature_scores_df = feature_scores_df[
    [
        "feature",
        "f_score",
        "f_rank",
        "rf_importance",
        "is_noise"
    ]
]

feature_scores_df = feature_scores_df.sort_values(
    by=["f_rank", "feature"],
    ascending=[True, True]
)

feature_scores_df.to_csv(
    OUTPUT_DIR / "part_c_feature_scores.csv",
    index=False
)

print("\nFeature importance and F-score ranking:")

print(
    feature_scores_df.to_string(index=False)
)


# ------------------------------------------------------------
# 3.4 Calculate combined Random Forest importance of noise
# ------------------------------------------------------------

noise_importance_share = feature_scores_df.loc[
    feature_scores_df["is_noise"],
    "rf_importance"
].sum()

print(
    "\nCombined Random Forest importance of five noise features:",
    round(noise_importance_share, 6)
)


# ------------------------------------------------------------
# 3.5 Test different SelectKBest feature counts
# ------------------------------------------------------------

print("\nTesting SelectKBest feature counts...")

selectk_results = []

for k in [5, 7, 10, 12, 17]:

    selectk_model = Pipeline([
        ("selector", SelectKBest(
            score_func=f_classif,
            k=k
        )),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    selectk_model.fit(X_train, y_train)

    selectk_predictions = selectk_model.predict(X_val)

    validation_f1 = f1_score(
        y_val,
        selectk_predictions,
        zero_division=0
    )

    selectk_results.append({
        "k": k,
        "val_f1": validation_f1
    })


# ------------------------------------------------------------
# 3.6 Save and display SelectKBest results
# ------------------------------------------------------------

selectk_df = pd.DataFrame(selectk_results)

selectk_df.to_csv(
    OUTPUT_DIR / "part_c_selectk.csv",
    index=False
)

print("\nSelectKBest validation results:")

print(
    selectk_df.to_string(index=False)
)

print("\nPart C - Step 3 completed.")



# ============================================================
# STEP 4: RANDOM FOREST HYPERPARAMETER TUNING
# ============================================================

print("\nStarting Random Forest hyperparameter tuning...")


# ------------------------------------------------------------
# 4.1 Select the top 7 features using training data only
# ------------------------------------------------------------

print("\nSelecting the top 7 features...")

tuning_selector = SelectKBest(
    score_func=f_classif,
    k=7
)

X_train_selected = tuning_selector.fit_transform(
    X_train,
    y_train
)

X_val_selected = tuning_selector.transform(X_val)

print("Training features after selection:", X_train_selected.shape[1])
print("Validation features after selection:", X_val_selected.shape[1])


# ------------------------------------------------------------
# 4.2 Define the hyperparameter search space
# ------------------------------------------------------------

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10]
}

# Total combinations:
# 3 * 3 * 3 = 27


# ------------------------------------------------------------
# 4.3 Grid Search
# ------------------------------------------------------------

print("\nRunning Grid Search...")

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(
        random_state=42
    ),
    param_grid=param_grid,
    cv=5
)

grid_search.fit(
    X_train_selected,
    y_train
)

grid_predictions = grid_search.predict(
    X_val_selected
)

grid_val_f1 = f1_score(
    y_val,
    grid_predictions,
    zero_division=0
)

grid_result = {
    "search": "grid_search",
    "candidates": 27,
    "cv_folds": 5,
    "total_fits": 27 * 5,
    "best_params": str(grid_search.best_params_),
    "best_cv_accuracy": grid_search.best_score_,
    "val_f1": grid_val_f1
}

print("Grid Search best parameters:", grid_search.best_params_)
print("Grid Search best CV accuracy:", grid_search.best_score_)
print("Grid Search validation F1:", grid_val_f1)


# ------------------------------------------------------------
# 4.4 Randomized Search
# ------------------------------------------------------------

print("\nRunning Randomized Search...")

randomized_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(
        random_state=42
    ),
    param_distributions=param_grid,
    n_iter=9,
    cv=5,
    random_state=42
)

randomized_search.fit(
    X_train_selected,
    y_train
)

randomized_predictions = randomized_search.predict(
    X_val_selected
)

randomized_val_f1 = f1_score(
    y_val,
    randomized_predictions,
    zero_division=0
)

randomized_result = {
    "search": "randomized_search",
    "candidates": 9,
    "cv_folds": 5,
    "total_fits": 9 * 5,
    "best_params": str(randomized_search.best_params_),
    "best_cv_accuracy": randomized_search.best_score_,
    "val_f1": randomized_val_f1
}

print("Randomized Search best parameters:", randomized_search.best_params_)
print("Randomized Search best CV accuracy:", randomized_search.best_score_)
print("Randomized Search validation F1:", randomized_val_f1)


# ------------------------------------------------------------
# 4.5 Evaluate default Random Forest on the same 7 features
# ------------------------------------------------------------

print("\nEvaluating default Random Forest on 7 features...")

default_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

default_forest.fit(
    X_train_selected,
    y_train
)

default_predictions = default_forest.predict(
    X_val_selected
)

default_val_f1 = f1_score(
    y_val,
    default_predictions,
    zero_division=0
)

default_result = {
    "search": "default_forest_same_7_features",
    "candidates": 1,
    "cv_folds": 0,
    "total_fits": 1,
    "best_params": "defaults",
    "best_cv_accuracy": np.nan,
    "val_f1": default_val_f1
}

print("Default forest validation F1:", default_val_f1)


# ------------------------------------------------------------
# 4.6 Save tuning results
# ------------------------------------------------------------

tuning_results = [
    grid_result,
    randomized_result,
    default_result
]

tuning_df = pd.DataFrame(tuning_results)

tuning_df = tuning_df[
    [
        "search",
        "candidates",
        "cv_folds",
        "total_fits",
        "best_params",
        "best_cv_accuracy",
        "val_f1"
    ]
]

tuning_df.to_csv(
    OUTPUT_DIR / "part_c_tuning.csv",
    index=False
)

print("\nHyperparameter tuning results:")

print(
    tuning_df.to_string(index=False)
)

print("\nPart C - Step 4 completed.")


# ============================================================
# STEP 5: SELECT AND TRAIN FINAL CLASSIFICATION MODEL
# ============================================================

print("\nSelecting the final classification model...")

# ------------------------------------------------------------
# 5.1 Apply the required selection and tie-breaking rules
# ------------------------------------------------------------

selection_priority = {
    "default_forest_same_7_features": 0,
    "grid_search": 1,
    "randomized_search": 2
}

tuning_df["_priority"] = tuning_df["search"].map(
    selection_priority
)

tuning_df = tuning_df.sort_values(
    by=["val_f1", "_priority"],
    ascending=[False, True]
)

chosen_from = tuning_df.iloc[0]["search"]

print("Selected configuration:", chosen_from)


# ------------------------------------------------------------
# 5.2 Get the selected Random Forest parameters
# ------------------------------------------------------------

if chosen_from == "grid_search":
    chosen_params = grid_search.best_params_

elif chosen_from == "randomized_search":
    chosen_params = randomized_search.best_params_

else:
    chosen_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1
    }

print("Selected parameters:", chosen_params)


# ------------------------------------------------------------
# 5.3 Build final pipeline with SelectKBest and Random Forest
# ------------------------------------------------------------

final_model = Pipeline([
    ("selector", SelectKBest(
        score_func=f_classif,
        k=7
    )),
    ("model", RandomForestClassifier(
        random_state=42,
        **chosen_params
    ))
])


# ------------------------------------------------------------
# 5.4 Fit final model on training data only
# ------------------------------------------------------------

print("\nTraining final classification pipeline...")

final_model.fit(
    X_train,
    y_train
)

print("Final model training completed.")

print("\nSelected final model configuration:")
print("Selector k:", 7)
print("Number of estimators:", chosen_params["n_estimators"])
print("Maximum depth:", chosen_params["max_depth"])
print("Minimum samples per leaf:", chosen_params["min_samples_leaf"])
print("Chosen from:", chosen_from)

print("\nPart C - Step 5 completed.")



# ============================================================
# STEP 6: DECISION THRESHOLD SELECTION
# ============================================================

print("\nEvaluating decision thresholds...")

# ------------------------------------------------------------
# 6.1 Get delayed-shipment probabilities on validation data
# ------------------------------------------------------------

validation_probabilities = final_model.predict_proba(X_val)

# Find the probability column corresponding to class 1 (delayed).
delayed_class_index = list(
    final_model.named_steps["model"].classes_
).index(1)

validation_delayed_probabilities = (
    validation_probabilities[:, delayed_class_index]
)


# ------------------------------------------------------------
# 6.2 Evaluate all required thresholds
# ------------------------------------------------------------

threshold_values = [
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9
]

threshold_results = []

for threshold in threshold_values:

    threshold_predictions = (
        validation_delayed_probabilities >= threshold
    ).astype(int)

    threshold_result = evaluate_classifier(
        f"threshold_{threshold}",
        y_val,
        threshold_predictions
    )

    # Compare the model's default predict() decision with the explicit
    # probability >= 0.5 rule. These can differ for probabilities exactly 0.5.
    if threshold == 0.5:
        default_validation_predictions = final_model.predict(X_val)
        different_predictions = int(np.sum(
            default_validation_predictions != threshold_predictions
        ))

        print("\nDefault predict() vs explicit 0.5 threshold:")
        print("Different validation predictions:", different_predictions)
        print(
            "Default predict() F1:",
            f1_score(y_val, default_validation_predictions, zero_division=0)
        )
        print(
            "Explicit 0.5 threshold F1:",
            f1_score(y_val, threshold_predictions, zero_division=0)
        )

    threshold_results.append({
        "threshold": threshold,
        "TN": threshold_result["TN"],
        "FP": threshold_result["FP"],
        "FN": threshold_result["FN"],
        "TP": threshold_result["TP"],
        "accuracy": threshold_result["accuracy"],
        "precision": threshold_result["precision"],
        "recall": threshold_result["recall"],
        "f1": threshold_result["f1"]
    })


# ------------------------------------------------------------
# 6.3 Save threshold comparison results
# ------------------------------------------------------------

thresholds_df = pd.DataFrame(threshold_results)

thresholds_df.to_csv(
    OUTPUT_DIR / "part_c_thresholds.csv",
    index=False
)

print("\nValidation threshold results:")

print(
    thresholds_df.to_string(index=False)
)


# ------------------------------------------------------------
# 6.4 Select the highest threshold with recall >= 0.80
# ------------------------------------------------------------

eligible_thresholds = thresholds_df[
    thresholds_df["recall"] >= 0.80
]

if not eligible_thresholds.empty:

    chosen_threshold = float(
        eligible_thresholds["threshold"].max()
    )

else:

    chosen_threshold = 0.1


print("\nSelected operational threshold:", chosen_threshold)

print(
    "Validation recall at selected threshold:",
    thresholds_df.loc[
        thresholds_df["threshold"] == chosen_threshold,
        "recall"
    ].iloc[0]
)

print("\nPart C - Step 6 completed.")


# ------------------------------------------------------------
# STEP 7: FINAL TEST EVALUATION
# ------------------------------------------------------------

print("\nEvaluating final model on the test set...")

# Generate test probabilities once
test_probabilities = final_model.predict_proba(X_test)
test_delayed_probabilities = test_probabilities[:, delayed_class_index]

# Apply the already-selected operational threshold
test_predictions = (
    test_delayed_probabilities >= chosen_threshold
).astype(int)

# Evaluate final test performance
test_result = evaluate_classifier(
    "final_test",
    y_test,
    test_predictions
)

final_test_df = pd.DataFrame([{
    "threshold": chosen_threshold,
    "TN": test_result["TN"],
    "FP": test_result["FP"],
    "FN": test_result["FN"],
    "TP": test_result["TP"],
    "accuracy": test_result["accuracy"],
    "precision": test_result["precision"],
    "recall": test_result["recall"],
    "f1": test_result["f1"]
}])

final_test_df.to_csv(
    OUTPUT_DIR / "part_c_final_test.csv",
    index=False
)

# Save the final model configuration
max_depth_value = chosen_params["max_depth"]

if max_depth_value is None:
    max_depth_value = "unlimited"

final_config_df = pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": chosen_params["n_estimators"],
    "max_depth": max_depth_value,
    "min_samples_leaf": chosen_params["min_samples_leaf"],
    "threshold": chosen_threshold,
    "chosen_from": chosen_from
}])

final_config_df.to_csv(
    OUTPUT_DIR / "part_c_final_config.csv",
    index=False
)

print("\nFinal test results:")
print(final_test_df.to_string(index=False))

print("\nFinal model configuration:")
print(final_config_df.to_string(index=False))

print("\nPart C - Step 7 completed.")