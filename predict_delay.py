from pathlib import Path

import joblib
import pandas as pd

from src.shipment_prep import clean_shipments, build_features


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STEP 1: LOAD INPUT DATA AND SAVED ARTEFACTS
# ============================================================

print("Loading incoming shipments and model artefacts...")

new_shipments = pd.read_csv(DATA_DIR / "new_shipments.csv")
hubs = pd.read_csv(DATA_DIR / "hubs.csv")

model = joblib.load(MODELS_DIR / "delay_model.pkl")

threshold = float(
    pd.read_csv(MODELS_DIR / "delay_threshold.csv")
    ["threshold"].iloc[0]
)

feature_order = (
    pd.read_csv(MODELS_DIR / "feature_order.csv")
    ["feature"].tolist()
)

impute_values_df = pd.read_csv(
    MODELS_DIR / "impute_values.csv"
)

stored_medians = dict(
    zip(
        impute_values_df["column"],
        impute_values_df["median"]
    )
)

hub_segments = pd.read_csv(
    OUTPUT_DIR / "hub_segments.csv"
)

print("Incoming shipments:", len(new_shipments))
print("Stored threshold:", threshold)
print("Features loaded:", len(feature_order))


# ============================================================
# STEP 2: PRESERVE ORIGINAL SHIPMENT ORDER
# ============================================================

# This index allows us to restore the exact input file order
# after joining the hub segment information.

new_shipments["_original_order"] = range(len(new_shipments))


# ============================================================
# STEP 3: CLEAN INCOMING SHIPMENTS
# ============================================================

print("\nCleaning incoming shipments...")

cleaned_shipments, _ = clean_shipments(
    new_shipments,
    medians=stored_medians
)

print("Cleaning completed.")


# ============================================================
# STEP 4: BUILD FEATURES USING SHARED PREPROCESSING
# ============================================================

print("\nBuilding model features...")

shipment_features = build_features(
    cleaned_shipments,
    hubs
)

# Arrange the features in exactly the saved training order.
X_new = shipment_features[feature_order]

print("Feature matrix shape:", X_new.shape)


# ============================================================
# STEP 5: PREDICT DELAY PROBABILITIES
# ============================================================

print("\nScoring incoming shipments...")

# Column 1 represents the positive class: delayed (is_delayed = 1).
raw_probabilities = model.predict_proba(X_new)[:, 1]

# Round probabilities to four decimal places as required.
rounded_probabilities = raw_probabilities.round(4)

# Apply the saved threshold to the rounded probability.
flagged = (
    rounded_probabilities >= threshold
).astype(int)


# ============================================================
# STEP 6: JOIN HUB SEGMENTS
# ============================================================

shipment_results = cleaned_shipments[
    ["shipment_id", "hub_id", "_original_order"]
].copy()

shipment_results["delay_probability"] = rounded_probabilities
shipment_results["flagged"] = flagged

shipment_results = shipment_results.merge(
    hub_segments[["hub_id", "segment"]],
    on="hub_id",
    how="left",
    validate="many_to_one"
)

# Restore original incoming shipment file order.
shipment_results = (
    shipment_results
    .sort_values("_original_order")
    .reset_index(drop=True)
)


# ============================================================
# STEP 7: ASSIGN OPERATIONAL ACTIONS
# ============================================================

shipment_results["action"] = "No action"

shipment_results.loc[
    (shipment_results["flagged"] == 1)
    & (shipment_results["segment"] == "Escalate"),
    "action"
] = "Escalate to hub manager"

shipment_results.loc[
    (shipment_results["flagged"] == 1)
    & (shipment_results["segment"] != "Escalate"),
    "action"
] = "Proactive customer call"


# ============================================================
# STEP 8: SAVE FINAL ACTION LIST
# ============================================================

final_columns = [
    "shipment_id",
    "hub_id",
    "segment",
    "delay_probability",
    "flagged",
    "action"
]

final_output = shipment_results[final_columns].copy()

output_path = OUTPUT_DIR / "new_shipment_actions.csv"

final_output.to_csv(
    output_path,
    index=False
)


# ============================================================
# STEP 9: VALIDATE OUTPUT
# ============================================================

print("\nValidating scoring output...")

assert len(final_output) == 48, (
    "Expected exactly 48 scored shipments."
)

assert final_output["shipment_id"].is_unique, (
    "Shipment IDs must be unique."
)

assert not final_output.isnull().any().any(), (
    "Scoring output contains missing values."
)

assert (
    final_output["flagged"]
    == (
        final_output["delay_probability"] >= threshold
    ).astype(int)
).all(), "Flagged values do not match the stored threshold."

# Validate the action rules.
expected_actions = []

for row in final_output.itertuples(index=False):

    if row.flagged == 0:
        expected_actions.append("No action")

    elif row.segment == "Escalate":
        expected_actions.append("Escalate to hub manager")

    else:
        expected_actions.append("Proactive customer call")

assert (
    final_output["action"].tolist()
    == expected_actions
), "One or more actions do not follow the required rules."

# Validate original shipment order.
assert (
    final_output["shipment_id"].tolist()
    == new_shipments.sort_values("_original_order")[
        "shipment_id"
    ].tolist()
), "Shipment order changed during scoring."


# ============================================================
# STEP 10: DISPLAY OPERATIONAL SUMMARY
# ============================================================

flagged_count = int(final_output["flagged"].sum())

manager_escalations = int(
    (
        final_output["action"]
        == "Escalate to hub manager"
    ).sum()
)

customer_calls = int(
    (
        final_output["action"]
        == "Proactive customer call"
    ).sum()
)

print("\nScoring summary:")
print("Total shipments:", len(final_output))
print("Flagged shipments:", flagged_count)
print("Hub-manager escalations:", manager_escalations)
print("Proactive customer calls:", customer_calls)
print("No-action shipments:", len(final_output) - flagged_count)

print("\nAction distribution:")
print(final_output["action"].value_counts())

print("\nFirst 10 scored shipments:")
print(final_output.head(10).to_string(index=False))

print("\nOutput saved to:")
print(output_path)

print("\nPart D - Step 6 completed successfully.")
