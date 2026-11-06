import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.shipment_prep import (
    clean_shipments,
    build_features,
    FEATURES,
)


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

RAW_SHIPMENTS_PATH = DATA_DIR / "raw_shipments.csv"
HUBS_PATH = DATA_DIR / "hubs.csv"
MODELLING_TABLE_PATH = DATA_DIR / "modelling_table.csv"
CHECKS_PATH = OUTPUT_DIR / "part_a_checks.csv"
MEDIANS_PATH = MODEL_DIR / "imputation_medians.json"


# --------------------------------------------------
# 2. Read input data
# --------------------------------------------------

print("Reading input CSV files...")

raw_shipments = pd.read_csv(RAW_SHIPMENTS_PATH)
hubs = pd.read_csv(HUBS_PATH)

raw_rows = len(raw_shipments)

print(f"Raw historical shipments: {raw_rows}")
print(f"Available hubs: {len(hubs)}")


# --------------------------------------------------
# 3. Capture BEFORE-cleaning metrics
# --------------------------------------------------

# Remove exact duplicates first, as required for all
# before-cleaning counts.
dedup_raw = raw_shipments.drop_duplicates().copy()

rows_after_dedup = len(dedup_raw)
duplicate_rows_removed = raw_rows - rows_after_dedup

# Missing values before any imputation.
missing_distance_km_before = dedup_raw["distance_km"].isna().sum()

# This counts actual missing weights only.
# Sentinel values above 100 kg are counted separately.
missing_parcel_weight_before = (
    dedup_raw["parcel_weight_kg"].isna().sum()
)

missing_declared_value_before = (
    dedup_raw["declared_value"].isna().sum()
)

missing_payment_mode_before = (
    dedup_raw["payment_mode"].isna().sum()
)

# Count scanner-fault sentinel weights separately.
numeric_weights = pd.to_numeric(
    dedup_raw["parcel_weight_kg"],
    errors="coerce"
)

sentinel_weight_rows = (numeric_weights > 100).sum()

# Number of distinct service_type strings before normalization.
service_type_spellings_before = (
    dedup_raw["service_type"].nunique()
)


# --------------------------------------------------
# 4. Clean historical shipments
# --------------------------------------------------

print("\nCleaning historical shipments...")

clean_df, medians = clean_shipments(raw_shipments)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

with open(MEDIANS_PATH, "w", encoding="utf-8") as file:
    json.dump(medians, file, indent=4)

print(f"Historical medians saved to: {MEDIANS_PATH}")

service_type_spellings_after = (
    clean_df["service_type"].nunique()
)

print("Historical medians:")
for column, value in medians.items():
    print(f"{column}: {value}")


# --------------------------------------------------
# 5. Build features and attach hub information
# --------------------------------------------------

print("\nBuilding shipment features...")

modelling_df = build_features(clean_df, hubs)

rows_after_hub_merge = len(modelling_df)

# A left merge must preserve all historical shipment rows.
if rows_after_hub_merge != rows_after_dedup:
    raise ValueError(
        "Hub merge changed the shipment row count. "
        f"Expected {rows_after_dedup}, got {rows_after_hub_merge}."
    )


# --------------------------------------------------
# 6. Create target variable
# --------------------------------------------------

print("\nCreating target variable...")

modelling_df["is_delayed"] = (
    modelling_df["delivery_hours"]
    > modelling_df["promised_hours"]
).astype(int)


# --------------------------------------------------
# 7. Validate modelling table
# --------------------------------------------------

if modelling_df["shipment_id"].duplicated().any():
    raise ValueError("Duplicate shipment IDs found.")

if modelling_df["hub_city"].isna().any():
    raise ValueError("Some shipments could not be matched to a hub.")

# Required features plus target and actual delivery outcome.
columns_to_check = FEATURES + [
    "delivery_hours",
    "is_delayed",
]

nulls_in_features_and_target = (
    modelling_df[columns_to_check].isna().sum().sum()
)

if nulls_in_features_and_target != 0:
    raise ValueError(
        f"Found {nulls_in_features_and_target} missing cells "
        "in features and target columns."
    )


# --------------------------------------------------
# 8. Calculate declared-value metrics
# --------------------------------------------------

# These statistics must use the median-filled modelling table.
declared_values = modelling_df["declared_value_inr"]
log_declared_values = np.log(declared_values)

declared_value_mean_inr = declared_values.mean()
declared_value_median_inr = declared_values.median()

log_declared_value_mean = log_declared_values.mean()
log_declared_value_median = log_declared_values.median()


# --------------------------------------------------
# 9. Calculate delivery-hours outlier metrics
# --------------------------------------------------

delivery_hours = modelling_df["delivery_hours"]

# Calculate unrounded quartiles.
hours_q1 = delivery_hours.quantile(0.25)
hours_q3 = delivery_hours.quantile(0.75)

hours_iqr = hours_q3 - hours_q1

# Calculate bounds from unrounded quartiles.
hours_lower_bound = hours_q1 - (1.5 * hours_iqr)
hours_upper_bound = hours_q3 + (1.5 * hours_iqr)

hours_outlier_count = (
    (delivery_hours < hours_lower_bound)
    | (delivery_hours > hours_upper_bound)
).sum()


# --------------------------------------------------
# 10. Calculate delay metrics
# --------------------------------------------------

delayed_count = modelling_df["is_delayed"].sum()
delay_rate = modelling_df["is_delayed"].mean()


# --------------------------------------------------
# 11. Build checks table
# --------------------------------------------------

checks = {
    "raw_rows": raw_rows,
    "duplicate_rows_removed": duplicate_rows_removed,
    "rows_after_dedup": rows_after_dedup,

    "missing_distance_km_before": missing_distance_km_before,
    "missing_parcel_weight_before": missing_parcel_weight_before,
    "sentinel_weight_rows": sentinel_weight_rows,
    "missing_declared_value_before": missing_declared_value_before,
    "missing_payment_mode_before": missing_payment_mode_before,

    "service_type_spellings_before": service_type_spellings_before,
    "service_type_spellings_after": service_type_spellings_after,

    "median_distance_km": medians["distance_km"],
    "median_parcel_weight_kg": medians["parcel_weight_kg"],
    "median_declared_value_inr": medians["declared_value_inr"],

    "rows_after_hub_merge": rows_after_hub_merge,
    "nulls_in_features_and_target": nulls_in_features_and_target,

    "declared_value_mean_inr": declared_value_mean_inr,
    "declared_value_median_inr": declared_value_median_inr,
    "log_declared_value_mean": log_declared_value_mean,
    "log_declared_value_median": log_declared_value_median,

    "delayed_count": delayed_count,
    "delay_rate": delay_rate,

    "hours_q1": hours_q1,
    "hours_q3": hours_q3,
    "hours_lower_bound": hours_lower_bound,
    "hours_upper_bound": hours_upper_bound,
    "hours_outlier_count": hours_outlier_count,
}

checks_df = pd.DataFrame(
    list(checks.items()),
    columns=["metric", "value"]
)

# Round decimal metrics only when writing the output.
checks_df["value"] = checks_df["value"].apply(
    lambda value: round(float(value), 4)
    if isinstance(value, (float, np.floating))
    else value
)


# --------------------------------------------------
# 12. Create delay evidence table
# --------------------------------------------------

print("\nGenerating delay evidence table...")

# Create distance bands.
# right=True means the right edge is inclusive:
# (0, 200], (200, 400], (400, 600], (600, 2000]
distance_bins = [0, 200, 400, 600, 2000]

distance_labels = [
    "0-200",
    "201-400",
    "401-600",
    "601+"
]

modelling_df["distance_band"] = pd.cut(
    modelling_df["distance_km"],
    bins=distance_bins,
    labels=distance_labels,
    right=True,
    include_lowest=True
)


# Helper function to summarize each group
def summarize_delay_groups(df, dimension):
    grouped = (
        df.groupby(dimension, observed=False)
        .agg(
            shipments=("shipment_id", "count"),
            delayed=("is_delayed", "sum"),
            delay_rate=("is_delayed", "mean"),
            mean_hours=("delivery_hours", "mean")
        )
        .reset_index()
    )

    grouped = grouped.rename(
        columns={dimension: "group"}
    )

    grouped.insert(
        0,
        "dimension",
        dimension
    )

    return grouped[
        [
            "dimension",
            "group",
            "shipments",
            "delayed",
            "delay_rate",
            "mean_hours"
        ]
    ]


# Generate summaries for the three required dimensions.
service_summary = summarize_delay_groups(
    modelling_df,
    "service_type"
)

hub_tier_summary = summarize_delay_groups(
    modelling_df,
    "hub_tier"
)

distance_summary = summarize_delay_groups(
    modelling_df,
    "distance_band"
)

# Combine all three summaries.
delay_by_group = pd.concat(
    [
        service_summary,
        hub_tier_summary,
        distance_summary
    ],
    ignore_index=True
)

# Round decimal metrics to four places.
delay_by_group["delay_rate"] = (
    delay_by_group["delay_rate"].round(4)
)

delay_by_group["mean_hours"] = (
    delay_by_group["mean_hours"].round(4)
)

# Save output.
DELAY_GROUP_PATH = OUTPUT_DIR / "part_a_delay_by_group.csv"

delay_by_group.to_csv(
    DELAY_GROUP_PATH,
    index=False
)

print("\nDelay evidence table:")
print(delay_by_group.to_string(index=False))

print(f"\nSaved: {DELAY_GROUP_PATH}")


# --------------------------------------------------
# 13. Save output files
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

modelling_df.to_csv(
    MODELLING_TABLE_PATH,
    index=False
)

checks_df.to_csv(
    CHECKS_PATH,
    index=False
)


# --------------------------------------------------
# 14. Generate correlation matrix
# --------------------------------------------------
print("\nGenerating correlation matrix...")

correlations = modelling_df.corr(numeric_only=True)

# Keep the row labels as the first CSV column
correlations.index.name = "column"

CORRELATIONS_PATH = OUTPUT_DIR / "part_a_correlations.csv"
correlations.to_csv(CORRELATIONS_PATH, index=True)

print(f"Correlation matrix saved to: {CORRELATIONS_PATH}")


# --------------------------------------------------
# 15. Display summary
# --------------------------------------------------

print("\n" + "=" * 55)
print("PART A - DATA CLEANING CHECKS")
print("=" * 55)

print(checks_df.to_string(index=False))

print("\n" + "=" * 55)
print("FILES GENERATED SUCCESSFULLY")
print("=" * 55)

print(f"Modelling table: {MODELLING_TABLE_PATH}")
print(f"Checks file:     {CHECKS_PATH}")

print(f"\nModelling table rows: {len(modelling_df)}")
print(f"Modelling table columns: {len(modelling_df.columns)}")

print("\nTarget distribution:")
print(modelling_df["is_delayed"].value_counts().sort_index())

print("\nPart A checks completed.")
