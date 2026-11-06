# ==========================================
# STEP 2: TEST FEATURE ENGINEERING (build_features)
# ==========================================

import pandas as pd
import numpy as np

from src.shipment_prep import (
    PROMISED_HOURS,
    FEATURES,
    clean_shipments,
    build_features,
)

print("=" * 65)
print("TESTING build_features() FROM src/shipment_prep.py")
print("=" * 65)

# 1. Load Datasets
print("\n[1] Loading data...")
raw_df = pd.read_csv("data/raw_shipments.csv")
hubs_df = pd.read_csv("data/hubs.csv")
print(f"  -> raw_shipments.csv : {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
print(f"  -> hubs.csv          : {hubs_df.shape[0]} rows, {hubs_df.shape[1]} columns")

# 2. Clean Shipments
print("\n[2] Cleaning raw shipments...")
clean_df, medians = clean_shipments(raw_df)
print(f"  -> Cleaned shape     : {clean_df.shape}")
print(f"  -> Imputed medians   : {medians}")

# 3. Build Features
print("\n[3] Running build_features(clean, hubs)...")
featured_df = build_features(clean_df, hubs_df)
print(f"  -> Featured shape    : {featured_df.shape}")

# 4. Feature Verification & Checks
print("\n[4] Verification Checks:")

# Check 1: Row count preservation
assert len(featured_df) == len(clean_df), "Row count mismatch after hub merge!"
print(f"  [PASS] Row count preserved ({len(featured_df)} rows)")

# Check 2: Hub merge columns
hub_cols = ["hub_city", "region", "hub_tier", "hub_daily_capacity"]
missing_hub_cols = [col for col in hub_cols if col not in featured_df.columns]
assert not missing_hub_cols, f"Missing hub columns: {missing_hub_cols}"
print(f"  [PASS] Hub columns merged: {hub_cols}")

# Check 3: Check engineered features exist
engineered_cols = [
    "promised_hours",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "log_declared_value",
]
missing_eng_cols = [col for col in engineered_cols if col not in featured_df.columns]
assert not missing_eng_cols, f"Missing engineered columns: {missing_eng_cols}"
print(f"  [PASS] Engineered columns created: {engineered_cols}")

# Check 4: Check model FEATURES completeness and null count
missing_features = [col for col in FEATURES if col not in featured_df.columns]
assert not missing_features, f"Missing FEATURES: {missing_features}"
null_counts = featured_df[FEATURES].isnull().sum()
total_nulls = null_counts.sum()
assert total_nulls == 0, f"Found nulls in FEATURES:\n{null_counts[null_counts > 0]}"
print(f"  [PASS] All {len(FEATURES)} model FEATURES present with 0 null values")

# Check 5: Promised hours mapping logic
for service, hours in PROMISED_HOURS.items():
    sample = featured_df[featured_df["service_type"] == service]
    if len(sample) > 0:
        assert (sample["promised_hours"] == hours).all(), f"Mismatch in {service} hours"
print(f"  [PASS] PROMISED_HOURS mapping verified (Express: 36, Standard: 60, Economy: 96)")

# Check 6: Binary flags
assert set(featured_df["is_cod"].unique()).issubset({0, 1}), "is_cod is not binary"
assert set(featured_df["is_weekend"].unique()).issubset({0, 1}), "is_weekend is not binary"
print("  [PASS] Binary flags (is_cod, is_weekend) verified")

# Check 7: Log transform check
expected_log = np.log(featured_df["declared_value_inr"])
assert np.allclose(featured_df["log_declared_value"], expected_log), "log_declared_value mismatch!"
print("  [PASS] log_declared_value formula verified [np.log(declared_value_inr)]")

# 5. Summary Statistics for Engineered Features
print("\n[5] Summary Statistics of Engineered Features:")
print("  * Service Type vs Promised Hours:")
for s, count in featured_df["service_type"].value_counts().items():
    print(f"      - {s:<10}: {count} rows (Promised: {PROMISED_HOURS.get(s)}h)")

print(f"  * COD Shipments (is_cod = 1) : {featured_df['is_cod'].sum()} / {len(featured_df)}")
print(f"  * Weekend Dispatches (is_weekend = 1) : {featured_df['is_weekend'].sum()} / {len(featured_df)}")
print(f"  * Dispatch Hours Range : {featured_df['dispatch_hour'].min()} to {featured_df['dispatch_hour'].max()}")
print(f"  * Log Declared Value Range : {featured_df['log_declared_value'].min():.2f} to {featured_df['log_declared_value'].max():.2f}")

# 6. Display Sample Outputs
print("\n[6] Sample Preview of Engineered Features (first 5 rows):")
sample_cols = [
    "shipment_id",
    "hub_id",
    "hub_city",
    "service_type",
    "promised_hours",
    "payment_mode",
    "is_cod",
    "dispatch_hour",
    "is_weekend",
    "declared_value_inr",
    "log_declared_value",
]
print(featured_df[sample_cols].head(5).to_string(index=False))

print("\n" + "=" * 65)
print("ALL TESTS PASSED SUCCESSFULLY!")
print("=" * 65)
