
from pathlib import Path

import pandas as pd

from src.shipment_prep import (
    clean_shipments,
    build_features,
)


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_SHIPMENTS_PATH = DATA_DIR / "raw_shipments.csv"
HUBS_PATH = DATA_DIR / "hubs.csv"
MODELLING_TABLE_PATH = DATA_DIR / "modelling_table.csv"


# --------------------------------------------------
# 2. Read input CSV files
# --------------------------------------------------

print("Reading input CSV files...")

raw_shipments = pd.read_csv(RAW_SHIPMENTS_PATH)
hubs = pd.read_csv(HUBS_PATH)

print(f"Raw historical shipments: {len(raw_shipments)}")
print(f"Available hubs: {len(hubs)}")


# --------------------------------------------------
# 3. Clean historical shipment data
# --------------------------------------------------

print("\nCleaning historical shipments...")

clean_shipments_df, medians = clean_shipments(raw_shipments)

print(f"Cleaned shipments: {len(clean_shipments_df)}")
print("\nHistorical imputation medians:")

for column, value in medians.items():
    print(f"{column}: {value}")


# --------------------------------------------------
# 4. Build shipment features
# --------------------------------------------------

print("\nBuilding shipment features...")

modelling_df = build_features(clean_shipments_df, hubs)


# --------------------------------------------------
# 5. Create target variable: is_delayed
# --------------------------------------------------

print("\nCreating target variable...")

modelling_df["is_delayed"] = (
    modelling_df["delivery_hours"] > modelling_df["promised_hours"]
).astype(int)


# --------------------------------------------------
# 6. Validate modelling table
# --------------------------------------------------

print("\nValidating modelling table...")

if modelling_df["shipment_id"].duplicated().any():
    raise ValueError("Duplicate shipment IDs found in modelling table.")

if modelling_df["is_delayed"].isna().any():
    raise ValueError("Missing values found in target variable is_delayed.")

if modelling_df["hub_city"].isna().any():
    raise ValueError("Some shipments could not be matched to a hub.")

print("Unique shipment IDs: verified")
print("Target variable: verified")
print("Hub information: verified")


# --------------------------------------------------
# 7. Save modelling table
# --------------------------------------------------

modelling_df.to_csv(MODELLING_TABLE_PATH, index=False)

print("\nModelling table saved successfully!")
print(f"Output: {MODELLING_TABLE_PATH}")

print("\nFinal modelling table details:")
print(f"Rows: {len(modelling_df)}")
print(f"Columns: {len(modelling_df.columns)}")

print("\nTarget distribution:")
print(modelling_df["is_delayed"].value_counts().sort_index())

print("\nTarget percentages:")
print(
    (modelling_df["is_delayed"].value_counts(normalize=True).sort_index() * 100)
    .round(2)
)

print("\nFirst five rows:")
print(modelling_df.head())
