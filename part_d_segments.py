
from pathlib import Path
import pandas as pd

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "modelling_table.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STEP 1: BUILD HUB PROFILE
# ============================================================

print("Loading modelling table...")

df = pd.read_csv(DATA_PATH)

print("Historical shipments:", len(df))
print("Unique hubs:", df["hub_id"].nunique())


# ------------------------------------------------------------
# 1.1 Aggregate shipment-level data into hub-level profiles
# ------------------------------------------------------------

hub_profile = (
    df.groupby("hub_id", as_index=False)
    .agg(
        shipments=("shipment_id", "count"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean")
    )
    .sort_values("hub_id")
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# 1.2 Save hub profile
# ------------------------------------------------------------

hub_profile.to_csv(
    OUTPUT_DIR / "hub_profile.csv",
    index=False
)


# ------------------------------------------------------------
# 1.3 Display and verify results
# ------------------------------------------------------------

print("\nHub profile:")
print(hub_profile.to_string(index=False))

print("\nHub profile checks:")
print("Number of hubs:", len(hub_profile))
print("Shipment counts per hub:")
print(hub_profile["shipments"].value_counts().sort_index())

print("\nHUB01 reference check:")
print(
    hub_profile.loc[
        hub_profile["hub_id"] == "HUB01"
    ].to_string(index=False)
)

print("\nHUB13 reference check:")
print(
    hub_profile.loc[
        hub_profile["hub_id"] == "HUB13"
    ].to_string(index=False)
)

assert len(hub_profile) == 24, "Expected exactly 24 hubs."
assert (hub_profile["shipments"] == 50).all(), (
    "Every hub must have exactly 50 shipments."
)

print("\nPart D - Step 1 completed.")



# ============================================================
# STEP 2: K-MEANS HUB SEGMENTATION
# STEP 3: SEPARATION EVIDENCE
# ============================================================

import numpy as np
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


# ------------------------------------------------------------
# 2.1 Prepare the four hub-level clustering features
# ------------------------------------------------------------

PROFILE_FEATURES = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops"
]

# Keep hubs in ascending hub_id order.
hub_profile = hub_profile.sort_values("hub_id").reset_index(drop=True)

X_hubs = hub_profile[PROFILE_FEATURES]


# ------------------------------------------------------------
# 2.2 Scale the hub profile
# ------------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_hubs)


# ------------------------------------------------------------
# 2.3 Fit the required K=3 model
# ------------------------------------------------------------

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(X_scaled)

# Keep the arbitrary cluster number for now.
# We will assign business names in Step 4.
hub_profile["cluster"] = cluster_labels

print("\nK=3 cluster counts:")
print(hub_profile["cluster"].value_counts().sort_index())


# ------------------------------------------------------------
# STEP 3: Calculate across-to-within distance ratios
# ------------------------------------------------------------

# Calculate every pairwise Euclidean distance between hubs.
distance_matrix = pairwise_distances(
    X_scaled,
    metric="euclidean"
)

separation_rows = []

for k in [2, 3, 4, 5]:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    # Collect distances for pairs in the same and
    # different clusters. Only unordered pairs are used.
    within_distances = []
    across_distances = []

    for i, j in combinations(range(len(hub_profile)), 2):

        distance = distance_matrix[i, j]

        if labels[i] == labels[j]:
            within_distances.append(distance)
        else:
            across_distances.append(distance)

    mean_within = np.mean(within_distances)
    mean_across = np.mean(across_distances)

    across_over_within = mean_across / mean_within

    # Cluster sizes in descending order.
    cluster_sizes = sorted(
        np.bincount(labels).tolist(),
        reverse=True
    )

    separation_rows.append({
        "K": k,
        "across_over_within": across_over_within,
        "cluster_sizes": "-".join(map(str, cluster_sizes))
    })


# ------------------------------------------------------------
# 3.1 Save separation evidence
# ------------------------------------------------------------

separation_df = pd.DataFrame(separation_rows)

separation_df.to_csv(
    OUTPUT_DIR / "part_d_separation.csv",
    index=False
)

print("\nSeparation evidence:")
print(separation_df.to_string(index=False))


# ------------------------------------------------------------
# 3.2 Validate reference results
# ------------------------------------------------------------

k2 = separation_df.loc[separation_df["K"] == 2].iloc[0]
k3 = separation_df.loc[separation_df["K"] == 3].iloc[0]
k4 = separation_df.loc[separation_df["K"] == 4].iloc[0]
k5 = separation_df.loc[separation_df["K"] == 5].iloc[0]

assert k2["cluster_sizes"] == "15-9"
assert k3["cluster_sizes"] == "9-8-7"

assert abs(k2["across_over_within"] - 2.894) <= 0.01
assert abs(k3["across_over_within"] - 5.393) <= 0.01

assert k4["cluster_sizes"] == "8-7-5-4"
assert k5["cluster_sizes"] == "8-7-4-3-2"

assert abs(k4["across_over_within"] - 6.305) <= 0.01
assert abs(k5["across_over_within"] - 6.887) <= 0.01

print("\nPart D - Steps 2 and 3 completed.")



# ============================================================
# STEP 4: SEGMENT NAMES AND TIER MISMATCH
# ============================================================

# ------------------------------------------------------------
# 4.1 Calculate the mean delay rate for each cluster
# ------------------------------------------------------------

cluster_delay_rates = (
    hub_profile.groupby("cluster")["delay_rate"]
    .mean()
    .sort_values()
)

print("\nMean delay rate by cluster:")
print(cluster_delay_rates)


# ------------------------------------------------------------
# 4.2 Map arbitrary cluster numbers to business segment names
# ------------------------------------------------------------

segment_names = {
    cluster_delay_rates.index[0]: "Standard",
    cluster_delay_rates.index[1]: "Watch",
    cluster_delay_rates.index[2]: "Escalate"
}

hub_profile["segment"] = hub_profile["cluster"].map(segment_names)


# ------------------------------------------------------------
# 4.3 Save hub_segments.csv
# ------------------------------------------------------------

hub_details = pd.read_csv(DATA_PATH)[
    ["hub_id", "hub_city", "hub_tier"]
].drop_duplicates(subset="hub_id")

hub_segments = hub_profile[
    ["hub_id", "segment"]
].merge(
    hub_details,
    on="hub_id",
    how="left",
    validate="one_to_one"
)

hub_segments = hub_segments[
    ["hub_id", "segment", "hub_city", "hub_tier"]
].sort_values("hub_id").reset_index(drop=True)

hub_segments.to_csv(
    OUTPUT_DIR / "hub_segments.csv",
    index=False
)


# ------------------------------------------------------------
# 4.4 Build segment-level profile
# ------------------------------------------------------------

segment_profile = (
    hub_profile.groupby("segment", as_index=False)
    .agg(
        hubs=("hub_id", "count"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean")
    )
)

# Keep the business order.
segment_order = ["Standard", "Watch", "Escalate"]

segment_profile["segment"] = pd.Categorical(
    segment_profile["segment"],
    categories=segment_order,
    ordered=True
)

segment_profile = (
    segment_profile.sort_values("segment")
    .reset_index(drop=True)
)

segment_profile["segment"] = segment_profile["segment"].astype(str)

segment_profile.to_csv(
    OUTPUT_DIR / "segment_profile.csv",
    index=False
)


# ------------------------------------------------------------
# 4.5 Identify tier-versus-segment mismatches
# ------------------------------------------------------------

tier_expected_segment = {
    1: "Standard",
    2: "Watch",
    3: "Escalate"
}

hub_segments["expected_segment_from_tier"] = (
    hub_segments["hub_tier"].map(tier_expected_segment)
)

tier_mismatch = hub_segments[
    hub_segments["segment"] != hub_segments["expected_segment_from_tier"]
].copy()

# Save only the required hub_segments columns.
tier_mismatch = tier_mismatch[
    ["hub_id", "segment", "hub_city", "hub_tier"]
]

tier_mismatch.to_csv(
    OUTPUT_DIR / "tier_vs_segment_mismatch.csv",
    index=False
)


# ------------------------------------------------------------
# 4.6 Display results and validate reference values
# ------------------------------------------------------------

print("\nHub segments:")
print(hub_segments.to_string(index=False))

print("\nSegment profile:")
print(segment_profile.to_string(index=False))

print("\nTier versus segment mismatches:")
print(tier_mismatch.to_string(index=False))


# Expected segment sizes
expected_sizes = {
    "Standard": 7,
    "Watch": 8,
    "Escalate": 9
}

actual_sizes = dict(
    zip(segment_profile["segment"], segment_profile["hubs"])
)

assert actual_sizes == expected_sizes

# Expected mismatch hubs
expected_mismatch_hubs = {
    "HUB07",
    "HUB13",
    "HUB15",
    "HUB22"
}

actual_mismatch_hubs = set(tier_mismatch["hub_id"])

assert actual_mismatch_hubs == expected_mismatch_hubs

print("\nPart D - Step 4 completed.")
