PROMISED_HOURS = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96
}

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

NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no"
]

def clean_shipments(df, medians=None):
    df = df.copy()

    df = df.drop_duplicates().reset_index(drop=True)

    df["service_type"] = (
        df["service_type"]
        .astype("string")
        .str.strip()
        .str.title()
    )

    df["payment_mode"] = (
        df["payment_mode"]
        .fillna("Unknown")
        .astype("string")
        .str.strip()
        .str.title()
    )

    df["declared_value"] = pd.to_numeric(
        df["declared_value"]
        .astype("string")
        .str.replace(r"[₹,]", "", regex=True),
        errors="coerce",
    )

    df["dispatch_time"] = pd.to_datetime(
        df["dispatch_time"],
        errors="coerce",
    )

    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    median_columns = [
        "distance_km",
        "parcel_weight_kg",
        "declared_value_inr"
    ]

    if medians is None:
        medians = {
            column: float(df[column].median())
            for column in median_columns
        }
    else:
        expected_keys = set(median_columns)

        if set(medians.keys()) != expected_keys:
            raise ValueError(
                f"medians must contain exactly these keys: {median_columns}"
            )

        medians = {
            column: float(medians[column])
            for column in median_columns
        }
    
    for column in median_columns:
        clean_df[column] = clean_df[column].fillna(medians[column])


    return df, medians