from pathlib import Path

import pandas as pd


# ==============================================================
# PATHS
# ==============================================================

INPUT_PATH = Path(
    "data/processed/ml/customer_features.csv"
)

OUTPUT_DIR = Path(
    "data/processed/ml/splits"
)


# ==============================================================
# REQUIRED COLUMNS
# ==============================================================

REQUIRED_COLUMNS = {
    "customer_id",
    "observation_date",
    "target",
}


# ==============================================================
# VALIDATION
# ==============================================================

def validate_dataset(
    df: pd.DataFrame,
) -> None:

    print("\n" + "=" * 70)
    print("ML DATASET VALIDATION")
    print("=" * 70)

    # ----------------------------------------------------------
    # Required columns
    # ----------------------------------------------------------

    missing_columns = (
        REQUIRED_COLUMNS
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    print("[PASS] Required columns present")

    # ----------------------------------------------------------
    # Customer ID
    # ----------------------------------------------------------

    customer_id_nulls = (
        df["customer_id"]
        .isna()
        .sum()
    )

    if customer_id_nulls > 0:
        raise ValueError(
            f"customer_id contains "
            f"{customer_id_nulls} null values"
        )

    print(
        f"[PASS] customer_id nulls: "
        f"{customer_id_nulls}"
    )

    # ----------------------------------------------------------
    # Observation date
    # ----------------------------------------------------------

    df["observation_date"] = pd.to_datetime(
        df["observation_date"],
        errors="coerce",
    )

    invalid_dates = (
        df["observation_date"]
        .isna()
        .sum()
    )

    if invalid_dates > 0:
        raise ValueError(
            f"Invalid observation dates: "
            f"{invalid_dates}"
        )

    print(
        f"[PASS] observation_date invalid: "
        f"{invalid_dates}"
    )

    # ----------------------------------------------------------
    # Target
    # ----------------------------------------------------------

    target_nulls = (
        df["target"]
        .isna()
        .sum()
    )

    if target_nulls > 0:
        raise ValueError(
            f"target contains "
            f"{target_nulls} null values"
        )

    print(
        f"[PASS] target nulls: "
        f"{target_nulls}"
    )

    # ----------------------------------------------------------
    # Target values
    # ----------------------------------------------------------

    invalid_targets = (
        ~df["target"].isin([0, 1])
    ).sum()

    if invalid_targets > 0:
        raise ValueError(
            f"Invalid target values: "
            f"{invalid_targets}"
        )

    print(
        f"[PASS] target values: "
        f"invalid={invalid_targets}"
    )

    # ----------------------------------------------------------
    # Both classes must exist
    # ----------------------------------------------------------

    unique_targets = sorted(
        df["target"].unique().tolist()
    )

    if unique_targets != [0, 1]:
        raise ValueError(
            "Dataset must contain both "
            "target classes 0 and 1"
        )

    print(
        "[PASS] Both target classes present"
    )

    # ----------------------------------------------------------
    # Duplicate customer + observation snapshot
    # ----------------------------------------------------------

    duplicate_snapshots = (
        df[
            ["customer_id", "observation_date"]
        ]
        .duplicated()
        .sum()
    )

    if duplicate_snapshots > 0:
        raise ValueError(
            "Duplicate customer observation "
            f"snapshots found: {duplicate_snapshots}"
        )

    print(
        f"[PASS] Duplicate customer snapshots: "
        f"{duplicate_snapshots}"
    )


# ==============================================================
# TIME-BASED SPLIT
# ==============================================================

def create_time_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    # ----------------------------------------------------------
    # Sort chronologically
    # ----------------------------------------------------------

    df = df.sort_values(
        [
            "observation_date",
            "customer_id",
        ]
    ).reset_index(drop=True)

    unique_dates = sorted(
        df["observation_date"]
        .dropna()
        .unique()
    )

    if len(unique_dates) < 4:
        raise ValueError(
            "Not enough observation dates "
            "to create a reliable time split."
        )

    # ----------------------------------------------------------
    # Use approximately 75% historical dates for training
    # and the remaining dates for testing.
    #
    # For our current dataset:
    #
    # Apr, May, Jun, Jul, Aug, Sep -> TRAIN
    # Oct, Nov                    -> TEST
    # ----------------------------------------------------------

    split_index = int(
        len(unique_dates) * 0.75
    )

    # Make sure we don't create an empty split.
    split_index = max(
        1,
        min(
            split_index,
            len(unique_dates) - 1,
        ),
    )

    train_dates = unique_dates[
        :split_index
    ]

    test_dates = unique_dates[
        split_index:
    ]

    train_df = df[
        df["observation_date"].isin(
            train_dates
        )
    ].copy()

    test_df = df[
        df["observation_date"].isin(
            test_dates
        )
    ].copy()

    return train_df, test_df


# ==============================================================
# VERIFY TEMPORAL SEPARATION
# ==============================================================

def verify_time_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:

    train_max_date = (
        train_df["observation_date"].max()
    )

    test_min_date = (
        test_df["observation_date"].min()
    )

    print("\n" + "=" * 70)
    print("TIME SPLIT VALIDATION")
    print("=" * 70)

    print(
        f"Training date range: "
        f"{train_df['observation_date'].min().date()} "
        f"→ "
        f"{train_max_date.date()}"
    )

    print(
        f"Testing date range:  "
        f"{test_df['observation_date'].min().date()} "
        f"→ "
        f"{test_df['observation_date'].max().date()}"
    )

    # ----------------------------------------------------------
    # Critical leakage check:
    # no test observation can occur before or on the latest
    # training observation date.
    # ----------------------------------------------------------

    if train_max_date >= test_min_date:
        raise ValueError(
            "Temporal leakage detected: "
            "training data reaches the test period."
        )

    print(
        "[PASS] Training data occurs strictly "
        "before testing data"
    )

    # ----------------------------------------------------------
    # Ensure neither split is empty
    # ----------------------------------------------------------

    if train_df.empty:
        raise ValueError(
            "Training dataset is empty."
        )

    if test_df.empty:
        raise ValueError(
            "Testing dataset is empty."
        )

    print(
        f"[PASS] Training rows: "
        f"{len(train_df):,}"
    )

    print(
        f"[PASS] Testing rows: "
        f"{len(test_df):,}"
    )


# ==============================================================
# TARGET DISTRIBUTION
# ==============================================================

def print_target_distribution(
    name: str,
    df: pd.DataFrame,
) -> None:

    print(
        f"\n{name} target distribution:"
    )

    counts = (
        df["target"]
        .value_counts()
        .sort_index()
    )

    for target, count in counts.items():

        percentage = (
            count
            / len(df)
            * 100
        )

        label = (
            "HIGH RISK"
            if target == 1
            else "NOT HIGH RISK"
        )

        print(
            f"  {target} ({label}): "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )


# ==============================================================
# SAVE DATASETS
# ==============================================================

def save_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = (
        OUTPUT_DIR / "train.csv"
    )

    test_path = (
        OUTPUT_DIR / "test.csv"
    )

    train_df.to_csv(
        train_path,
        index=False,
    )

    test_df.to_csv(
        test_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("ML DATASETS SAVED")
    print("=" * 70)

    print(
        f"Training dataset: {train_path}"
    )

    print(
        f"Testing dataset:  {test_path}"
    )


# ==============================================================
# MAIN
# ==============================================================

def main() -> None:

    print("\n" + "=" * 70)
    print("CUSTOMER RISK INTELLIGENCE")
    print("ML DATASET PREPARATION")
    print("=" * 70)

    # ----------------------------------------------------------
    # Load feature dataset
    # ----------------------------------------------------------

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: "
            f"{INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    print(
        f"\nLoaded rows: "
        f"{len(df):,}"
    )

    print(
        f"Loaded columns: "
        f"{len(df.columns):,}"
    )

    # ----------------------------------------------------------
    # Validate
    # ----------------------------------------------------------

    validate_dataset(df)

    # ----------------------------------------------------------
    # Create chronological split
    # ----------------------------------------------------------

    train_df, test_df = create_time_split(
        df
    )

    # ----------------------------------------------------------
    # Verify split
    # ----------------------------------------------------------

    verify_time_split(
        train_df,
        test_df,
    )

    # ----------------------------------------------------------
    # Print target distribution
    # ----------------------------------------------------------

    print_target_distribution(
        "TRAIN",
        train_df,
    )

    print_target_distribution(
        "TEST",
        test_df,
    )

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    save_datasets(
        train_df,
        test_df,
    )

    print("\n" + "=" * 70)
    print("ML DATASET PREPARATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
