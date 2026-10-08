from pathlib import Path

import pandas as pd


RAW_DATA_DIR = Path("data/raw")


DATASETS = {
    "customers": RAW_DATA_DIR / "customers" / "customers.csv",
    "transactions": RAW_DATA_DIR / "transactions" / "transactions.csv",
    "interactions": RAW_DATA_DIR / "interactions" / "interactions.csv",
    "documents": RAW_DATA_DIR / "documents" / "documents.csv",
}


def profile_dataset(
    name: str,
    file_path: Path,
) -> dict:

    print()
    print("=" * 70)
    print(f"DATASET PROFILE: {name.upper()}")
    print("=" * 70)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    # ---------------------------------------------------------
    # Basic information
    # ---------------------------------------------------------

    row_count = len(df)
    column_count = len(df.columns)

    print(f"File: {file_path}")
    print(f"Rows: {row_count:,}")
    print(f"Columns: {column_count}")

    # ---------------------------------------------------------
    # Column information
    # ---------------------------------------------------------

    print()
    print("Column information:")

    for column in df.columns:

        dtype = str(df[column].dtype)

        null_count = int(
            df[column].isna().sum()
        )

        null_percentage = (
            null_count / row_count * 100
            if row_count > 0
            else 0
        )

        unique_count = int(
            df[column].nunique(
                dropna=True
            )
        )

        print(
            f"  {column:<25}"
            f"dtype={dtype:<12}"
            f"nulls={null_count:<6}"
            f"null%={null_percentage:>6.2f}%"
            f"unique={unique_count}"
        )

    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------

    missing_values = (
        df.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing_values = (
        missing_values[
            missing_values > 0
        ]
        .to_dict()
    )

    print()
    print("Missing values:")

    if missing_values:
        for column, count in missing_values.items():

            percentage = (
                count / row_count * 100
            )

            print(
                f"  {column}: "
                f"{count:,} "
                f"({percentage:.2f}%)"
            )
    else:
        print("  None")

    # ---------------------------------------------------------
    # Duplicate rows
    # ---------------------------------------------------------

    duplicate_rows = int(
        df.duplicated().sum()
    )

    duplicate_percentage = (
        duplicate_rows / row_count * 100
        if row_count > 0
        else 0
    )

    print()
    print(
        f"Duplicate complete rows: "
        f"{duplicate_rows:,} "
        f"({duplicate_percentage:.2f}%)"
    )

    # ---------------------------------------------------------
    # Duplicate IDs
    # ---------------------------------------------------------

    id_columns = [
        column
        for column in df.columns
        if column.endswith("_id")
    ]

    duplicate_id_counts = {}

    for column in id_columns:

        duplicate_count = int(
            df[column]
            .duplicated()
            .sum()
        )

        duplicate_id_counts[column] = (
            duplicate_count
        )

        print(
            f"Duplicate values in {column}: "
            f"{duplicate_count:,}"
        )

    # ---------------------------------------------------------
    # Numeric distributions
    # ---------------------------------------------------------

    numeric_columns = (
        df.select_dtypes(
            include="number"
        ).columns
    )

    if len(numeric_columns) > 0:

        print()
        print("Numeric column statistics:")

        for column in numeric_columns:

            series = df[column]

            print(
                f"\n  {column}"
            )

            print(
                f"    min    = "
                f"{series.min()}"
            )

            print(
                f"    max    = "
                f"{series.max()}"
            )

            print(
                f"    mean   = "
                f"{series.mean():.2f}"
            )

            print(
                f"    median = "
                f"{series.median():.2f}"
            )

    # ---------------------------------------------------------
    # Categorical distributions
    # ---------------------------------------------------------

    categorical_columns = (
        df.select_dtypes(
            include=["str", "object", "category"]
        ).columns
    )

    if len(categorical_columns) > 0:

        print()
        print(
            "Categorical distributions:"
        )

        for column in categorical_columns:

            print()
            print(
                f"  {column}:"
            )

            distribution = (
                df[column]
                .value_counts(
                    dropna=False
                )
                .head(15)
            )

            for value, count in (
                distribution.items()
            ):

                print(
                    f"    {str(value):<30}"
                    f"{count:,}"
                )

    # ---------------------------------------------------------
    # Build profile result
    # ---------------------------------------------------------

    profile = {
        "dataset": name,
        "file": str(file_path),
        "rows": row_count,
        "columns": column_count,
        "column_names": list(df.columns),
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "duplicate_percentage": duplicate_percentage,
        "duplicate_id_counts": duplicate_id_counts,
    }

    return profile


def generate_profile_report() -> dict:

    print()
    print("=" * 70)
    print("CUSTOMER RISK INTELLIGENCE")
    print("RAW DATA PROFILING")
    print("=" * 70)

    profiles = {}

    for name, file_path in DATASETS.items():

        profiles[name] = profile_dataset(
            name=name,
            file_path=file_path,
        )

    return profiles


def main() -> None:

    profiles = generate_profile_report()

    print()
    print("=" * 70)
    print("RAW DATA PROFILING COMPLETED")
    print("=" * 70)

    print()
    print("Summary:")

    for name, profile in profiles.items():

        print(
            f"{name:<15}"
            f"rows={profile['rows']:<8,}"
            f"columns={profile['columns']:<3}"
            f"duplicates={profile['duplicate_rows']:,}"
        )


if __name__ == "__main__":
    main()
