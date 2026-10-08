from pathlib import Path

import pandas as pd


RAW_DATA_DIR = Path("data/raw")


CUSTOMER_FILE = (
    RAW_DATA_DIR
    / "customers"
    / "customers.csv"
)

TRANSACTION_FILE = (
    RAW_DATA_DIR
    / "transactions"
    / "transactions.csv"
)

INTERACTION_FILE = (
    RAW_DATA_DIR
    / "interactions"
    / "interactions.csv"
)

DOCUMENT_FILE = (
    RAW_DATA_DIR
    / "documents"
    / "documents.csv"
)


# =========================================================
# Allowed business values
# =========================================================

VALID_ACCOUNT_STATUSES = {
    "ACTIVE",
    "INACTIVE",
    "SUSPENDED",
}

VALID_PAYMENT_STATUSES = {
    "PAID",
    "PENDING",
    "OVERDUE",
    "FAILED",
}

VALID_INTERACTION_TYPES = {
    "SUPPORT_REQUEST",
    "COMPLAINT",
    "PAYMENT_QUERY",
    "RENEWAL",
    "TECHNICAL_ISSUE",
    "ACCOUNT_QUERY",
    "FEEDBACK",
    "ESCALATION",
}

VALID_CHANNELS = {
    "EMAIL",
    "PHONE",
    "CHAT",
    "PORTAL",
}

VALID_SENTIMENTS = {
    "POSITIVE",
    "NEUTRAL",
    "NEGATIVE",
}

VALID_RESOLUTION_STATUSES = {
    "RESOLVED",
    "PENDING",
    "ESCALATED",
}

VALID_DOCUMENT_TYPES = {
    "COMPLAINT",
    "PAYMENT_ISSUE",
    "SUPPORT_NOTE",
    "RENEWAL_DISCUSSION",
    "ESCALATION",
    "ACCOUNT_NOTE",
    "CUSTOMER_PREFERENCE",
}


# =========================================================
# Validation helper
# =========================================================

def print_check(
    check_name: str,
    passed: bool,
    details: str,
) -> bool:

    status = "PASS" if passed else "FAIL"

    print(
        f"[{status}] "
        f"{check_name}: "
        f"{details}"
    )

    return passed


# =========================================================
# Customer validation
# =========================================================

def validate_customers() -> dict:

    print()
    print("=" * 70)
    print("CUSTOMER VALIDATION")
    print("=" * 70)

    df = pd.read_csv(
        CUSTOMER_FILE
    )

    results = {}

    # -----------------------------------------------------
    # Required columns
    # -----------------------------------------------------

    required_columns = {
        "customer_id",
        "customer_name",
        "customer_since",
        "customer_segment",
        "industry",
        "country",
        "account_status",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    results["required_columns"] = print_check(
        "Required columns",
        not missing_columns,
        (
            "all present"
            if not missing_columns
            else f"missing={missing_columns}"
        ),
    )

    # -----------------------------------------------------
    # Customer ID nulls
    # -----------------------------------------------------

    null_ids = int(
        df["customer_id"].isna().sum()
    )

    results["customer_id_nulls"] = print_check(
        "Customer ID nulls",
        null_ids == 0,
        f"count={null_ids}",
    )

    # -----------------------------------------------------
    # Duplicate customer IDs
    # -----------------------------------------------------

    duplicate_ids = int(
        df["customer_id"]
        .duplicated()
        .sum()
    )

    results["duplicate_customer_ids"] = print_check(
        "Duplicate customer IDs",
        duplicate_ids == 0,
        f"count={duplicate_ids}",
    )

    # -----------------------------------------------------
    # Customer date validation
    # -----------------------------------------------------

    customer_dates = pd.to_datetime(
        df["customer_since"],
        errors="coerce",
    )

    invalid_dates = int(
        customer_dates.isna().sum()
    )

    results["invalid_customer_dates"] = print_check(
        "Customer since dates",
        invalid_dates == 0,
        f"invalid={invalid_dates}",
    )

    # -----------------------------------------------------
    # Account status
    # -----------------------------------------------------

    invalid_statuses = int(
        (
            df["account_status"].notna()
            & ~df["account_status"].isin(
                VALID_ACCOUNT_STATUSES
            )
        ).sum()
    )

    results["invalid_account_status"] = print_check(
        "Account status",
        invalid_statuses == 0,
        f"invalid={invalid_statuses}",
    )

    # -----------------------------------------------------
    # Required business fields
    # -----------------------------------------------------

    required_fields = [
        "customer_name",
        "customer_segment",
        "industry",
        "country",
    ]

    for field in required_fields:

        null_count = int(
            df[field].isna().sum()
        )

        results[f"{field}_nulls"] = print_check(
            f"{field} nulls",
            null_count == 0,
            f"count={null_count}",
        )

    return results


# =========================================================
# Transaction validation
# =========================================================

def validate_transactions(
    valid_customer_ids: set,
) -> dict:

    print()
    print("=" * 70)
    print("TRANSACTION VALIDATION")
    print("=" * 70)

    df = pd.read_csv(
        TRANSACTION_FILE
    )

    results = {}

    # -----------------------------------------------------
    # Required columns
    # -----------------------------------------------------

    required_columns = {
        "transaction_id",
        "customer_id",
        "transaction_date",
        "transaction_amount",
        "payment_status",
        "due_date",
        "payment_date",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    results["required_columns"] = print_check(
        "Required columns",
        not missing_columns,
        (
            "all present"
            if not missing_columns
            else f"missing={missing_columns}"
        ),
    )

    # -----------------------------------------------------
    # Transaction ID
    # -----------------------------------------------------

    null_ids = int(
        df["transaction_id"].isna().sum()
    )

    duplicate_ids = int(
        df["transaction_id"]
        .duplicated()
        .sum()
    )

    results["transaction_id_nulls"] = print_check(
        "Transaction ID nulls",
        null_ids == 0,
        f"count={null_ids}",
    )

    results["duplicate_transaction_ids"] = print_check(
        "Duplicate transaction IDs",
        duplicate_ids == 0,
        f"count={duplicate_ids}",
    )

    # -----------------------------------------------------
    # Customer referential integrity
    # -----------------------------------------------------

    invalid_customer_ids = int(
        (
            df["customer_id"].notna()
            & ~df["customer_id"].isin(
                valid_customer_ids
            )
        ).sum()
    )

    results["invalid_customer_ids"] = print_check(
        "Customer referential integrity",
        invalid_customer_ids == 0,
        f"invalid={invalid_customer_ids}",
    )

    # -----------------------------------------------------
    # Transaction date
    # -----------------------------------------------------

    transaction_dates = pd.to_datetime(
        df["transaction_date"],
        errors="coerce",
    )

    invalid_transaction_dates = int(
        transaction_dates.isna().sum()
    )

    results["invalid_transaction_dates"] = print_check(
        "Transaction dates",
        invalid_transaction_dates == 0,
        f"invalid={invalid_transaction_dates}",
    )

    # -----------------------------------------------------
    # Due date
    # -----------------------------------------------------

    due_dates = pd.to_datetime(
        df["due_date"],
        errors="coerce",
    )

    invalid_due_dates = int(
        due_dates.isna().sum()
    )

    results["invalid_due_dates"] = print_check(
        "Due dates",
        invalid_due_dates == 0,
        f"invalid={invalid_due_dates}",
    )

    # -----------------------------------------------------
    # Date relationship
    # -----------------------------------------------------

    invalid_date_order = int(
        (
            transaction_dates.notna()
            & due_dates.notna()
            & (
                due_dates
                < transaction_dates
            )
        ).sum()
    )

    results["invalid_date_order"] = print_check(
        "Transaction date <= due date",
        invalid_date_order == 0,
        f"invalid={invalid_date_order}",
    )

    # -----------------------------------------------------
    # Transaction amount
    # -----------------------------------------------------

    df["transaction_amount"] = pd.to_numeric(
        df["transaction_amount"],
        errors="coerce",
    )

    null_amounts = int(
        df["transaction_amount"]
        .isna()
        .sum()
    )

    invalid_amounts = int(
        (
            df["transaction_amount"]
            <= 0
        ).sum()
    )

    results["null_transaction_amount"] = print_check(
        "Transaction amount nulls",
        null_amounts == 0,
        f"count={null_amounts}",
    )

    results["invalid_transaction_amount"] = print_check(
        "Transaction amount > 0",
        invalid_amounts == 0,
        f"invalid={invalid_amounts}",
    )

    # -----------------------------------------------------
    # Payment status
    # -----------------------------------------------------

    invalid_statuses = int(
        (
            df["payment_status"].notna()
            & ~df["payment_status"].isin(
                VALID_PAYMENT_STATUSES
            )
        ).sum()
    )

    missing_statuses = int(
        df["payment_status"].isna().sum()
    )

    results["missing_payment_status"] = print_check(
        "Payment status nulls",
        missing_statuses == 0,
        f"count={missing_statuses}",
    )

    results["invalid_payment_status"] = print_check(
        "Payment status values",
        invalid_statuses == 0,
        f"invalid={invalid_statuses}",
    )

    # -----------------------------------------------------
    # Payment date
    # -----------------------------------------------------

    payment_dates = pd.to_datetime(
        df["payment_date"],
        errors="coerce",
    )

    invalid_payment_dates = int(
        (
            df["payment_date"].notna()
            & payment_dates.isna()
        ).sum()
    )

    results["invalid_payment_dates"] = print_check(
        "Payment dates",
        invalid_payment_dates == 0,
        f"invalid={invalid_payment_dates}",
    )

    return results


# =========================================================
# Interaction validation
# =========================================================

def validate_interactions(
    valid_customer_ids: set,
) -> dict:

    print()
    print("=" * 70)
    print("INTERACTION VALIDATION")
    print("=" * 70)

    df = pd.read_csv(
        INTERACTION_FILE
    )

    results = {}

    # -----------------------------------------------------
    # Required columns
    # -----------------------------------------------------

    required_columns = {
        "interaction_id",
        "customer_id",
        "interaction_date",
        "interaction_type",
        "channel",
        "sentiment",
        "resolution_status",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    results["required_columns"] = print_check(
        "Required columns",
        not missing_columns,
        (
            "all present"
            if not missing_columns
            else f"missing={missing_columns}"
        ),
    )

    # -----------------------------------------------------
    # ID validation
    # -----------------------------------------------------

    null_ids = int(
        df["interaction_id"].isna().sum()
    )

    duplicate_ids = int(
        df["interaction_id"]
        .duplicated()
        .sum()
    )

    results["interaction_id_nulls"] = print_check(
        "Interaction ID nulls",
        null_ids == 0,
        f"count={null_ids}",
    )

    results["duplicate_interaction_ids"] = print_check(
        "Duplicate interaction IDs",
        duplicate_ids == 0,
        f"count={duplicate_ids}",
    )

    # -----------------------------------------------------
    # Referential integrity
    # -----------------------------------------------------

    invalid_customer_ids = int(
        (
            df["customer_id"].notna()
            & ~df["customer_id"].isin(
                valid_customer_ids
            )
        ).sum()
    )

    results["invalid_customer_ids"] = print_check(
        "Customer referential integrity",
        invalid_customer_ids == 0,
        f"invalid={invalid_customer_ids}",
    )

    # -----------------------------------------------------
    # Interaction date
    # -----------------------------------------------------

    interaction_dates = pd.to_datetime(
        df["interaction_date"],
        errors="coerce",
    )

    invalid_dates = int(
        interaction_dates.isna().sum()
    )

    results["invalid_interaction_dates"] = print_check(
        "Interaction dates",
        invalid_dates == 0,
        f"invalid={invalid_dates}",
    )

    # -----------------------------------------------------
    # Interaction type
    # -----------------------------------------------------

    invalid_types = int(
        (
            df["interaction_type"].notna()
            & ~df["interaction_type"].isin(
                VALID_INTERACTION_TYPES
            )
        ).sum()
    )

    results["invalid_interaction_types"] = print_check(
        "Interaction types",
        invalid_types == 0,
        f"invalid={invalid_types}",
    )

    # -----------------------------------------------------
    # Channel
    # -----------------------------------------------------

    invalid_channels = int(
        (
            df["channel"].notna()
            & ~df["channel"].isin(
                VALID_CHANNELS
            )
        ).sum()
    )

    results["invalid_channels"] = print_check(
        "Interaction channels",
        invalid_channels == 0,
        f"invalid={invalid_channels}",
    )

    # -----------------------------------------------------
    # Sentiment
    # -----------------------------------------------------

    missing_sentiment = int(
        df["sentiment"].isna().sum()
    )

    invalid_sentiment = int(
        (
            df["sentiment"].notna()
            & ~df["sentiment"].isin(
                VALID_SENTIMENTS
            )
        ).sum()
    )

    results["missing_sentiment"] = print_check(
        "Sentiment nulls",
        missing_sentiment == 0,
        f"count={missing_sentiment}",
    )

    results["invalid_sentiment"] = print_check(
        "Sentiment values",
        invalid_sentiment == 0,
        f"invalid={invalid_sentiment}",
    )

    # -----------------------------------------------------
    # Resolution status
    # -----------------------------------------------------

    invalid_resolution = int(
        (
            df["resolution_status"].notna()
            & ~df["resolution_status"].isin(
                VALID_RESOLUTION_STATUSES
            )
        ).sum()
    )

    results["invalid_resolution_status"] = print_check(
        "Resolution status",
        invalid_resolution == 0,
        f"invalid={invalid_resolution}",
    )

    return results


# =========================================================
# Document validation
# =========================================================

def validate_documents(
    valid_customer_ids: set,
) -> dict:

    print()
    print("=" * 70)
    print("DOCUMENT VALIDATION")
    print("=" * 70)

    df = pd.read_csv(
        DOCUMENT_FILE
    )

    results = {}

    # -----------------------------------------------------
    # Required columns
    # -----------------------------------------------------

    required_columns = {
        "document_id",
        "customer_id",
        "document_type",
        "document_date",
        "source",
        "content",
        "is_active",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    results["required_columns"] = print_check(
        "Required columns",
        not missing_columns,
        (
            "all present"
            if not missing_columns
            else f"missing={missing_columns}"
        ),
    )

    # -----------------------------------------------------
    # Document ID
    # -----------------------------------------------------

    null_ids = int(
        df["document_id"].isna().sum()
    )

    duplicate_ids = int(
        df["document_id"]
        .duplicated()
        .sum()
    )

    results["document_id_nulls"] = print_check(
        "Document ID nulls",
        null_ids == 0,
        f"count={null_ids}",
    )

    results["duplicate_document_ids"] = print_check(
        "Duplicate document IDs",
        duplicate_ids == 0,
        f"count={duplicate_ids}",
    )

    # -----------------------------------------------------
    # Customer reference
    # -----------------------------------------------------

    invalid_customer_ids = int(
        (
            df["customer_id"].notna()
            & ~df["customer_id"].isin(
                valid_customer_ids
            )
        ).sum()
    )

    results["invalid_customer_ids"] = print_check(
        "Customer referential integrity",
        invalid_customer_ids == 0,
        f"invalid={invalid_customer_ids}",
    )

    # -----------------------------------------------------
    # Document date
    # -----------------------------------------------------

    document_dates = pd.to_datetime(
        df["document_date"],
        errors="coerce",
    )

    invalid_dates = int(
        document_dates.isna().sum()
    )

    results["invalid_document_dates"] = print_check(
        "Document dates",
        invalid_dates == 0,
        f"invalid={invalid_dates}",
    )

    # -----------------------------------------------------
    # Document type
    # -----------------------------------------------------

    invalid_types = int(
        (
            df["document_type"].notna()
            & ~df["document_type"].isin(
                VALID_DOCUMENT_TYPES
            )
        ).sum()
    )

    results["invalid_document_types"] = print_check(
        "Document types",
        invalid_types == 0,
        f"invalid={invalid_types}",
    )

    # -----------------------------------------------------
    # Source
    # -----------------------------------------------------

    missing_source = int(
        df["source"].isna().sum()
    )

    results["missing_source"] = print_check(
        "Document source nulls",
        missing_source == 0,
        f"count={missing_source}",
    )

    # -----------------------------------------------------
    # Content
    # -----------------------------------------------------

    missing_content = int(
        df["content"]
        .fillna("")
        .str.strip()
        .eq("")
        .sum()
    )

    results["missing_content"] = print_check(
        "Document content",
        missing_content == 0,
        f"empty={missing_content}",
    )

    return results


# =========================================================
# Main validation pipeline
# =========================================================

def run_validation() -> dict:

    print()
    print("=" * 70)
    print("CUSTOMER RISK INTELLIGENCE")
    print("RAW DATA VALIDATION")
    print("=" * 70)

    # -----------------------------------------------------
    # Customer master is the reference dataset
    # -----------------------------------------------------

    customers = pd.read_csv(
        CUSTOMER_FILE
    )

    valid_customer_ids = set(
        customers["customer_id"]
        .dropna()
    )

    # -----------------------------------------------------
    # Validate every dataset
    # -----------------------------------------------------

    results = {}

    results["customers"] = (
        validate_customers()
    )

    results["transactions"] = (
        validate_transactions(
            valid_customer_ids
        )
    )

    results["interactions"] = (
        validate_interactions(
            valid_customer_ids
        )
    )

    results["documents"] = (
        validate_documents(
            valid_customer_ids
        )
    )

    return results


def main() -> None:

    results = run_validation()

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    total_checks = 0
    failed_checks = 0

    for dataset, checks in results.items():

        dataset_failed = 0

        for check_name, passed in checks.items():

            total_checks += 1

            if not passed:
                failed_checks += 1
                dataset_failed += 1

        status = (
            "VALID"
            if dataset_failed == 0
            else "INVALID"
        )

        print(
            f"{dataset:<15}"
            f"{status:<10}"
            f"failed_checks={dataset_failed}"
        )

    print()
    print(
        f"Total checks: {total_checks}"
    )

    print(
        f"Failed checks: {failed_checks}"
    )

    print()

    if failed_checks > 0:
        print(
            "Validation detected data-quality issues."
        )
    else:
        print(
            "All validation checks passed."
        )


if __name__ == "__main__":
    main()
