from pathlib import Path

import pandas as pd


# ==============================================================
# PATH CONFIGURATION
# ==============================================================

PROCESSED_DIR = Path("data/processed")


# ==============================================================
# VALID VALUES AFTER CLEANING
# ==============================================================

VALID_ACCOUNT_STATUSES = {
    "ACTIVE",
    "INACTIVE",
    "SUSPENDED",
    "UNKNOWN",
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

VALID_INTERACTION_CHANNELS = {
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
    "UNKNOWN",
}


# ==============================================================
# VALIDATION RESULT TRACKING
# ==============================================================

class ValidationTracker:

    def __init__(self):
        self.total_checks = 0
        self.failed_checks = 0

    def check(
        self,
        condition: bool,
        name: str,
        details: str,
    ) -> None:
        """
        Register and print one validation check.
        """

        self.total_checks += 1

        if condition:
            print(
                f"[PASS] {name}: {details}"
            )
        else:
            self.failed_checks += 1

            print(
                f"[FAIL] {name}: {details}"
            )


# ==============================================================
# GENERIC HELPERS
# ==============================================================

def load_dataset(
    dataset_name: str,
) -> pd.DataFrame:

    path = (
        PROCESSED_DIR
        / dataset_name
        / f"{dataset_name}_clean.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}"
        )

    return pd.read_csv(path)


def validate_required_columns(
    tracker: ValidationTracker,
    df: pd.DataFrame,
    required_columns: list[str],
) -> None:

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    tracker.check(
        len(missing_columns) == 0,
        "Required columns",
        (
            "all present"
            if not missing_columns
            else f"missing={missing_columns}"
        ),
    )


def validate_no_nulls(
    tracker: ValidationTracker,
    df: pd.DataFrame,
    column: str,
) -> None:

    null_count = df[column].isna().sum()

    tracker.check(
        null_count == 0,
        f"{column} nulls",
        f"count={null_count}",
    )


def validate_no_duplicates(
    tracker: ValidationTracker,
    df: pd.DataFrame,
    column: str,
) -> None:

    duplicate_count = (
        df[column]
        .duplicated()
        .sum()
    )

    tracker.check(
        duplicate_count == 0,
        f"Duplicate {column}",
        f"count={duplicate_count}",
    )


def validate_datetime_column(
    tracker: ValidationTracker,
    df: pd.DataFrame,
    column: str,
) -> None:

    parsed = pd.to_datetime(
        df[column],
        errors="coerce",
    )

    invalid_count = parsed.isna().sum()

    tracker.check(
        invalid_count == 0,
        f"{column} dates",
        f"invalid={invalid_count}",
    )


# ==============================================================
# CUSTOMER VALIDATION
# ==============================================================

def validate_customers(
    tracker: ValidationTracker,
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("PROCESSED CUSTOMER VALIDATION")
    print("=" * 70)

    df = load_dataset("customers")

    required_columns = [
        "customer_id",
        "customer_name",
        "customer_since",
        "customer_segment",
        "industry",
        "country",
        "account_status",
    ]

    validate_required_columns(
        tracker,
        df,
        required_columns,
    )

    validate_no_nulls(
        tracker,
        df,
        "customer_id",
    )

    validate_no_duplicates(
        tracker,
        df,
        "customer_id",
    )

    validate_datetime_column(
        tracker,
        df,
        "customer_since",
    )

    validate_no_nulls(
        tracker,
        df,
        "customer_name",
    )

    validate_no_nulls(
        tracker,
        df,
        "customer_segment",
    )

    validate_no_nulls(
        tracker,
        df,
        "industry",
    )

    validate_no_nulls(
        tracker,
        df,
        "country",
    )

    invalid_status_count = (
        ~df["account_status"].isin(
            VALID_ACCOUNT_STATUSES
        )
    ).sum()

    tracker.check(
        invalid_status_count == 0,
        "Account status values",
        f"invalid={invalid_status_count}",
    )

    return df


# ==============================================================
# TRANSACTION VALIDATION
# ==============================================================

def validate_transactions(
    tracker: ValidationTracker,
    customers: pd.DataFrame,
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("PROCESSED TRANSACTION VALIDATION")
    print("=" * 70)

    df = load_dataset("transactions")

    required_columns = [
        "transaction_id",
        "customer_id",
        "transaction_date",
        "transaction_amount",
        "payment_status",
        "due_date",
        "payment_date",
    ]

    validate_required_columns(
        tracker,
        df,
        required_columns,
    )

    validate_no_nulls(
        tracker,
        df,
        "transaction_id",
    )

    validate_no_duplicates(
        tracker,
        df,
        "transaction_id",
    )

    # ----------------------------------------------------------
    # Customer referential integrity
    # ----------------------------------------------------------

    valid_customer_ids = set(
        customers["customer_id"]
    )

    invalid_customer_count = (
        ~df["customer_id"].isin(
            valid_customer_ids
        )
    ).sum()

    tracker.check(
        invalid_customer_count == 0,
        "Customer referential integrity",
        f"invalid={invalid_customer_count}",
    )

    # ----------------------------------------------------------
    # Date validation
    # ----------------------------------------------------------

    transaction_dates = pd.to_datetime(
        df["transaction_date"],
        errors="coerce",
    )

    due_dates = pd.to_datetime(
        df["due_date"],
        errors="coerce",
    )

    payment_dates = pd.to_datetime(
        df["payment_date"],
        errors="coerce",
    )

    tracker.check(
        transaction_dates.isna().sum() == 0,
        "Transaction dates",
        f"invalid={transaction_dates.isna().sum()}",
    )

    tracker.check(
        due_dates.isna().sum() == 0,
        "Due dates",
        f"invalid={due_dates.isna().sum()}",
    )

    # ----------------------------------------------------------
    # Transaction date <= due date
    # ----------------------------------------------------------

    invalid_order_count = (
        transaction_dates > due_dates
    ).sum()

    tracker.check(
        invalid_order_count == 0,
        "Transaction date <= due date",
        f"invalid={invalid_order_count}",
    )

    # ----------------------------------------------------------
    # Transaction date >= customer_since
    #
    # A customer can't transact before they were onboarded.
    # ----------------------------------------------------------

    customer_since = pd.to_datetime(
        df["customer_id"].map(
            customers.set_index("customer_id")[
                "customer_since"
            ]
        ),
        errors="coerce",
    )

    before_onboarding_count = (
        transaction_dates < customer_since
    ).sum()

    tracker.check(
        before_onboarding_count == 0,
        "Transaction date >= customer_since",
        f"invalid={before_onboarding_count}",
    )

    # ----------------------------------------------------------
    # Transaction amount
    # ----------------------------------------------------------

    amount = pd.to_numeric(
        df["transaction_amount"],
        errors="coerce",
    )

    tracker.check(
        amount.isna().sum() == 0,
        "Transaction amount nulls",
        f"count={amount.isna().sum()}",
    )

    invalid_amount_count = (
        amount <= 0
    ).sum()

    tracker.check(
        invalid_amount_count == 0,
        "Transaction amount > 0",
        f"invalid={invalid_amount_count}",
    )

    # ----------------------------------------------------------
    # Payment status
    # ----------------------------------------------------------

    validate_no_nulls(
        tracker,
        df,
        "payment_status",
    )

    invalid_payment_status_count = (
        ~df["payment_status"].isin(
            VALID_PAYMENT_STATUSES
        )
    ).sum()

    tracker.check(
        invalid_payment_status_count == 0,
        "Payment status values",
        f"invalid={invalid_payment_status_count}",
    )

    # ----------------------------------------------------------
    # Payment date
    #
    # Payment date is allowed to be NULL because transactions
    # that are still pending or overdue may not have a
    # payment date.
    # ----------------------------------------------------------

    invalid_payment_date_count = (
        payment_dates.notna()
        & (
            payment_dates
            < transaction_dates
        )
    ).sum()

    tracker.check(
        invalid_payment_date_count == 0,
        "Payment date >= transaction date",
        f"invalid={invalid_payment_date_count}",
    )

    return df


# ==============================================================
# INTERACTION VALIDATION
# ==============================================================

def validate_interactions(
    tracker: ValidationTracker,
    customers: pd.DataFrame,
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("PROCESSED INTERACTION VALIDATION")
    print("=" * 70)

    df = load_dataset("interactions")

    required_columns = [
        "interaction_id",
        "customer_id",
        "interaction_date",
        "interaction_type",
        "channel",
        "sentiment",
        "resolution_status",
    ]

    validate_required_columns(
        tracker,
        df,
        required_columns,
    )

    validate_no_nulls(
        tracker,
        df,
        "interaction_id",
    )

    validate_no_duplicates(
        tracker,
        df,
        "interaction_id",
    )

    # ----------------------------------------------------------
    # Referential integrity
    # ----------------------------------------------------------

    valid_customer_ids = set(
        customers["customer_id"]
    )

    invalid_customer_count = (
        ~df["customer_id"].isin(
            valid_customer_ids
        )
    ).sum()

    tracker.check(
        invalid_customer_count == 0,
        "Customer referential integrity",
        f"invalid={invalid_customer_count}",
    )

    # ----------------------------------------------------------
    # Interaction date
    # ----------------------------------------------------------

    interaction_dates = pd.to_datetime(
        df["interaction_date"],
        errors="coerce",
    )

    invalid_date_count = (
        interaction_dates.isna().sum()
    )

    tracker.check(
        invalid_date_count == 0,
        "Interaction dates",
        f"invalid={invalid_date_count}",
    )

    # ----------------------------------------------------------
    # Interaction type
    # ----------------------------------------------------------

    invalid_type_count = (
        ~df["interaction_type"].isin(
            VALID_INTERACTION_TYPES
        )
    ).sum()

    tracker.check(
        invalid_type_count == 0,
        "Interaction types",
        f"invalid={invalid_type_count}",
    )

    # ----------------------------------------------------------
    # Channel
    # ----------------------------------------------------------

    invalid_channel_count = (
        ~df["channel"].isin(
            VALID_INTERACTION_CHANNELS
        )
    ).sum()

    tracker.check(
        invalid_channel_count == 0,
        "Interaction channels",
        f"invalid={invalid_channel_count}",
    )

    # ----------------------------------------------------------
    # Sentiment
    # ----------------------------------------------------------

    validate_no_nulls(
        tracker,
        df,
        "sentiment",
    )

    invalid_sentiment_count = (
        ~df["sentiment"].isin(
            VALID_SENTIMENTS
        )
    ).sum()

    tracker.check(
        invalid_sentiment_count == 0,
        "Sentiment values",
        f"invalid={invalid_sentiment_count}",
    )

    # ----------------------------------------------------------
    # Resolution status
    # ----------------------------------------------------------

    invalid_resolution_count = (
        ~df["resolution_status"].isin(
            VALID_RESOLUTION_STATUSES
        )
    ).sum()

    tracker.check(
        invalid_resolution_count == 0,
        "Resolution status",
        f"invalid={invalid_resolution_count}",
    )

    return df


# ==============================================================
# DOCUMENT VALIDATION
# ==============================================================

def validate_documents(
    tracker: ValidationTracker,
    customers: pd.DataFrame,
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("PROCESSED DOCUMENT VALIDATION")
    print("=" * 70)

    df = load_dataset("documents")

    required_columns = [
        "document_id",
        "customer_id",
        "document_type",
        "document_date",
        "source",
        "content",
        "is_active",
    ]

    validate_required_columns(
        tracker,
        df,
        required_columns,
    )

    validate_no_nulls(
        tracker,
        df,
        "document_id",
    )

    validate_no_duplicates(
        tracker,
        df,
        "document_id",
    )

    # ----------------------------------------------------------
    # Referential integrity
    # ----------------------------------------------------------

    valid_customer_ids = set(
        customers["customer_id"]
    )

    invalid_customer_count = (
        ~df["customer_id"].isin(
            valid_customer_ids
        )
    ).sum()

    tracker.check(
        invalid_customer_count == 0,
        "Customer referential integrity",
        f"invalid={invalid_customer_count}",
    )

    # ----------------------------------------------------------
    # Document dates
    # ----------------------------------------------------------

    document_dates = pd.to_datetime(
        df["document_date"],
        errors="coerce",
    )

    invalid_date_count = (
        document_dates.isna().sum()
    )

    tracker.check(
        invalid_date_count == 0,
        "Document dates",
        f"invalid={invalid_date_count}",
    )

    # ----------------------------------------------------------
    # Document type
    # ----------------------------------------------------------

    invalid_type_count = (
        ~df["document_type"].isin(
            VALID_DOCUMENT_TYPES
        )
    ).sum()

    tracker.check(
        invalid_type_count == 0,
        "Document types",
        f"invalid={invalid_type_count}",
    )

    # ----------------------------------------------------------
    # Source
    # ----------------------------------------------------------

    validate_no_nulls(
        tracker,
        df,
        "source",
    )

    # ----------------------------------------------------------
    # Content
    # ----------------------------------------------------------

    empty_content_count = (
        df["content"]
        .isna()
        .sum()
    )

    if empty_content_count == 0:
        empty_content_count = (
            df["content"]
            .astype("string")
            .str.strip()
            .eq("")
            .sum()
        )

    tracker.check(
        empty_content_count == 0,
        "Document content",
        f"empty={empty_content_count}",
    )

    return df


# ==============================================================
# MAIN
# ==============================================================

def main() -> None:

    print("\n" + "=" * 70)
    print("CUSTOMER RISK INTELLIGENCE")
    print("PROCESSED DATA VALIDATION")
    print("=" * 70)

    tracker = ValidationTracker()

    # ----------------------------------------------------------
    # Validate customer master first.
    # Other datasets depend on customer IDs.
    # ----------------------------------------------------------

    customers = validate_customers(
        tracker
    )

    transactions = validate_transactions(
        tracker,
        customers,
    )

    interactions = validate_interactions(
        tracker,
        customers,
    )

    documents = validate_documents(
        tracker,
        customers,
    )

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(
        f"Total checks: {tracker.total_checks}"
    )

    print(
        f"Failed checks: {tracker.failed_checks}"
    )

    if tracker.failed_checks == 0:
        print(
            "\nProcessed data validation PASSED."
        )
        print(
            "All cleaned datasets are valid."
        )
    else:
        print(
            "\nProcessed data validation FAILED."
        )
        print(
            "Review the failed checks above."
        )

        # Non-zero exit so the pipeline stops here instead of
        # building features on bad data.
        raise SystemExit(1)


if __name__ == "__main__":
    main()
