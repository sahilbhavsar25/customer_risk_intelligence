from pathlib import Path

import numpy as np
import pandas as pd

from src.data_generation.config import (
    NUM_TRANSACTIONS,
    DATA_START_DATE,
    DATA_END_DATE,
)


RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)

CUSTOMER_FILE = Path("data/raw/customers/customers.csv")

OUTPUT_DIR = Path("data/raw/transactions")
OUTPUT_FILE = OUTPUT_DIR / "transactions.csv"


SEGMENT_AMOUNT_RANGES = {
    "Enterprise": (20000, 200000),
    "Mid-Market": (10000, 100000),
    "SMB": (5000, 50000),
    "Startup": (2000, 30000),
    "Individual": (500, 10000),
}


PAYMENT_STATUS_PROBABILITIES = {
    "PAID": 0.72,
    "OVERDUE": 0.12,
    "PENDING": 0.08,
    "FAILED": 0.08,
}


def generate_transactions() -> pd.DataFrame:

    # ---------------------------------------------------------
    # 1. Load customers
    # ---------------------------------------------------------

    if not CUSTOMER_FILE.exists():
        raise FileNotFoundError(
            f"Customer file not found: {CUSTOMER_FILE}"
        )

    customers = pd.read_csv(CUSTOMER_FILE)

    if customers.empty:
        raise ValueError(
            "Customer dataset is empty. Generate customers first."
        )

    customers["customer_since"] = pd.to_datetime(
        customers["customer_since"]
    )

    transactions = []

    # ---------------------------------------------------------
    # 2. Generate transactions
    # ---------------------------------------------------------

    for index in range(1, NUM_TRANSACTIONS + 1):

        # Random customer
        customer = customers.iloc[
            rng.integers(0, len(customers))
        ]

        customer_id = customer["customer_id"]
        segment = customer["customer_segment"]
        customer_since = customer["customer_since"]

        # -----------------------------------------------------
        # Transaction date
        #
        # IMPORTANT:
        # Transaction cannot happen before customer joined.
        # -----------------------------------------------------

        start_date = max(
            pd.Timestamp(DATA_START_DATE),
            customer_since,
        )

        end_date = pd.Timestamp(DATA_END_DATE)

        # Safety check
        if start_date > end_date:
            start_date = end_date

        available_days = (
            end_date - start_date
        ).days

        transaction_date = (
            start_date
            + pd.Timedelta(
                days=int(
                    rng.integers(
                        0,
                        available_days + 1,
                    )
                )
            )
        )

        # -----------------------------------------------------
        # Transaction amount
        # -----------------------------------------------------

        min_amount, max_amount = SEGMENT_AMOUNT_RANGES.get(
            segment,
            (1000, 10000),
        )

        transaction_amount = round(
            rng.uniform(
                min_amount,
                max_amount,
            ),
            2,
        )

        # -----------------------------------------------------
        # Due date
        # -----------------------------------------------------

        payment_terms_days = int(
            rng.integers(15, 31)
        )

        due_date = (
            transaction_date
            + pd.Timedelta(
                days=payment_terms_days
            )
        )

        # -----------------------------------------------------
        # Payment status
        # -----------------------------------------------------

        payment_status = rng.choice(
            list(PAYMENT_STATUS_PROBABILITIES.keys()),
            p=list(PAYMENT_STATUS_PROBABILITIES.values()),
        )

        # -----------------------------------------------------
        # Payment date
        # -----------------------------------------------------

        payment_date = None

        if payment_status == "PAID":

            # Payment can happen slightly before or after
            # due date to simulate realistic behavior.

            payment_delay = int(
                rng.integers(-3, 11)
            )

            payment_date = (
                due_date
                + pd.Timedelta(
                    days=payment_delay
                )
            )

        # -----------------------------------------------------
        # Build transaction
        # -----------------------------------------------------

        transaction = {
            "transaction_id": f"TXN_{index:08d}",
            "customer_id": customer_id,
            "transaction_date": transaction_date.date(),
            "transaction_amount": transaction_amount,
            "payment_status": payment_status,
            "due_date": due_date.date(),
            "payment_date": (
                payment_date.date()
                if payment_date is not None
                else None
            ),
        }

        transactions.append(transaction)

    return pd.DataFrame(transactions)


def save_transactions(df: pd.DataFrame) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Transaction dataset saved successfully: {OUTPUT_FILE}"
    )


def validate_transactions(
    df: pd.DataFrame,
) -> None:

    print()
    print("=" * 60)
    print("TRANSACTION VALIDATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # Record count
    # ---------------------------------------------------------

    print(
        f"Total transactions: {len(df):,}"
    )

    # ---------------------------------------------------------
    # Required columns
    # ---------------------------------------------------------

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

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    print("Required columns: PASS")

    # ---------------------------------------------------------
    # Duplicate transaction IDs
    # ---------------------------------------------------------

    duplicate_ids = df["transaction_id"].duplicated().sum()

    print(
        f"Duplicate transaction IDs: {duplicate_ids}"
    )

    if duplicate_ids > 0:
        raise ValueError(
            "Duplicate transaction IDs found."
        )

    # ---------------------------------------------------------
    # Customer IDs
    # ---------------------------------------------------------

    customer_ids = set(
        pd.read_csv(CUSTOMER_FILE)["customer_id"]
    )

    invalid_customer_ids = (
        ~df["customer_id"].isin(customer_ids)
    ).sum()

    print(
        f"Invalid customer IDs: {invalid_customer_ids}"
    )

    if invalid_customer_ids > 0:
        raise ValueError(
            "Transactions contain invalid customer IDs."
        )

    # ---------------------------------------------------------
    # Amount validation
    # ---------------------------------------------------------

    invalid_amounts = (
        df["transaction_amount"] <= 0
    ).sum()

    print(
        f"Invalid transaction amounts: {invalid_amounts}"
    )

    if invalid_amounts > 0:
        raise ValueError(
            "Transactions contain invalid amounts."
        )

    # ---------------------------------------------------------
    # Date validation
    # ---------------------------------------------------------

    df["transaction_date"] = pd.to_datetime(
        df["transaction_date"]
    )

    df["due_date"] = pd.to_datetime(
        df["due_date"]
    )

    invalid_due_dates = (
        df["due_date"] < df["transaction_date"]
    ).sum()

    print(
        f"Invalid due dates: {invalid_due_dates}"
    )

    if invalid_due_dates > 0:
        raise ValueError(
            "Due date cannot be before transaction date."
        )

    # ---------------------------------------------------------
    # Payment status validation
    # ---------------------------------------------------------

    allowed_statuses = {
        "PAID",
        "PENDING",
        "OVERDUE",
        "FAILED",
    }

    invalid_statuses = (
        ~df["payment_status"].isin(
            allowed_statuses
        )
    ).sum()

    print(
        f"Invalid payment statuses: {invalid_statuses}"
    )

    if invalid_statuses > 0:
        raise ValueError(
            "Invalid payment status found."
        )

    print()
    print("Transaction validation: PASS")


def main() -> None:

    print("=" * 60)
    print("TRANSACTION DATA GENERATION")
    print("=" * 60)

    # Generate
    df = generate_transactions()

    print(
        f"Generated transactions: {len(df):,}"
    )

    # Validate
    validate_transactions(df)

    # Status distribution
    print()
    print("Payment status distribution:")

    print(
        df["payment_status"]
        .value_counts()
        .to_string()
    )

    # Save
    save_transactions(df)

    # Preview
    print()
    print("Preview:")

    print(
        df.head(10).to_string(index=False)
    )

    print()
    print(
        "Transaction generation completed successfully."
    )


if __name__ == "__main__":
    main()