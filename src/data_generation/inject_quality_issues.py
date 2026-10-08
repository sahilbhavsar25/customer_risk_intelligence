from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)


BASE_DIR = Path("data/raw")


CUSTOMER_FILE = (
    BASE_DIR
    / "customers"
    / "customers.csv"
)

TRANSACTION_FILE = (
    BASE_DIR
    / "transactions"
    / "transactions.csv"
)

INTERACTION_FILE = (
    BASE_DIR
    / "interactions"
    / "interactions.csv"
)

DOCUMENT_FILE = (
    BASE_DIR
    / "documents"
    / "documents.csv"
)


def inject_customer_issues() -> None:

    print()
    print("=" * 60)
    print("CUSTOMER DATA QUALITY INJECTION")
    print("=" * 60)

    df = pd.read_csv(CUSTOMER_FILE)

    original_count = len(df)

    # ---------------------------------------------------------
    # 1. Missing values
    # ---------------------------------------------------------

    missing_indices = rng.choice(
        df.index,
        size=20,
        replace=False,
    )

    df.loc[
        missing_indices,
        "country",
    ] = np.nan

    print("Injected missing country values: 20")

    # ---------------------------------------------------------
    # 2. Another missing field
    # ---------------------------------------------------------

    missing_indices = rng.choice(
        df.index,
        size=10,
        replace=False,
    )

    df.loc[
        missing_indices,
        "industry",
    ] = np.nan

    print("Injected missing industry values: 10")

    # ---------------------------------------------------------
    # 3. Invalid account status
    # ---------------------------------------------------------

    invalid_indices = rng.choice(
        df.index,
        size=5,
        replace=False,
    )

    df.loc[
        invalid_indices,
        "account_status",
    ] = "UNKNOWN_STATUS"

    print("Injected invalid account statuses: 5")

    # ---------------------------------------------------------
    # 4. Duplicate customer records
    # ---------------------------------------------------------

    duplicate_rows = df.sample(
        n=5,
        random_state=RANDOM_SEED,
    )

    df = pd.concat(
        [
            df,
            duplicate_rows,
        ],
        ignore_index=True,
    )

    print("Injected duplicate customer records: 5")

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    df.to_csv(
        CUSTOMER_FILE,
        index=False,
    )

    print(
        f"Original records: {original_count:,}"
    )

    print(
        f"Dirty records: {len(df):,}"
    )


def inject_transaction_issues() -> None:

    print()
    print("=" * 60)
    print("TRANSACTION DATA QUALITY INJECTION")
    print("=" * 60)

    df = pd.read_csv(
        TRANSACTION_FILE
    )

    original_count = len(df)

    # ---------------------------------------------------------
    # 1. Missing payment status
    # ---------------------------------------------------------

    indices = rng.choice(
        df.index,
        size=50,
        replace=False,
    )

    df.loc[
        indices,
        "payment_status",
    ] = np.nan

    print(
        "Injected missing payment statuses: 50"
    )

    # ---------------------------------------------------------
    # 2. Invalid customer IDs
    # ---------------------------------------------------------

    indices = rng.choice(
        df.index,
        size=20,
        replace=False,
    )

    df.loc[
        indices,
        "customer_id",
    ] = "CUST_INVALID"

    print(
        "Injected invalid customer IDs: 20"
    )

    # ---------------------------------------------------------
    # 3. Invalid transaction amounts
    # ---------------------------------------------------------

    indices = rng.choice(
        df.index,
        size=10,
        replace=False,
    )

    df.loc[
        indices,
        "transaction_amount",
    ] = -500

    print(
        "Injected negative transaction amounts: 10"
    )

    # ---------------------------------------------------------
    # 4. Extreme outliers
    # ---------------------------------------------------------

    indices = rng.choice(
        df.index,
        size=5,
        replace=False,
    )

    df.loc[
        indices,
        "transaction_amount",
    ] = 999999999

    print(
        "Injected transaction amount outliers: 5"
    )

    # ---------------------------------------------------------
    # 5. Invalid payment status
    # ---------------------------------------------------------

    indices = rng.choice(
        df.index,
        size=10,
        replace=False,
    )

    df.loc[
        indices,
        "payment_status",
    ] = "UNKNOWN"

    print(
        "Injected invalid payment statuses: 10"
    )

    # ---------------------------------------------------------
    # 6. Invalid dates
    # ---------------------------------------------------------

    indices = rng.choice(
        df.index,
        size=10,
        replace=False,
    )

    df.loc[
        indices,
        "transaction_date",
    ] = "INVALID_DATE"

    print(
        "Injected invalid transaction dates: 10"
    )

    # ---------------------------------------------------------
    # 7. Duplicate transactions
    # ---------------------------------------------------------

    duplicate_rows = df.sample(
        n=20,
        random_state=RANDOM_SEED,
    )

    df = pd.concat(
        [
            df,
            duplicate_rows,
        ],
        ignore_index=True,
    )

    print(
        "Injected duplicate transactions: 20"
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    df.to_csv(
        TRANSACTION_FILE,
        index=False,
    )

    print(
        f"Original records: {original_count:,}"
    )

    print(
        f"Dirty records: {len(df):,}"
    )


def inject_interaction_issues() -> None:

    print()
    print("=" * 60)
    print("INTERACTION DATA QUALITY INJECTION")
    print("=" * 60)

    df = pd.read_csv(
        INTERACTION_FILE
    )

    original_count = len(df)

    # ---------------------------------------------------------
    # 1. Missing sentiment
    # ---------------------------------------------------------

    indices = rng.choice(
        df.index,
        size=40,
        replace=False,
    )

    df.loc[
        indices,
        "sentiment",
    ] = np.nan

    print(
        "Injected missing sentiments: 40"
    )

    # ---------------------------------------------------------
    # 2. Invalid customer IDs
    # ---------------------------------------------------------

    indices = rng.choice(
        df.index,
        size=20,
        replace=False,
    )

    df.loc[
        indices,
        "customer_id",
    ] = "CUST_INVALID"

    print(
        "Injected invalid customer IDs: 20"
    )

    # ---------------------------------------------------------
    # 3. Invalid interaction type
    # ---------------------------------------------------------

    indices = rng.choice(
        df.index,
        size=10,
        replace=False,
    )

    df.loc[
        indices,
        "interaction_type",
    ] = "UNKNOWN_INTERACTION"

    print(
        "Injected invalid interaction types: 10"
    )

    # ---------------------------------------------------------
    # 4. Invalid resolution status
    # ---------------------------------------------------------

    indices = rng.choice(
        df.index,
        size=10,
        replace=False,
    )

    df.loc[
        indices,
        "resolution_status",
    ] = "UNKNOWN_STATUS"

    print(
        "Injected invalid resolution statuses: 10"
    )

    # ---------------------------------------------------------
    # 5. Duplicate interactions
    # ---------------------------------------------------------

    duplicate_rows = df.sample(
        n=20,
        random_state=RANDOM_SEED,
    )

    df = pd.concat(
        [
            df,
            duplicate_rows,
        ],
        ignore_index=True,
    )

    print(
        "Injected duplicate interactions: 20"
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    df.to_csv(
        INTERACTION_FILE,
        index=False,
    )

    print(
        f"Original records: {original_count:,}"
    )

    print(
        f"Dirty records: {len(df):,}"
    )


def inject_document_issues() -> None:

    print()
    print("=" * 60)
    print("DOCUMENT DATA QUALITY INJECTION")
    print("=" * 60)

    df = pd.read_csv(
        DOCUMENT_FILE
    )

    original_count = len(df)

    # ---------------------------------------------------------
    # 1. Missing content
    # ---------------------------------------------------------

    indices = rng.choice(
        df.index,
        size=10,
        replace=False,
    )

    df.loc[
        indices,
        "content",
    ] = np.nan

    print(
        "Injected missing document content: 10"
    )

    # ---------------------------------------------------------
    # 2. Invalid customer IDs
    # ---------------------------------------------------------

    indices = rng.choice(
        df.index,
        size=5,
        replace=False,
    )

    df.loc[
        indices,
        "customer_id",
    ] = "CUST_INVALID"

    print(
        "Injected invalid customer IDs: 5"
    )

    # ---------------------------------------------------------
    # 3. Invalid document types
    # ---------------------------------------------------------

    indices = rng.choice(
        df.index,
        size=5,
        replace=False,
    )

    df.loc[
        indices,
        "document_type",
    ] = "UNKNOWN_DOCUMENT"

    print(
        "Injected invalid document types: 5"
    )

    # ---------------------------------------------------------
    # 4. Missing source
    # ---------------------------------------------------------

    indices = rng.choice(
        df.index,
        size=5,
        replace=False,
    )

    df.loc[
        indices,
        "source",
    ] = np.nan

    print(
        "Injected missing document sources: 5"
    )

    # ---------------------------------------------------------
    # 5. Duplicate documents
    # ---------------------------------------------------------

    duplicate_rows = df.sample(
        n=5,
        random_state=RANDOM_SEED,
    )

    df = pd.concat(
        [
            df,
            duplicate_rows,
        ],
        ignore_index=True,
    )

    print(
        "Injected duplicate documents: 5"
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    df.to_csv(
        DOCUMENT_FILE,
        index=False,
    )

    print(
        f"Original records: {original_count:,}"
    )

    print(
        f"Dirty records: {len(df):,}"
    )


def main() -> None:

    print("=" * 60)
    print("DATA QUALITY INJECTION")
    print("=" * 60)

    inject_customer_issues()

    inject_transaction_issues()

    inject_interaction_issues()

    inject_document_issues()

    print()
    print("=" * 60)
    print("DATA QUALITY INJECTION COMPLETED")
    print("=" * 60)

    print()
    print("The raw datasets now intentionally contain:")
    print("- Missing values")
    print("- Duplicate records")
    print("- Invalid customer IDs")
    print("- Invalid dates")
    print("- Invalid categorical values")
    print("- Negative transaction amounts")
    print("- Extreme transaction outliers")
    print("- Missing document content")


if __name__ == "__main__":
    main()
