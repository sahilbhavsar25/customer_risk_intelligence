from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

from src.data_generation.config import (
    NUM_DOCUMENTS,
    DATA_START_DATE,
    DATA_END_DATE,
    DOCUMENT_TYPES,
)


RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)

fake = Faker()
fake.seed_instance(RANDOM_SEED)

CUSTOMER_FILE = Path(
    "data/raw/customers/customers.csv"
)

OUTPUT_DIR = Path(
    "data/raw/documents"
)

OUTPUT_FILE = OUTPUT_DIR / "documents.csv"


DOCUMENT_TYPE_PROBS = {
    "COMPLAINT": 0.20,
    "PAYMENT_ISSUE": 0.20,
    "SUPPORT_NOTE": 0.20,
    "RENEWAL_DISCUSSION": 0.15,
    "ESCALATION": 0.10,
    "ACCOUNT_NOTE": 0.10,
    "CUSTOMER_PREFERENCE": 0.05,
}


def generate_document_content(
    document_type: str,
    customer_name: str,
) -> str:

    templates = {

        "COMPLAINT": [
            (
                f"{customer_name} contacted support regarding "
                "repeated service issues. The customer reported "
                "that the issue affected normal business operations "
                "and requested a quick resolution."
            ),
            (
                f"Customer {customer_name} submitted a complaint "
                "about delayed service response and requested "
                "additional follow-up from the support team."
            ),
            (
                f"{customer_name} expressed dissatisfaction with "
                "the recent service experience and requested "
                "the issue to be escalated."
            ),
        ],

        "PAYMENT_ISSUE": [
            (
                f"{customer_name} reported a payment-related issue. "
                "The customer indicated that a recent payment was "
                "delayed and requested clarification regarding the "
                "outstanding amount."
            ),
            (
                f"Payment discussion with {customer_name}. "
                "The customer requested additional time to resolve "
                "an outstanding payment and asked for the payment "
                "status to be reviewed."
            ),
            (
                f"{customer_name} contacted the finance team regarding "
                "an overdue invoice. The customer requested details "
                "about the due amount and payment timeline."
            ),
        ],

        "SUPPORT_NOTE": [
            (
                f"Support interaction with {customer_name}. "
                "The customer requested assistance with a technical "
                "issue. Support reviewed the case and provided "
                "troubleshooting guidance."
            ),
            (
                f"Support team assisted {customer_name} with an "
                "account-related issue. The case was reviewed and "
                "the customer was provided with the required guidance."
            ),
            (
                f"{customer_name} contacted the support team for "
                "assistance. The issue was documented for follow-up "
                "and monitoring."
            ),
        ],

        "RENEWAL_DISCUSSION": [
            (
                f"Renewal discussion with {customer_name}. "
                "The customer requested information about renewal "
                "terms, pricing, and available options."
            ),
            (
                f"{customer_name} discussed the upcoming renewal "
                "and requested clarification regarding contract "
                "terms and service continuity."
            ),
        ],

        "ESCALATION": [
            (
                f"Case involving {customer_name} was escalated "
                "to a senior support team because the issue "
                "required additional investigation."
            ),
            (
                f"{customer_name} requested escalation after "
                "the initial support interaction did not fully "
                "resolve the reported issue."
            ),
        ],

        "ACCOUNT_NOTE": [
            (
                f"Account review completed for {customer_name}. "
                "Customer account information was reviewed and "
                "updated as part of routine account management."
            ),
            (
                f"Account note for {customer_name}. "
                "Customer profile and recent account activity "
                "were reviewed by the account management team."
            ),
        ],

        "CUSTOMER_PREFERENCE": [
            (
                f"{customer_name} prefers communication through "
                "email and requested important account updates "
                "to be shared through the registered email address."
            ),
            (
                f"Communication preference recorded for "
                f"{customer_name}. The customer prefers receiving "
                "support updates through the online portal."
            ),
        ],
    }

    selected_templates = templates.get(
        document_type,
        templates["ACCOUNT_NOTE"],
    )

    return rng.choice(selected_templates)


def generate_documents() -> pd.DataFrame:

    # ---------------------------------------------------------
    # 1. Validate customer source
    # ---------------------------------------------------------

    if not CUSTOMER_FILE.exists():
        raise FileNotFoundError(
            f"Customer file not found: {CUSTOMER_FILE}"
        )

    customers = pd.read_csv(
        CUSTOMER_FILE
    )

    if customers.empty:
        raise ValueError(
            "Customer dataset is empty."
        )

    customers["customer_since"] = pd.to_datetime(
        customers["customer_since"]
    )

    documents = []

    # ---------------------------------------------------------
    # 2. Generate documents
    # ---------------------------------------------------------

    for index in range(
        1,
        NUM_DOCUMENTS + 1,
    ):

        customer = customers.iloc[
            rng.integers(
                0,
                len(customers),
            )
        ]

        customer_id = customer[
            "customer_id"
        ]

        customer_name = customer[
            "customer_name"
        ]

        customer_since = customer[
            "customer_since"
        ]

        # -----------------------------------------------------
        # Document date cannot be before customer creation
        # -----------------------------------------------------

        start_date = max(
            pd.Timestamp(DATA_START_DATE),
            customer_since,
        )

        end_date = pd.Timestamp(
            DATA_END_DATE
        )

        if start_date > end_date:
            start_date = end_date

        available_days = (
            end_date - start_date
        ).days

        document_date = (
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
        # Document type
        # -----------------------------------------------------

        document_type = rng.choice(
            list(
                DOCUMENT_TYPE_PROBS.keys()
            ),
            p=list(
                DOCUMENT_TYPE_PROBS.values()
            ),
        )

        # -----------------------------------------------------
        # Content
        # -----------------------------------------------------

        content = generate_document_content(
            document_type=document_type,
            customer_name=customer_name,
        )

        # -----------------------------------------------------
        # Source
        # -----------------------------------------------------

        source = rng.choice(
            [
                "CRM",
                "SUPPORT_SYSTEM",
                "EMAIL",
                "ACCOUNT_MANAGEMENT",
                "FINANCE_SYSTEM",
            ]
        )

        document = {
            "document_id": (
                f"DOC_{index:06d}"
            ),
            "customer_id": customer_id,
            "document_type": document_type,
            "document_date": document_date.date(),
            "source": source,
            "content": content,
            "is_active": True,
        }

        documents.append(
            document
        )

    return pd.DataFrame(
        documents
    )


def validate_documents(
    df: pd.DataFrame,
) -> None:

    print()
    print("=" * 60)
    print("DOCUMENT VALIDATION")
    print("=" * 60)

    print(
        f"Total documents: {len(df):,}"
    )

    # ---------------------------------------------------------
    # Required columns
    # ---------------------------------------------------------

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

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    print(
        "Required columns: PASS"
    )

    # ---------------------------------------------------------
    # Duplicate IDs
    # ---------------------------------------------------------

    duplicate_ids = (
        df["document_id"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate document IDs: "
        f"{duplicate_ids}"
    )

    if duplicate_ids > 0:
        raise ValueError(
            "Duplicate document IDs found."
        )

    # ---------------------------------------------------------
    # Customer referential integrity
    # ---------------------------------------------------------

    customers = pd.read_csv(
        CUSTOMER_FILE
    )

    valid_customer_ids = set(
        customers["customer_id"]
    )

    invalid_customer_ids = (
        ~df["customer_id"].isin(
            valid_customer_ids
        )
    ).sum()

    print(
        f"Invalid customer IDs: "
        f"{invalid_customer_ids}"
    )

    if invalid_customer_ids > 0:
        raise ValueError(
            "Invalid customer IDs found."
        )

    # ---------------------------------------------------------
    # Document types
    # ---------------------------------------------------------

    invalid_document_types = (
        ~df["document_type"].isin(
            DOCUMENT_TYPES
        )
    ).sum()

    print(
        f"Invalid document types: "
        f"{invalid_document_types}"
    )

    if invalid_document_types > 0:
        raise ValueError(
            "Invalid document types found."
        )

    # ---------------------------------------------------------
    # Content validation
    # ---------------------------------------------------------

    empty_content = (
        df["content"]
        .fillna("")
        .str.strip()
        .eq("")
        .sum()
    )

    print(
        f"Empty document content: "
        f"{empty_content}"
    )

    if empty_content > 0:
        raise ValueError(
            "Empty document content found."
        )

    print()
    print(
        "Document validation: PASS"
    )


def save_documents(
    df: pd.DataFrame,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Document dataset saved successfully: "
        f"{OUTPUT_FILE}"
    )


def main() -> None:

    print("=" * 60)
    print("CUSTOMER DOCUMENT GENERATION")
    print("=" * 60)

    df = generate_documents()

    print(
        f"Generated documents: "
        f"{len(df):,}"
    )

    validate_documents(df)

    print()
    print(
        "Document type distribution:"
    )

    print(
        df["document_type"]
        .value_counts()
        .to_string()
    )

    save_documents(df)

    print()
    print("Preview:")

    print(
        df.head(5).to_string(
            index=False
        )
    )

    print()
    print(
        "Document generation completed successfully."
    )


if __name__ == "__main__":
    main()
