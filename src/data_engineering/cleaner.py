import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config.settings import settings


logger = logging.getLogger("data_pipeline")


# ==============================================================
# PATH CONFIGURATION
# ==============================================================

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
QUARANTINE_DIR = Path("data/quarantine")
LOG_DIR = Path("logs")

MANIFEST_FILE = "_manifest.json"
STAGING_DIR_NAME = "_staging"


# ==============================================================
# VALID VALUES
# ==============================================================

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

# Largest legitimate invoice in the generator is 200k
# (Enterprise). Anything above 1M is treated as a data-entry /
# system error. A fixed business cap is used instead of a
# statistical one (IQR etc.) so that a small incremental batch
# is judged by exactly the same rule as the full load.
MAX_TRANSACTION_AMOUNT = 1_000_000


# ==============================================================
# SCHEMA CONTRACTS
#
# Columns listed here are the contract with the upstream
# source. Missing columns -> the run fails. Extra columns ->
# logged as a schema change and dropped, so unreviewed fields
# never reach features or the model.
# ==============================================================

SCHEMAS = {
    "customers": {
        "primary_key": "customer_id",
        "columns": [
            "customer_id",
            "customer_name",
            "customer_since",
            "customer_segment",
            "industry",
            "country",
            "account_status",
        ],
    },
    "transactions": {
        "primary_key": "transaction_id",
        "columns": [
            "transaction_id",
            "customer_id",
            "transaction_date",
            "transaction_amount",
            "payment_status",
            "due_date",
            "payment_date",
        ],
    },
    "interactions": {
        "primary_key": "interaction_id",
        "columns": [
            "interaction_id",
            "customer_id",
            "interaction_date",
            "interaction_type",
            "channel",
            "sentiment",
            "resolution_status",
        ],
    },
    "documents": {
        "primary_key": "document_id",
        "columns": [
            "document_id",
            "customer_id",
            "document_type",
            "document_date",
            "source",
            "content",
            "is_active",
        ],
    },
}

DATASET_ORDER = [
    "customers",
    "transactions",
    "interactions",
    "documents",
]


class SchemaValidationError(Exception):
    pass


# ==============================================================
# LOGGING
# ==============================================================

def configure_logging() -> None:
    """
    Console + logs/data_pipeline.log. Safe to call more than
    once.
    """

    if logger.handlers:
        return

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter("%(message)s")
    )

    file_handler = logging.FileHandler(
        LOG_DIR / "data_pipeline.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s"
        )
    )

    logger.addHandler(console)
    logger.addHandler(file_handler)


# ==============================================================
# HELPER FUNCTIONS
# ==============================================================

def normalize_text_column(
    df: pd.DataFrame,
    column: str,
) -> None:
    """
    Convert a column to pandas string dtype and remove
    leading/trailing whitespace.
    """
    if column in df.columns:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )


def check_schema(
    df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Enforce the column contract for one dataset.
    """

    expected = SCHEMAS[dataset_name]["columns"]

    missing = [
        column
        for column in expected
        if column not in df.columns
    ]

    if missing:
        logger.error(
            f"[SCHEMA] {dataset_name}: missing required "
            f"column(s) {missing}. Refusing to continue."
        )

        raise SchemaValidationError(
            f"{dataset_name} is missing required "
            f"column(s): {missing}"
        )

    extra = [
        column
        for column in df.columns
        if column not in expected
    ]

    if extra:
        logger.warning(
            f"[SCHEMA] {dataset_name}: new column(s) "
            f"detected {extra}. Not part of the contract, "
            f"dropping them for this run."
        )

    return df[expected].copy()


def quarantine_rows(
    df: pd.DataFrame,
    mask: pd.Series,
    reason: str,
    quarantined: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Move rows matching `mask` into the quarantine list and
    return the remaining rows.
    """

    if mask.any():
        rejected = df.loc[mask].copy()
        rejected["quarantine_reason"] = reason
        quarantined.append(rejected)

    return df.loc[~mask].copy()


def combine_quarantine(
    quarantined: list[pd.DataFrame],
) -> pd.DataFrame:

    if not quarantined:
        return pd.DataFrame()

    return pd.concat(
        quarantined,
        ignore_index=True,
    )


def write_csv_atomic(
    df: pd.DataFrame,
    path: Path,
) -> None:
    """
    Write to a temp file in the same directory and rename.
    os.replace is atomic, so readers see either the old file
    or the new one - never a half-written CSV.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    df.to_csv(
        tmp_path,
        index=False,
    )

    os.replace(tmp_path, path)


def file_sha256(path: Path) -> str:

    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for block in iter(
            lambda: file.read(1 << 16),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def processed_path(
    processed_dir: Path,
    dataset_name: str,
) -> Path:

    return (
        processed_dir
        / dataset_name
        / f"{dataset_name}_clean.csv"
    )


def load_raw_dataset(
    raw_dir: Path,
    dataset_name: str,
) -> pd.DataFrame:

    path = (
        raw_dir
        / dataset_name
        / f"{dataset_name}.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {path}"
        )

    return check_schema(
        pd.read_csv(path),
        dataset_name,
    )


# ==============================================================
# CUSTOMER CLEANING
# ==============================================================

def transform_customers(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    df = df.copy()
    quarantined = []

    # ----------------------------------------------------------
    # Normalize text fields
    # ----------------------------------------------------------

    text_columns = [
        "customer_id",
        "customer_name",
        "customer_segment",
        "industry",
        "country",
        "account_status",
    ]

    for column in text_columns:
        normalize_text_column(
            df,
            column,
        )

    # ----------------------------------------------------------
    # Parse dates
    # ----------------------------------------------------------

    df["customer_since"] = pd.to_datetime(
        df["customer_since"],
        errors="coerce",
    )

    # ----------------------------------------------------------
    # Remove duplicate customers
    # ----------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(
        subset=["customer_id"],
        keep="first",
    )

    logger.info(
        f"Removed duplicate customers: "
        f"{before - len(df)}"
    )

    # ----------------------------------------------------------
    # Fill missing industry / country
    # ----------------------------------------------------------

    missing_industry = df["industry"].isna().sum()

    df["industry"] = df["industry"].fillna(
        "Unknown"
    )

    logger.info(
        f"Filled missing industries: "
        f"{missing_industry}"
    )

    missing_country = df["country"].isna().sum()

    df["country"] = df["country"].fillna(
        "Unknown"
    )

    logger.info(
        f"Filled missing countries: "
        f"{missing_country}"
    )

    # ----------------------------------------------------------
    # Normalize invalid account status
    # ----------------------------------------------------------

    invalid_status_mask = (
        ~df["account_status"].isin(
            VALID_ACCOUNT_STATUSES
        )
    )

    invalid_status_count = invalid_status_mask.sum()

    df.loc[
        invalid_status_mask,
        "account_status",
    ] = "UNKNOWN"

    logger.info(
        f"Normalized invalid account statuses: "
        f"{invalid_status_count}"
    )

    # ----------------------------------------------------------
    # Quarantine rows with essential fields missing
    # ----------------------------------------------------------

    essential_columns = [
        "customer_id",
        "customer_name",
        "customer_since",
    ]

    missing_essential_mask = (
        df[essential_columns]
        .isna()
        .any(axis=1)
    )

    logger.info(
        f"Quarantined customers with invalid essential "
        f"fields: {missing_essential_mask.sum()}"
    )

    df = quarantine_rows(
        df,
        missing_essential_mask,
        "MISSING_REQUIRED_FIELD",
        quarantined,
    )

    return df, combine_quarantine(quarantined)


# ==============================================================
# TRANSACTION CLEANING
# ==============================================================

def transform_transactions(
    df: pd.DataFrame,
    customers: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    df = df.copy()
    quarantined = []

    # ----------------------------------------------------------
    # Normalize text columns
    # ----------------------------------------------------------

    text_columns = [
        "transaction_id",
        "customer_id",
        "payment_status",
    ]

    for column in text_columns:
        normalize_text_column(
            df,
            column,
        )

    # ----------------------------------------------------------
    # Parse date columns
    # ----------------------------------------------------------

    for column in [
        "transaction_date",
        "due_date",
        "payment_date",
    ]:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
        )

    # ----------------------------------------------------------
    # Convert transaction amount to numeric
    # ----------------------------------------------------------

    df["transaction_amount"] = pd.to_numeric(
        df["transaction_amount"],
        errors="coerce",
    )

    # ----------------------------------------------------------
    # Remove duplicate transactions
    # ----------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(
        subset=["transaction_id"],
        keep="first",
    )

    logger.info(
        f"Removed duplicate transactions: "
        f"{before - len(df)}"
    )

    # ----------------------------------------------------------
    # Referential integrity
    # ----------------------------------------------------------

    orphan_mask = (
        ~df["customer_id"].isin(
            set(customers["customer_id"])
        )
    )

    logger.info(
        f"Quarantined orphan transactions: "
        f"{orphan_mask.sum()}"
    )

    df = quarantine_rows(
        df,
        orphan_mask,
        "UNKNOWN_CUSTOMER",
        quarantined,
    )

    # ----------------------------------------------------------
    # Invalid dates
    # ----------------------------------------------------------

    invalid_date_mask = (
        df["transaction_date"].isna()
        | df["due_date"].isna()
    )

    logger.info(
        f"Quarantined transactions with invalid dates: "
        f"{invalid_date_mask.sum()}"
    )

    df = quarantine_rows(
        df,
        invalid_date_mask,
        "INVALID_DATE",
        quarantined,
    )

    # ----------------------------------------------------------
    # Invalid transaction amounts
    # ----------------------------------------------------------

    invalid_amount_mask = (
        df["transaction_amount"].isna()
        | (df["transaction_amount"] <= 0)
    )

    logger.info(
        f"Quarantined invalid transaction amounts: "
        f"{invalid_amount_mask.sum()}"
    )

    df = quarantine_rows(
        df,
        invalid_amount_mask,
        "INVALID_AMOUNT",
        quarantined,
    )

    # ----------------------------------------------------------
    # Extreme outliers
    # ----------------------------------------------------------

    outlier_mask = (
        df["transaction_amount"]
        > MAX_TRANSACTION_AMOUNT
    )

    logger.info(
        f"Quarantined transaction amount outliers "
        f"(> {MAX_TRANSACTION_AMOUNT:,}): "
        f"{outlier_mask.sum()}"
    )

    df = quarantine_rows(
        df,
        outlier_mask,
        "AMOUNT_OUTLIER",
        quarantined,
    )

    # ----------------------------------------------------------
    # Validate date ordering
    # transaction_date <= due_date
    # ----------------------------------------------------------

    invalid_order_mask = (
        df["transaction_date"]
        > df["due_date"]
    )

    logger.info(
        f"Quarantined invalid date-order records: "
        f"{invalid_order_mask.sum()}"
    )

    df = quarantine_rows(
        df,
        invalid_order_mask,
        "DUE_DATE_BEFORE_TRANSACTION",
        quarantined,
    )

    # ----------------------------------------------------------
    # Transaction before the customer was onboarded
    # ----------------------------------------------------------

    customer_since = pd.to_datetime(
        df["customer_id"].map(
            customers.set_index("customer_id")[
                "customer_since"
            ]
        ),
        errors="coerce",
    )

    before_onboarding_mask = (
        df["transaction_date"]
        < customer_since
    )

    logger.info(
        f"Quarantined transactions before customer_since: "
        f"{before_onboarding_mask.sum()}"
    )

    df = quarantine_rows(
        df,
        before_onboarding_mask,
        "BEFORE_CUSTOMER_SINCE",
        quarantined,
    )

    # ----------------------------------------------------------
    # Resolve missing payment statuses
    #
    # IMPORTANT:
    # We use the fixed DATA_AS_OF_DATE instead of
    # pd.Timestamp.now().
    # ----------------------------------------------------------

    as_of_date = pd.Timestamp(
        settings.DATA_AS_OF_DATE
    )

    missing_status_mask = (
        df["payment_status"].isna()
    )

    missing_status_count = missing_status_mask.sum()

    # If payment_date exists, classify as PAID.
    paid_mask = (
        missing_status_mask
        & df["payment_date"].notna()
    )

    df.loc[
        paid_mask,
        "payment_status",
    ] = "PAID"

    # If payment_date does not exist and the due date
    # has passed as of the historical processing date,
    # classify as OVERDUE.
    overdue_mask = (
        missing_status_mask
        & df["payment_date"].isna()
        & (
            df["due_date"]
            < as_of_date
        )
    )

    df.loc[
        overdue_mask,
        "payment_status",
    ] = "OVERDUE"

    # Remaining missing statuses are PENDING.
    df.loc[
        df["payment_status"].isna(),
        "payment_status",
    ] = "PENDING"

    logger.info(
        f"Resolved missing payment statuses: "
        f"{missing_status_count}"
    )

    # ----------------------------------------------------------
    # Normalize invalid payment statuses
    # ----------------------------------------------------------

    invalid_status_mask = (
        ~df["payment_status"].isin(
            VALID_PAYMENT_STATUSES
        )
    )

    df.loc[
        invalid_status_mask,
        "payment_status",
    ] = "PENDING"

    logger.info(
        f"Normalized invalid payment statuses: "
        f"{invalid_status_mask.sum()}"
    )

    return df, combine_quarantine(quarantined)


# ==============================================================
# INTERACTION CLEANING
# ==============================================================

def transform_interactions(
    df: pd.DataFrame,
    customers: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    df = df.copy()
    quarantined = []

    # ----------------------------------------------------------
    # Normalize text columns
    # ----------------------------------------------------------

    text_columns = [
        "interaction_id",
        "customer_id",
        "interaction_type",
        "channel",
        "sentiment",
        "resolution_status",
    ]

    for column in text_columns:
        normalize_text_column(
            df,
            column,
        )

    df["interaction_date"] = pd.to_datetime(
        df["interaction_date"],
        errors="coerce",
    )

    # ----------------------------------------------------------
    # Remove duplicates
    # ----------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(
        subset=["interaction_id"],
        keep="first",
    )

    logger.info(
        f"Removed duplicate interactions: "
        f"{before - len(df)}"
    )

    # ----------------------------------------------------------
    # Referential integrity
    # ----------------------------------------------------------

    orphan_mask = (
        ~df["customer_id"].isin(
            set(customers["customer_id"])
        )
    )

    logger.info(
        f"Quarantined orphan interactions: "
        f"{orphan_mask.sum()}"
    )

    df = quarantine_rows(
        df,
        orphan_mask,
        "UNKNOWN_CUSTOMER",
        quarantined,
    )

    # ----------------------------------------------------------
    # Invalid interaction dates
    # ----------------------------------------------------------

    invalid_date_mask = (
        df["interaction_date"].isna()
    )

    logger.info(
        f"Quarantined interactions with invalid dates: "
        f"{invalid_date_mask.sum()}"
    )

    df = quarantine_rows(
        df,
        invalid_date_mask,
        "INVALID_DATE",
        quarantined,
    )

    # ----------------------------------------------------------
    # Normalize interaction types
    # ----------------------------------------------------------

    invalid_type_mask = (
        ~df["interaction_type"].isin(
            VALID_INTERACTION_TYPES
        )
    )

    df.loc[
        invalid_type_mask,
        "interaction_type",
    ] = "SUPPORT_REQUEST"

    logger.info(
        f"Normalized invalid interaction types: "
        f"{invalid_type_mask.sum()}"
    )

    # ----------------------------------------------------------
    # Missing / invalid sentiments
    # ----------------------------------------------------------

    missing_sentiment_mask = (
        df["sentiment"].isna()
    )

    df.loc[
        missing_sentiment_mask,
        "sentiment",
    ] = "NEUTRAL"

    logger.info(
        f"Filled missing sentiments: "
        f"{missing_sentiment_mask.sum()}"
    )

    invalid_sentiment_mask = (
        ~df["sentiment"].isin(
            VALID_SENTIMENTS
        )
    )

    df.loc[
        invalid_sentiment_mask,
        "sentiment",
    ] = "NEUTRAL"

    logger.info(
        f"Normalized invalid sentiments: "
        f"{invalid_sentiment_mask.sum()}"
    )

    # ----------------------------------------------------------
    # Normalize resolution statuses
    # ----------------------------------------------------------

    invalid_resolution_mask = (
        ~df["resolution_status"].isin(
            VALID_RESOLUTION_STATUSES
        )
    )

    df.loc[
        invalid_resolution_mask,
        "resolution_status",
    ] = "PENDING"

    logger.info(
        f"Normalized invalid resolution statuses: "
        f"{invalid_resolution_mask.sum()}"
    )

    return df, combine_quarantine(quarantined)


# ==============================================================
# DOCUMENT CLEANING
# ==============================================================

def transform_documents(
    df: pd.DataFrame,
    customers: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    df = df.copy()
    quarantined = []

    text_columns = [
        "document_id",
        "customer_id",
        "document_type",
        "source",
        "content",
    ]

    for column in text_columns:
        normalize_text_column(
            df,
            column,
        )

    df["document_date"] = pd.to_datetime(
        df["document_date"],
        errors="coerce",
    )

    # ----------------------------------------------------------
    # Remove duplicates
    # ----------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(
        subset=["document_id"],
        keep="first",
    )

    logger.info(
        f"Removed duplicate documents: "
        f"{before - len(df)}"
    )

    # ----------------------------------------------------------
    # Referential integrity
    # ----------------------------------------------------------

    orphan_mask = (
        ~df["customer_id"].isin(
            set(customers["customer_id"])
        )
    )

    logger.info(
        f"Quarantined orphan documents: "
        f"{orphan_mask.sum()}"
    )

    df = quarantine_rows(
        df,
        orphan_mask,
        "UNKNOWN_CUSTOMER",
        quarantined,
    )

    # ----------------------------------------------------------
    # Invalid document dates
    # ----------------------------------------------------------

    invalid_date_mask = (
        df["document_date"].isna()
    )

    logger.info(
        f"Quarantined documents with invalid dates: "
        f"{invalid_date_mask.sum()}"
    )

    df = quarantine_rows(
        df,
        invalid_date_mask,
        "INVALID_DATE",
        quarantined,
    )

    # ----------------------------------------------------------
    # Normalize document type
    # ----------------------------------------------------------

    invalid_type_mask = (
        ~df["document_type"].isin(
            VALID_DOCUMENT_TYPES
        )
    )

    df.loc[
        invalid_type_mask,
        "document_type",
    ] = "UNKNOWN"

    logger.info(
        f"Normalized invalid document types: "
        f"{invalid_type_mask.sum()}"
    )

    # ----------------------------------------------------------
    # Fill missing source
    # ----------------------------------------------------------

    missing_source_mask = (
        df["source"].isna()
    )

    df.loc[
        missing_source_mask,
        "source",
    ] = "UNKNOWN"

    logger.info(
        f"Filled missing document sources: "
        f"{missing_source_mask.sum()}"
    )

    # ----------------------------------------------------------
    # Empty content
    #
    # Documents without content cannot provide useful
    # evidence for the RAG pipeline.
    # ----------------------------------------------------------

    empty_content_mask = (
        df["content"].isna()
        | (
            df["content"]
            .astype("string")
            .str.strip()
            == ""
        )
    )

    logger.info(
        f"Quarantined documents with empty content: "
        f"{empty_content_mask.sum()}"
    )

    df = quarantine_rows(
        df,
        empty_content_mask,
        "EMPTY_CONTENT",
        quarantined,
    )

    return df, combine_quarantine(quarantined)


TRANSFORMS = {
    "transactions": transform_transactions,
    "interactions": transform_interactions,
    "documents": transform_documents,
}


# ==============================================================
# STAGING / PROMOTION
# ==============================================================

def promote_outputs(
    staged: dict[str, pd.DataFrame],
    quarantine: dict[str, pd.DataFrame],
    processed_dir: Path,
    quarantine_dir: Path,
    run_id: str,
    run_type: str,
) -> dict:
    """
    Write every output to a staging directory first. Only when
    all of them are written successfully are they moved into
    their final location, and the manifest is written last.

    A crash at any point before promotion leaves the previous
    processed files untouched.
    """

    staging_dir = (
        processed_dir
        / STAGING_DIR_NAME
        / run_id
    )

    staged_files = {}

    for dataset_name, df in staged.items():

        path = processed_path(
            staging_dir,
            dataset_name,
        )

        write_csv_atomic(df, path)

        staged_files[dataset_name] = path

    # ----------------------------------------------------------
    # Promote. Each os.replace is atomic.
    # ----------------------------------------------------------

    for dataset_name, staged_file in staged_files.items():

        final_path = processed_path(
            processed_dir,
            dataset_name,
        )

        final_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        os.replace(staged_file, final_path)

    shutil.rmtree(
        staging_dir,
        ignore_errors=True,
    )

    # Remove the parent too if no other run is staging.
    try:
        staging_dir.parent.rmdir()
    except OSError:
        pass

    for dataset_name, df in quarantine.items():

        if df.empty:
            # Header-only file, so the quarantine CSV is always
            # readable and an old run's rows don't linger.
            df = pd.DataFrame(
                columns=SCHEMAS[dataset_name]["columns"]
                + ["quarantine_reason"]
            )

        write_csv_atomic(
            df,
            quarantine_dir
            / f"{dataset_name}_quarantine.csv",
        )

    # ----------------------------------------------------------
    # Manifest
    # ----------------------------------------------------------

    manifest = {
        "run_id": run_id,
        "run_type": run_type,
        "completed_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "as_of_date": str(
            settings.DATA_AS_OF_DATE
        ),
        "datasets": {},
    }

    for dataset_name in DATASET_ORDER:

        path = processed_path(
            processed_dir,
            dataset_name,
        )

        if not path.exists():
            continue

        quarantine_path = (
            quarantine_dir
            / f"{dataset_name}_quarantine.csv"
        )

        manifest["datasets"][dataset_name] = {
            "rows": int(
                len(pd.read_csv(path))
            ),
            "sha256": file_sha256(path),
            "quarantined_rows": (
                int(len(pd.read_csv(quarantine_path)))
                if quarantine_path.exists()
                else 0
            ),
        }

    manifest_path = processed_dir / MANIFEST_FILE

    tmp_path = manifest_path.with_suffix(".tmp")

    with open(
        tmp_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=4,
        )

    os.replace(tmp_path, manifest_path)

    return manifest


def discard_staging(
    processed_dir: Path,
    run_id: str,
) -> None:

    shutil.rmtree(
        processed_dir
        / STAGING_DIR_NAME
        / run_id,
        ignore_errors=True,
    )


# ==============================================================
# FULL CLEANING RUN
# ==============================================================

def run_cleaning(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    quarantine_dir: Path = QUARANTINE_DIR,
) -> dict:
    """
    Full refresh: raw -> processed.

    Idempotent by construction: same raw input -> same clean
    output (deterministic rules, fixed DATA_AS_OF_DATE, no
    appends). Output only replaces the previous version after
    every dataset has been cleaned successfully.
    """

    run_id = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")

    logger.info(
        f"Cleaning run {run_id} started "
        f"(as-of date {settings.DATA_AS_OF_DATE})"
    )

    staged = {}
    quarantine = {}

    try:

        # ------------------------------------------------------
        # Customer master first because all other datasets
        # depend on customer IDs.
        # ------------------------------------------------------

        logger.info("\n" + "=" * 70)
        logger.info("CLEANING CUSTOMERS")
        logger.info("=" * 70)

        customers, quarantine["customers"] = (
            transform_customers(
                load_raw_dataset(
                    raw_dir,
                    "customers",
                )
            )
        )

        staged["customers"] = customers

        for dataset_name, transform in TRANSFORMS.items():

            logger.info("\n" + "=" * 70)
            logger.info(
                f"CLEANING {dataset_name.upper()}"
            )
            logger.info("=" * 70)

            raw_df = load_raw_dataset(
                raw_dir,
                dataset_name,
            )

            clean_df, quarantine[dataset_name] = (
                transform(
                    raw_df,
                    customers,
                )
            )

            staged[dataset_name] = clean_df

            logger.info(
                f"{dataset_name}: "
                f"{len(raw_df):,} -> {len(clean_df):,}"
            )

        manifest = promote_outputs(
            staged,
            quarantine,
            processed_dir,
            quarantine_dir,
            run_id,
            run_type="full_refresh",
        )

        # A full refresh rebuilds only from raw, so any batch
        # applied earlier is no longer in processed data.
        # Reset the incremental ledger so those batches can be
        # re-applied.
        (processed_dir / "_batches.json").unlink(
            missing_ok=True
        )

    except Exception:

        discard_staging(
            processed_dir,
            run_id,
        )

        logger.exception(
            f"Cleaning run {run_id} FAILED. Partial output "
            f"was discarded; previously processed data is "
            f"unchanged. Fix the cause and rerun."
        )

        raise

    logger.info(
        f"Cleaning run {run_id} completed"
    )

    return manifest


# ==============================================================
# MAIN PIPELINE
# ==============================================================

def main() -> None:

    configure_logging()

    logger.info("\n" + "=" * 70)
    logger.info("CUSTOMER RISK INTELLIGENCE")
    logger.info("DATA CLEANING PIPELINE")
    logger.info("=" * 70)

    manifest = run_cleaning()

    logger.info("\n" + "=" * 70)
    logger.info("DATA CLEANING COMPLETED")
    logger.info("=" * 70)

    for dataset_name, info in manifest["datasets"].items():
        logger.info(
            f"{dataset_name:<14}"
            f"rows={info['rows']:>6,}   "
            f"quarantined={info['quarantined_rows']:,}"
        )

    logger.info(
        f"\nProcessed data directory: {PROCESSED_DIR}"
    )
    logger.info(
        f"Quarantine directory:     {QUARANTINE_DIR}"
    )


if __name__ == "__main__":
    main()
