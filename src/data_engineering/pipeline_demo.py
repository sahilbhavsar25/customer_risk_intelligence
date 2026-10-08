"""
Reproducible demonstration of the pipeline's operational
behaviour:

    1. Idempotency            (run cleaning twice)
    2. Incremental processing (new / late / updated / duplicate)
    3. Failure halfway        (crash, verify, recover)
    4. New / missing columns  (schema drift)
    5. Unknown customers      (referential integrity + quarantine)

Everything runs in a temporary copy of data/raw, so the real
processed data is never touched.

    python -m src.data_engineering.pipeline_demo

Writes docs/pipeline_demo_results.md.
"""

import contextlib
import io
import json
import logging
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from src.data_engineering import cleaner, incremental, validator
from src.data_engineering.cleaner import (
    DATASET_ORDER,
    SCHEMAS,
    SchemaValidationError,
    file_sha256,
    processed_path,
    run_cleaning,
)


REPORT_PATH = Path("docs/pipeline_demo_results.md")


class LogCapture(logging.Handler):

    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage().split("\n")[0])


def snapshot(processed_dir: Path) -> dict:
    """
    Row count, duplicate-key count and hash per dataset.
    """

    state = {}

    for name in DATASET_ORDER:

        path = processed_path(processed_dir, name)

        if not path.exists():
            continue

        df = pd.read_csv(path)
        key = SCHEMAS[name]["primary_key"]

        state[name] = {
            "rows": len(df),
            "duplicate_keys": int(df[key].duplicated().sum()),
            "sha256": file_sha256(path)[:12],
        }

    return state


def quiet(function, *args, **kwargs):
    """Run a noisy pipeline step without flooding the console."""

    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


class Workspace:

    def __init__(self, root: Path):
        self.root = root
        self.raw = root / "raw"
        self.processed = root / "processed"
        self.quarantine = root / "quarantine"

        shutil.copytree(cleaner.RAW_DIR, self.raw)

    def clean(self):
        return run_cleaning(
            raw_dir=self.raw,
            processed_dir=self.processed,
            quarantine_dir=self.quarantine,
        )


# ==============================================================
# 1. IDEMPOTENCY
# ==============================================================

def demo_idempotency(ws: Workspace) -> dict:

    ws.clean()
    run_1 = snapshot(ws.processed)

    ws.clean()
    run_2 = snapshot(ws.processed)

    return {
        "run_1": run_1,
        "run_2": run_2,
        "identical_output": run_1 == run_2,
    }


# ==============================================================
# 2. INCREMENTAL
# ==============================================================

def build_batch(ws: Workspace, batch_dir: Path) -> dict:

    batch_dir.mkdir(parents=True)

    transactions = pd.read_csv(
        processed_path(ws.processed, "transactions")
    )

    customers = pd.read_csv(
        processed_path(ws.processed, "customers")
    )

    existing_customer = customers.iloc[10]["customer_id"]

    pending = transactions[
        transactions["payment_status"] == "PENDING"
    ].iloc[0]

    unchanged = transactions.iloc[100]

    pd.DataFrame(
        [
            {
                "customer_id": "CUST_01001",
                "customer_name": "Demo New Customer",
                "customer_since": "2025-12-15",
                "customer_segment": "SMB",
                "industry": "Retail",
                "country": "India",
                "account_status": "ACTIVE",
            }
        ]
    ).to_csv(batch_dir / "customers.csv", index=False)

    new_transaction = {
        "transaction_id": "TXN_90000001",
        "customer_id": "CUST_01001",
        "transaction_date": "2025-12-31",
        "transaction_amount": 12500.00,
        "payment_status": "PENDING",
        "due_date": "2026-01-30",
        "payment_date": None,
    }

    rows = [
        # brand-new transaction for the new customer
        new_transaction,
        # same row twice in the batch (in-batch duplicate)
        new_transaction,
        # late-arriving: dated months before the latest data
        {
            "transaction_id": "TXN_90000002",
            "customer_id": existing_customer,
            "transaction_date": "2025-09-15",
            "transaction_amount": 8400.00,
            "payment_status": "PAID",
            "due_date": "2025-10-10",
            "payment_date": "2025-10-01",
        },
        # correction: a PENDING transaction is now PAID
        {
            **pending.to_dict(),
            "payment_status": "PAID",
            "payment_date": pending["due_date"],
        },
        # exact re-delivery of an existing row
        unchanged.to_dict(),
        # unknown customer
        {
            "transaction_id": "TXN_90000003",
            "customer_id": "CUST_99999",
            "transaction_date": "2025-12-20",
            "transaction_amount": 999.00,
            "payment_status": "PAID",
            "due_date": "2026-01-10",
            "payment_date": "2025-12-28",
        },
    ]

    pd.DataFrame(rows).to_csv(
        batch_dir / "transactions.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "interaction_id": "INT_90000001",
                "customer_id": "CUST_01001",
                "interaction_date": "2025-12-22",
                "interaction_type": "SUPPORT_REQUEST",
                "channel": "EMAIL",
                "sentiment": "NEUTRAL",
                "resolution_status": "RESOLVED",
            }
        ]
    ).to_csv(batch_dir / "interactions.csv", index=False)

    pd.DataFrame(
        [
            {
                "document_id": "DOC_900001",
                "customer_id": "CUST_01001",
                "document_type": "CUSTOMER_PREFERENCE",
                "document_date": "2025-12-22",
                "source": "CRM",
                "content": (
                    "Demo New Customer prefers communication "
                    "through email."
                ),
                "is_active": True,
            }
        ]
    ).to_csv(batch_dir / "documents.csv", index=False)

    return {
        "updated_transaction": pending["transaction_id"],
        "redelivered_transaction": unchanged["transaction_id"],
    }


def demo_incremental(ws: Workspace) -> dict:

    # Initial full load, then the batch on top of it.
    ws.clean()

    before = snapshot(ws.processed)

    batch_dir = ws.root / "incoming" / "batch_001"
    batch_info = build_batch(ws, batch_dir)

    first = incremental.apply_batch(
        batch_dir,
        processed_dir=ws.processed,
        quarantine_dir=ws.quarantine,
    )

    after_first = snapshot(ws.processed)

    second = incremental.apply_batch(
        batch_dir,
        processed_dir=ws.processed,
        quarantine_dir=ws.quarantine,
    )

    after_second = snapshot(ws.processed)

    transactions = pd.read_csv(
        processed_path(ws.processed, "transactions")
    ).set_index("transaction_id")

    quarantine = pd.read_csv(
        ws.quarantine / "transactions_quarantine.csv"
    )

    return {
        "before": before,
        "first_apply": first,
        "after_first": after_first,
        "second_apply": second,
        "after_second": after_second,
        "second_apply_changed_nothing": after_first == after_second,
        "updated_status_now": transactions.loc[
            batch_info["updated_transaction"],
            "payment_status",
        ],
        "orphan_in_quarantine": bool(
            (
                (quarantine["transaction_id"] == "TXN_90000003")
                & (quarantine["quarantine_reason"] == "UNKNOWN_CUSTOMER")
            ).any()
        ),
    }


# ==============================================================
# 3. FAILURE HALFWAY
# ==============================================================

def demo_failure(ws: Workspace) -> dict:

    ws.clean()
    before = snapshot(ws.processed)
    manifest_before = json.loads(
        (ws.processed / cleaner.MANIFEST_FILE).read_text()
    )["run_id"]

    original = cleaner.TRANSFORMS["interactions"]

    def crash(*args, **kwargs):
        raise RuntimeError(
            "Simulated crash while cleaning interactions"
        )

    # Customers and transactions are already cleaned and
    # staged when this fires.
    cleaner.TRANSFORMS["interactions"] = crash

    error = None

    try:
        ws.clean()
    except RuntimeError as exc:
        error = str(exc)
    finally:
        cleaner.TRANSFORMS["interactions"] = original

    after_failure = snapshot(ws.processed)

    staging = ws.processed / cleaner.STAGING_DIR_NAME
    leftover_staging = (
        staging.exists() and any(staging.iterdir())
    )

    manifest_after_failure = json.loads(
        (ws.processed / cleaner.MANIFEST_FILE).read_text()
    )["run_id"]

    ws.clean()
    after_recovery = snapshot(ws.processed)

    return {
        "error": error,
        "processed_unchanged_after_failure": before == after_failure,
        "manifest_unchanged_after_failure": (
            manifest_before == manifest_after_failure
        ),
        "leftover_staging_files": leftover_staging,
        "recovered_output_identical": before == after_recovery,
        "duplicates_after_recovery": {
            name: info["duplicate_keys"]
            for name, info in after_recovery.items()
        },
    }


# ==============================================================
# 4. SCHEMA DRIFT
# ==============================================================

def demo_schema(ws: Workspace) -> dict:

    ws.clean()
    baseline = snapshot(ws.processed)

    customers_path = ws.raw / "customers" / "customers.csv"
    transactions_path = ws.raw / "transactions" / "transactions.csv"

    original_customers = customers_path.read_bytes()
    original_transactions = transactions_path.read_bytes()

    # ---- new column ------------------------------------------

    df = pd.read_csv(customers_path)
    df["loyalty_tier"] = "GOLD"
    df.to_csv(customers_path, index=False)

    capture = LogCapture()
    cleaner.logger.addHandler(capture)

    try:
        ws.clean()
    finally:
        cleaner.logger.removeHandler(capture)

    output_columns = list(
        pd.read_csv(
            processed_path(ws.processed, "customers")
        ).columns
    )

    new_column = {
        "logged": [
            message
            for message in capture.messages
            if "[SCHEMA]" in message
        ],
        "output_columns": output_columns,
        "column_reached_output": "loyalty_tier" in output_columns,
        "row_counts_unchanged": (
            {k: v["rows"] for k, v in snapshot(ws.processed).items()}
            == {k: v["rows"] for k, v in baseline.items()}
        ),
    }

    customers_path.write_bytes(original_customers)

    # ---- missing required column -----------------------------

    before = snapshot(ws.processed)

    df = pd.read_csv(transactions_path)
    df = df.drop(columns=["payment_status"])
    df.to_csv(transactions_path, index=False)

    error = None

    try:
        ws.clean()
    except SchemaValidationError as exc:
        error = str(exc)
    finally:
        transactions_path.write_bytes(original_transactions)

    missing_column = {
        "error": error,
        "processed_unchanged": before == snapshot(ws.processed),
    }

    return {
        "new_column": new_column,
        "missing_column": missing_column,
    }


# ==============================================================
# 5. UNKNOWN CUSTOMERS
# ==============================================================

def demo_unknown_customers(ws: Workspace) -> dict:

    ws.clean()

    raw_customers = pd.read_csv(
        ws.raw / "customers" / "customers.csv"
    )

    original_file = validator.TRANSACTION_FILE
    validator.TRANSACTION_FILE = (
        ws.raw / "transactions" / "transactions.csv"
    )

    try:
        checks = quiet(
            validator.validate_transactions,
            set(raw_customers["customer_id"].dropna()),
        )
    finally:
        validator.TRANSACTION_FILE = original_file

    raw = pd.read_csv(validator.TRANSACTION_FILE)

    orphan_raw = int(
        (~raw["customer_id"].isin(raw_customers["customer_id"])).sum()
    )

    quarantine = pd.read_csv(
        ws.quarantine / "transactions_quarantine.csv"
    )

    clean = pd.read_csv(
        processed_path(ws.processed, "transactions")
    )

    customers = pd.read_csv(
        processed_path(ws.processed, "customers")
    )

    return {
        "raw_referential_integrity_check_passed": checks[
            "invalid_customer_ids"
        ],
        "orphan_rows_in_raw": orphan_raw,
        "quarantined_unknown_customer": int(
            (quarantine["quarantine_reason"] == "UNKNOWN_CUSTOMER").sum()
        ),
        "orphans_left_in_clean": int(
            (~clean["customer_id"].isin(customers["customer_id"])).sum()
        ),
    }


# ==============================================================
# REPORT
# ==============================================================

def counts_table(state: dict) -> str:

    lines = [
        "| Dataset | Rows | Duplicate keys | SHA-256 (prefix) |",
        "|---|---:|---:|---|",
    ]

    for name, info in state.items():
        lines.append(
            f"| {name} | {info['rows']:,} | "
            f"{info['duplicate_keys']} | `{info['sha256']}` |"
        )

    return "\n".join(lines)


def write_report(results: dict) -> None:

    idem = results["idempotency"]
    inc = results["incremental"]
    fail = results["failure"]
    schema = results["schema"]
    unknown = results["unknown_customers"]

    tx_stats = inc["first_apply"]["datasets"]["transactions"]

    report = f"""# Pipeline demonstration results

Generated by `python -m src.data_engineering.pipeline_demo`.
All runs use a temporary copy of `data/raw`.

## 1. Idempotency

Cleaning was run twice on the same raw input.

**Run 1**

{counts_table(idem['run_1'])}

**Run 2**

{counts_table(idem['run_2'])}

Identical output (rows, duplicates, file hashes): **{idem['identical_output']}**

## 2. Incremental processing

Batch `batch_001` contained: one new customer, one new
transaction (delivered twice in the same file), one
late-arriving transaction (2025-09-15), one corrected
transaction (PENDING -> PAID), one exact re-delivery of an
existing transaction, one transaction for unknown customer
`CUST_99999`, one new interaction and one new document.

Transactions in the batch:

| Received | Duplicates in batch | Inserted | Updated | Unchanged | Quarantined | Late-arriving |
|---:|---:|---:|---:|---:|---:|---:|
| {tx_stats['received']} | {tx_stats['duplicates_in_batch']} | {tx_stats['inserted']} | {tx_stats['updated']} | {tx_stats['unchanged']} | {tx_stats['quarantined']} | {tx_stats['late_arriving']} |

- Feature snapshots to rebuild from: `{tx_stats.get('recompute_features_from')}`
- Corrected transaction status after upsert: `{inc['updated_status_now']}`
- Unknown-customer row in quarantine: **{inc['orphan_in_quarantine']}**

**Before**

{counts_table(inc['before'])}

**After applying the batch**

{counts_table(inc['after_first'])}

Re-applying the same batch: `{inc['second_apply']['status']}`
({inc['second_apply'].get('reason')}). Output unchanged:
**{inc['second_apply_changed_nothing']}**

## 3. Failure halfway through processing

A crash was injected while cleaning interactions (customers and
transactions had already been cleaned and staged).

- Error: `{fail['error']}`
- Processed files unchanged after the failure: **{fail['processed_unchanged_after_failure']}**
- Manifest still points to the last good run: **{fail['manifest_unchanged_after_failure']}**
- Partial staging files left behind: **{fail['leftover_staging_files']}**
- Rerun produced identical output: **{fail['recovered_output_identical']}**
- Duplicate keys after recovery: `{fail['duplicates_after_recovery']}`

## 4. Schema changes

**New column** (`loyalty_tier` added to raw customers):

- Logged: `{schema['new_column']['logged']}`
- Column reached processed output: **{schema['new_column']['column_reached_output']}**
- Row counts unchanged: **{schema['new_column']['row_counts_unchanged']}**

**Missing required column** (`payment_status` removed from raw transactions):

- Error: `{schema['missing_column']['error']}`
- Processed data unchanged: **{schema['missing_column']['processed_unchanged']}**

## 5. Deleted / unknown customers

- Raw referential-integrity check passed: **{unknown['raw_referential_integrity_check_passed']}**
- Orphan transactions in raw data: {unknown['orphan_rows_in_raw']}
- Quarantined as `UNKNOWN_CUSTOMER`: {unknown['quarantined_unknown_customer']}
- Orphans remaining in clean data: {unknown['orphans_left_in_clean']}
"""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


# ==============================================================
# MAIN
# ==============================================================

def main() -> None:

    # Keep the demo output readable: only warnings/errors from
    # the pipeline itself.
    cleaner.logger.setLevel(logging.WARNING)
    cleaner.logger.addHandler(logging.StreamHandler())

    results = {}

    steps = [
        ("idempotency", demo_idempotency),
        ("incremental", demo_incremental),
        ("failure", demo_failure),
        ("schema", demo_schema),
        ("unknown_customers", demo_unknown_customers),
    ]

    for name, step in steps:

        print(f"\n### {name}")

        with tempfile.TemporaryDirectory() as tmp:
            results[name] = quiet(step, Workspace(Path(tmp)))

        print(
            json.dumps(
                results[name],
                indent=2,
                default=str,
            )[:1500]
        )

    write_report(results)

    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
