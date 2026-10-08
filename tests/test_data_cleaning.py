"""
Cleaning rules, idempotency, incremental upserts, failure
recovery and schema drift - on small synthetic inputs in a temp
directory.
"""

import pandas as pd
import pytest

from src.data_engineering import cleaner, incremental
from src.data_engineering.cleaner import (
    SchemaValidationError,
    check_schema,
    file_sha256,
    processed_path,
    run_cleaning,
    transform_customers,
    transform_transactions,
)


CUSTOMERS = pd.DataFrame(
    [
        ["CUST_00001", "A", "2025-01-10", "SMB", "Retail", "India", "ACTIVE"],
        ["CUST_00002", "B", "2025-02-01", "SMB", None, "USA", "WEIRD"],
        ["CUST_00002", "B", "2025-02-01", "SMB", None, "USA", "WEIRD"],
    ],
    columns=cleaner.SCHEMAS["customers"]["columns"],
)

TRANSACTIONS = pd.DataFrame(
    [
        ["TXN_1", "CUST_00001", "2025-03-01", 100.0, "PAID", "2025-03-20", "2025-03-10"],
        ["TXN_1", "CUST_00001", "2025-03-01", 100.0, "PAID", "2025-03-20", "2025-03-10"],
        ["TXN_2", "CUST_99999", "2025-03-01", 100.0, "PAID", "2025-03-20", None],
        ["TXN_3", "CUST_00001", "bad-date", 100.0, "PAID", "2025-03-20", None],
        ["TXN_4", "CUST_00001", "2025-03-01", -5.0, "PAID", "2025-03-20", None],
        ["TXN_5", "CUST_00001", "2025-03-01", 999999999, "PAID", "2025-03-20", None],
        ["TXN_6", "CUST_00001", "2025-01-01", 50.0, "PAID", "2025-01-20", None],
        ["TXN_7", "CUST_00002", "2025-03-01", 70.0, None, "2025-03-20", None],
    ],
    columns=cleaner.SCHEMAS["transactions"]["columns"],
)

INTERACTIONS = pd.DataFrame(
    [["INT_1", "CUST_00001", "2025-03-02", "COMPLAINT", "EMAIL", None, "RESOLVED"]],
    columns=cleaner.SCHEMAS["interactions"]["columns"],
)

DOCUMENTS = pd.DataFrame(
    [["DOC_1", "CUST_00001", "COMPLAINT", "2025-03-02", "CRM", "Unhappy.", True]],
    columns=cleaner.SCHEMAS["documents"]["columns"],
)


@pytest.fixture
def workspace(tmp_path):
    raw = tmp_path / "raw"
    for name, df in [
        ("customers", CUSTOMERS),
        ("transactions", TRANSACTIONS),
        ("interactions", INTERACTIONS),
        ("documents", DOCUMENTS),
    ]:
        (raw / name).mkdir(parents=True)
        df.to_csv(raw / name / f"{name}.csv", index=False)

    dirs = {
        "raw_dir": raw,
        "processed_dir": tmp_path / "processed",
        "quarantine_dir": tmp_path / "quarantine",
    }
    return dirs


def hashes(processed_dir):
    return {
        name: file_sha256(processed_path(processed_dir, name))
        for name in cleaner.DATASET_ORDER
    }


# --------------------------------------------------------------
# rules
# --------------------------------------------------------------

def test_customer_cleaning_rules():
    clean, quarantine = transform_customers(CUSTOMERS)
    assert clean["customer_id"].tolist() == ["CUST_00001", "CUST_00002"]
    assert clean.loc[clean["customer_id"] == "CUST_00002", "industry"].item() == "Unknown"
    assert clean.loc[clean["customer_id"] == "CUST_00002", "account_status"].item() == "UNKNOWN"
    assert quarantine.empty


def test_transaction_cleaning_quarantines_bad_rows():
    customers, _ = transform_customers(CUSTOMERS)
    clean, quarantine = transform_transactions(TRANSACTIONS, customers)

    assert sorted(clean["transaction_id"]) == ["TXN_1", "TXN_7"]

    reasons = dict(zip(quarantine["transaction_id"], quarantine["quarantine_reason"]))
    assert reasons == {
        "TXN_2": "UNKNOWN_CUSTOMER",
        "TXN_3": "INVALID_DATE",
        "TXN_4": "INVALID_AMOUNT",
        "TXN_5": "AMOUNT_OUTLIER",
        "TXN_6": "BEFORE_CUSTOMER_SINCE",
    }

    # Missing status, unpaid, due date in the past -> OVERDUE.
    assert clean.loc[clean["transaction_id"] == "TXN_7", "payment_status"].item() == "OVERDUE"


def test_new_column_is_logged_and_dropped(caplog):
    df = CUSTOMERS.assign(loyalty_tier="GOLD")
    with caplog.at_level("WARNING", logger="data_pipeline"):
        checked = check_schema(df, "customers")
    assert "loyalty_tier" not in checked.columns
    assert "loyalty_tier" in caplog.text


def test_missing_column_fails_clearly():
    with pytest.raises(SchemaValidationError, match="payment_status"):
        check_schema(TRANSACTIONS.drop(columns=["payment_status"]), "transactions")


# --------------------------------------------------------------
# idempotency / failure
# --------------------------------------------------------------

def test_cleaning_is_idempotent(workspace):
    run_cleaning(**workspace)
    first = hashes(workspace["processed_dir"])
    run_cleaning(**workspace)
    assert hashes(workspace["processed_dir"]) == first


def test_failure_halfway_keeps_previous_output(workspace, monkeypatch):
    run_cleaning(**workspace)
    before = hashes(workspace["processed_dir"])

    def crash(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setitem(cleaner.TRANSFORMS, "interactions", crash)

    with pytest.raises(RuntimeError):
        run_cleaning(**workspace)

    assert hashes(workspace["processed_dir"]) == before

    staging = workspace["processed_dir"] / cleaner.STAGING_DIR_NAME
    assert not staging.exists() or not any(staging.iterdir())

    monkeypatch.undo()
    run_cleaning(**workspace)
    assert hashes(workspace["processed_dir"]) == before


# --------------------------------------------------------------
# incremental
# --------------------------------------------------------------

def write_batch(path, **frames):
    path.mkdir(parents=True)
    for name, df in frames.items():
        df.to_csv(path / f"{name}.csv", index=False)
    return path


def test_incremental_upsert_and_reapply(workspace, tmp_path):
    run_cleaning(**workspace)
    processed = workspace["processed_dir"]

    batch = write_batch(
        tmp_path / "batch_001",
        transactions=pd.DataFrame(
            [
                # new
                ["TXN_8", "CUST_00002", "2025-03-05", 10.0, "PAID", "2025-03-30", "2025-03-06"],
                # late-arriving (before latest existing 2025-03-01)
                ["TXN_9", "CUST_00002", "2025-02-15", 20.0, "PAID", "2025-03-01", "2025-02-20"],
                # correction of TXN_7
                ["TXN_7", "CUST_00002", "2025-03-01", 70.0, "PAID", "2025-03-20", "2025-03-19"],
                # exact re-delivery
                ["TXN_1", "CUST_00001", "2025-03-01", 100.0, "PAID", "2025-03-20", "2025-03-10"],
                # unknown customer
                ["TXN_10", "CUST_77777", "2025-03-05", 10.0, "PAID", "2025-03-30", None],
            ],
            columns=cleaner.SCHEMAS["transactions"]["columns"],
        ),
    )

    report = incremental.apply_batch(
        batch,
        processed_dir=processed,
        quarantine_dir=workspace["quarantine_dir"],
    )

    stats = report["datasets"]["transactions"]
    assert stats["inserted"] == 2
    assert stats["updated"] == 1
    assert stats["unchanged"] == 1
    assert stats["quarantined"] == 1
    assert stats["late_arriving"] == 1

    transactions = pd.read_csv(processed_path(processed, "transactions"))
    assert not transactions["transaction_id"].duplicated().any()
    assert transactions.set_index("transaction_id").loc["TXN_7", "payment_status"] == "PAID"
    assert "TXN_10" not in set(transactions["transaction_id"])

    quarantine = pd.read_csv(workspace["quarantine_dir"] / "transactions_quarantine.csv")
    assert "TXN_10" in set(quarantine["transaction_id"])

    # Same batch again -> skipped, nothing changes.
    before = hashes(processed)
    again = incremental.apply_batch(
        batch,
        processed_dir=processed,
        quarantine_dir=workspace["quarantine_dir"],
    )
    assert again["status"] == "skipped"
    assert hashes(processed) == before
