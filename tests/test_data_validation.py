"""
Checks on the processed (clean) datasets.
"""

import pandas as pd
import pytest

from src.data_engineering.processed_validator import (
    VALID_ACCOUNT_STATUSES,
    VALID_DOCUMENT_TYPES,
    VALID_INTERACTION_TYPES,
    VALID_PAYMENT_STATUSES,
    VALID_SENTIMENTS,
)


def load(name):
    return pd.read_csv(f"data/processed/{name}/{name}_clean.csv")


@pytest.fixture(scope="module")
def data():
    return {
        name: load(name)
        for name in ["customers", "transactions", "interactions", "documents"]
    }


@pytest.mark.parametrize(
    "name, key",
    [
        ("customers", "customer_id"),
        ("transactions", "transaction_id"),
        ("interactions", "interaction_id"),
        ("documents", "document_id"),
    ],
)
def test_no_duplicate_ids(data, name, key):
    assert data[name][key].notna().all()
    assert not data[name][key].duplicated().any()


@pytest.mark.parametrize("name", ["transactions", "interactions", "documents"])
def test_no_orphan_records(data, name):
    known = set(data["customers"]["customer_id"])
    assert data[name]["customer_id"].isin(known).all()


def test_valid_categories(data):
    assert data["customers"]["account_status"].isin(VALID_ACCOUNT_STATUSES).all()
    assert data["transactions"]["payment_status"].isin(VALID_PAYMENT_STATUSES).all()
    assert data["interactions"]["interaction_type"].isin(VALID_INTERACTION_TYPES).all()
    assert data["interactions"]["sentiment"].isin(VALID_SENTIMENTS).all()
    assert data["documents"]["document_type"].isin(VALID_DOCUMENT_TYPES).all()


@pytest.mark.parametrize(
    "name, columns",
    [
        ("customers", ["customer_id", "customer_name", "customer_since", "account_status"]),
        ("transactions", ["transaction_id", "customer_id", "transaction_date",
                          "transaction_amount", "payment_status", "due_date"]),
        ("interactions", ["interaction_id", "customer_id", "interaction_date",
                          "interaction_type", "sentiment"]),
        ("documents", ["document_id", "customer_id", "document_type", "content"]),
    ],
)
def test_no_required_nulls(data, name, columns):
    assert data[name][columns].notna().all().all()


def test_transaction_amounts_valid(data):
    amounts = data["transactions"]["transaction_amount"]
    assert (amounts > 0).all()
    # Injected 999,999,999 outliers must not survive cleaning.
    assert amounts.max() <= 1_000_000


def test_no_activity_before_onboarding(data):
    since = data["customers"].set_index("customer_id")["customer_since"]
    transactions = data["transactions"]
    onboarding = pd.to_datetime(transactions["customer_id"].map(since))
    assert (pd.to_datetime(transactions["transaction_date"]) >= onboarding).all()


def test_assessment_minimum_sizes(data):
    assert len(data["customers"]) >= 1000
    assert len(data["transactions"]) >= 20000
    assert len(data["interactions"]) >= 10000
    assert len(data["documents"]) >= 100
