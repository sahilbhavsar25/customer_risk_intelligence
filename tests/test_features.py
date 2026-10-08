import pandas as pd
import pytest

from src.config.settings import settings
from src.ml.train_models import CATEGORICAL_FEATURES, NUMERIC_FEATURES


@pytest.fixture(scope="module")
def features():
    return pd.read_csv(
        "data/processed/ml/customer_features.csv",
        parse_dates=["observation_date"],
    )


def test_one_row_per_customer_and_date(features):
    assert not features.duplicated(["customer_id", "observation_date"]).any()


def test_target_is_binary(features):
    assert set(features["target"].unique()) <= {0, 1}


def test_target_not_used_as_feature():
    assert "target" not in NUMERIC_FEATURES + CATEGORICAL_FEATURES


def test_observation_dates_leave_room_for_prediction_window(features):
    # Every snapshot needs a full 30-day future window inside
    # the data, otherwise the target is incomplete.
    last_allowed = pd.Timestamp(settings.DATA_END_DATE) - pd.Timedelta(
        days=settings.PREDICTION_WINDOW_DAYS
    )
    assert features["observation_date"].max() <= last_allowed


def test_no_snapshot_before_customer_joined(features):
    customers = pd.read_csv(
        "data/processed/customers/customers_clean.csv",
        parse_dates=["customer_since"],
    ).set_index("customer_id")

    joined = features["customer_id"].map(customers["customer_since"])
    assert (joined <= features["observation_date"]).all()


def test_time_based_split_has_no_overlap():
    train = pd.read_csv("data/processed/ml/splits/train.csv", parse_dates=["observation_date"])
    test = pd.read_csv("data/processed/ml/splits/test.csv", parse_dates=["observation_date"])
    assert train["observation_date"].max() < test["observation_date"].min()


def test_features_only_use_past_transactions(features):
    """
    Spot check leakage: the transaction count at a snapshot
    must not exceed the number of transactions dated on or
    before the observation date.
    """

    transactions = pd.read_csv(
        "data/processed/transactions/transactions_clean.csv",
        parse_dates=["transaction_date"],
    )

    sample = features.sample(50, random_state=0)

    for _, row in sample.iterrows():
        available = transactions[
            (transactions["customer_id"] == row["customer_id"])
            & (transactions["transaction_date"] <= row["observation_date"])
        ]
        assert row["transaction_count"] <= len(available)
