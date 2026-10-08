from pathlib import Path

import numpy as np
import pandas as pd

from src.data_generation.config import (
    NUM_INTERACTIONS,
    DATA_START_DATE,
    DATA_END_DATE,
    INTERACTION_TYPES,
    INTERACTION_CHANNELS,
    SENTIMENTS,
    RESOLUTION_STATUSES,
)


RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)

CUSTOMER_FILE = Path(
    "data/raw/customers/customers.csv"
)

OUTPUT_DIR = Path(
    "data/raw/interactions"
)

OUTPUT_FILE = OUTPUT_DIR / "interactions.csv"


INTERACTION_TYPE_PROBS = {
    "SUPPORT_REQUEST": 0.25,
    "COMPLAINT": 0.15,
    "PAYMENT_QUERY": 0.15,
    "RENEWAL": 0.10,
    "TECHNICAL_ISSUE": 0.15,
    "ACCOUNT_QUERY": 0.08,
    "FEEDBACK": 0.07,
    "ESCALATION": 0.05,
}


CHANNEL_PROBS = {
    "EMAIL": 0.30,
    "PHONE": 0.25,
    "CHAT": 0.30,
    "PORTAL": 0.15,
}


SENTIMENT_PROBS = {
    "POSITIVE": 0.45,
    "NEUTRAL": 0.35,
    "NEGATIVE": 0.20,
}


RESOLUTION_PROBS = {
    "RESOLVED": 0.70,
    "PENDING": 0.20,
    "ESCALATED": 0.10,
}


def generate_interactions() -> pd.DataFrame:

    if not CUSTOMER_FILE.exists():
        raise FileNotFoundError(
            f"Customer file not found: {CUSTOMER_FILE}"
        )

    customers = pd.read_csv(CUSTOMER_FILE)

    if customers.empty:
        raise ValueError(
            "Customer dataset is empty."
        )

    customers["customer_since"] = pd.to_datetime(
        customers["customer_since"]
    )

    interactions = []

    for index in range(1, NUM_INTERACTIONS + 1):

        customer = customers.iloc[
            rng.integers(0, len(customers))
        ]

        customer_id = customer["customer_id"]
        customer_since = customer["customer_since"]

        start_date = max(
            pd.Timestamp(DATA_START_DATE),
            customer_since,
        )

        end_date = pd.Timestamp(DATA_END_DATE)

        if start_date > end_date:
            start_date = end_date

        available_days = (
            end_date - start_date
        ).days

        interaction_date = (
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

        interaction_type = rng.choice(
            list(INTERACTION_TYPE_PROBS.keys()),
            p=list(INTERACTION_TYPE_PROBS.values()),
        )

        channel = rng.choice(
            list(CHANNEL_PROBS.keys()),
            p=list(CHANNEL_PROBS.values()),
        )

        sentiment = rng.choice(
            list(SENTIMENT_PROBS.keys()),
            p=list(SENTIMENT_PROBS.values()),
        )

        resolution_status = rng.choice(
            list(RESOLUTION_PROBS.keys()),
            p=list(RESOLUTION_PROBS.values()),
        )

        # Complaints and escalations should have
        # a stronger negative-sentiment relationship.
        if interaction_type in [
            "COMPLAINT",
            "ESCALATION",
        ]:
            sentiment = rng.choice(
                ["NEGATIVE", "NEUTRAL"],
                p=[0.75, 0.25],
            )

        # Escalations are more likely to remain pending
        # or escalated.
        if interaction_type == "ESCALATION":
            resolution_status = rng.choice(
                ["PENDING", "ESCALATED"],
                p=[0.35, 0.65],
            )

        interaction = {
            "interaction_id": f"INT_{index:08d}",
            "customer_id": customer_id,
            "interaction_date": interaction_date.date(),
            "interaction_type": interaction_type,
            "channel": channel,
            "sentiment": sentiment,
            "resolution_status": resolution_status,
        }

        interactions.append(interaction)

    return pd.DataFrame(interactions)


def validate_interactions(
    df: pd.DataFrame,
) -> None:

    print()
    print("=" * 60)
    print("INTERACTION VALIDATION")
    print("=" * 60)

    print(
        f"Total interactions: {len(df):,}"
    )

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

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    print("Required columns: PASS")

    duplicate_ids = (
        df["interaction_id"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate interaction IDs: {duplicate_ids}"
    )

    if duplicate_ids > 0:
        raise ValueError(
            "Duplicate interaction IDs found."
        )

    customers = pd.read_csv(CUSTOMER_FILE)

    valid_customer_ids = set(
        customers["customer_id"]
    )

    invalid_customer_ids = (
        ~df["customer_id"].isin(valid_customer_ids)
    ).sum()

    print(
        f"Invalid customer IDs: "
        f"{invalid_customer_ids}"
    )

    if invalid_customer_ids > 0:
        raise ValueError(
            "Invalid customer IDs found."
        )

    invalid_types = (
        ~df["interaction_type"].isin(INTERACTION_TYPES)
    ).sum()

    print(
        f"Invalid interaction types: "
        f"{invalid_types}"
    )

    if invalid_types > 0:
        raise ValueError(
            "Invalid interaction types found."
        )

    invalid_channels = (
        ~df["channel"].isin(INTERACTION_CHANNELS)
    ).sum()

    print(
        f"Invalid channels: {invalid_channels}"
    )

    if invalid_channels > 0:
        raise ValueError(
            "Invalid interaction channels found."
        )

    invalid_sentiments = (
        ~df["sentiment"].isin(SENTIMENTS)
    ).sum()

    print(
        f"Invalid sentiments: "
        f"{invalid_sentiments}"
    )

    if invalid_sentiments > 0:
        raise ValueError(
            "Invalid sentiment values found."
        )

    invalid_resolution = (
        ~df["resolution_status"]
        .isin(RESOLUTION_STATUSES)
    ).sum()

    print(
        f"Invalid resolution statuses: "
        f"{invalid_resolution}"
    )

    if invalid_resolution > 0:
        raise ValueError(
            "Invalid resolution statuses found."
        )

    print()
    print("Interaction validation: PASS")


def save_interactions(
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
        f"Interaction dataset saved successfully: "
        f"{OUTPUT_FILE}"
    )


def main() -> None:

    print("=" * 60)
    print("INTERACTION DATA GENERATION")
    print("=" * 60)

    df = generate_interactions()

    print(
        f"Generated interactions: {len(df):,}"
    )

    validate_interactions(df)

    print()
    print("Interaction type distribution:")

    print(
        df["interaction_type"]
        .value_counts()
        .to_string()
    )

    print()
    print("Sentiment distribution:")

    print(
        df["sentiment"]
        .value_counts()
        .to_string()
    )

    print()
    print("Resolution distribution:")

    print(
        df["resolution_status"]
        .value_counts()
        .to_string()
    )

    save_interactions(df)

    print()
    print("Preview:")

    print(
        df.head(10).to_string(index=False)
    )

    print()
    print(
        "Interaction generation completed successfully."
    )


if __name__ == "__main__":
    main()
