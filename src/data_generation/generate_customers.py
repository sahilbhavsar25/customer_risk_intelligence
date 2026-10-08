from pathlib import Path

import pandas as pd
from faker import Faker

from src.data_generation.config import (
    NUM_CUSTOMERS,
    CUSTOMER_SEGMENTS,
    INDUSTRIES,
    COUNTRIES,
    ACCOUNT_STATUSES,
    DATA_START_DATE,
    DATA_END_DATE,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

RANDOM_SEED = 42

fake = Faker()
fake.seed_instance(RANDOM_SEED)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

OUTPUT_DIR = Path("data/raw/customers")
OUTPUT_FILE = OUTPUT_DIR / "customers.csv"


# ---------------------------------------------------------
# Customer generation
# ---------------------------------------------------------

def generate_customers() -> pd.DataFrame:
    """
    Generate synthetic customer master data.
    """

    customers = []

    for index in range(1, NUM_CUSTOMERS + 1):

        customer_id = f"CUST_{index:05d}"

        customer = {
            "customer_id": customer_id,
            "customer_name": fake.name(),
            # Bounded by DATA_END_DATE (not "today") so that
            # nobody joins after the transaction history ends
            # and reruns produce the same customer master.
            "customer_since": fake.date_between(
                start_date=DATA_START_DATE,
                end_date=DATA_END_DATE,
            ),
            "customer_segment": fake.random_element(
                elements=CUSTOMER_SEGMENTS
            ),
            "industry": fake.random_element(
                elements=INDUSTRIES
            ),
            "country": fake.random_element(
                elements=COUNTRIES
            ),
            "account_status": fake.random_element(
                elements=ACCOUNT_STATUSES
            ),
        }

        customers.append(customer)

    df = pd.DataFrame(customers)

    return df


# ---------------------------------------------------------
# Save dataset
# ---------------------------------------------------------

def save_customers(df: pd.DataFrame) -> None:
    """
    Save customer dataset to CSV.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Customer dataset saved successfully: "
        f"{OUTPUT_FILE}"
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    print("=" * 60)
    print("CUSTOMER DATA GENERATION")
    print("=" * 60)

    df = generate_customers()

    print(f"Generated customers: {len(df):,}")

    save_customers(df)

    print()
    print("Preview:")
    print(df.head().to_string(index=False))

    print()
    print("Customer generation completed successfully.")


if __name__ == "__main__":
    main()
