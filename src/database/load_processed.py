"""
Sync the processed CSVs into PostgreSQL.

    python -m src.database.load_processed

Idempotent: rows are upserted by primary key and rows that no
longer exist in processed data are deleted, all inside one
transaction. Running it twice leaves the database unchanged;
a failure part-way rolls everything back.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert

from src.database.connection import create_tables, engine
from src.database.models import (
    Customer,
    CustomerDocument,
    Interaction,
    Transaction,
)


PROCESSED_DIR = Path("data/processed")

BATCH_SIZE = 2000

# Parents first for upserts, children first for deletes.
TABLES = [
    ("customers", Customer, "customer_id"),
    ("transactions", Transaction, "transaction_id"),
    ("interactions", Interaction, "interaction_id"),
    ("documents", CustomerDocument, "document_id"),
]

DATE_COLUMNS = {
    "customers": ["customer_since"],
    "transactions": ["transaction_date", "due_date", "payment_date"],
    "interactions": ["interaction_date"],
    "documents": ["document_date"],
}


def load_dataset(name: str, model) -> list[dict]:

    df = pd.read_csv(
        PROCESSED_DIR / name / f"{name}_clean.csv"
    )

    for column in DATE_COLUMNS[name]:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
        ).dt.date

    # Only columns that exist on the table.
    columns = [
        column.name
        for column in model.__table__.columns
        if column.name in df.columns
    ]

    df = df[columns].astype(object)

    return df.where(pd.notna(df), None).to_dict(
        orient="records"
    )


def sync() -> dict:

    create_tables()

    now = datetime.utcnow()

    datasets = {
        name: load_dataset(name, model)
        for name, model, _ in TABLES
    }

    counts = {}

    with engine.begin() as connection:

        # ------------------------------------------------------
        # Delete rows that are no longer in processed data
        # (e.g. quarantined after a rule change).
        # ------------------------------------------------------

        for name, model, key in reversed(TABLES):

            keep = [row[key] for row in datasets[name]]
            key_column = getattr(model, key)

            connection.execute(
                delete(model).where(
                    key_column.not_in(keep)
                )
            )

        # ------------------------------------------------------
        # Upsert.
        # ------------------------------------------------------

        for name, model, key in TABLES:

            rows = datasets[name]

            for start in range(0, len(rows), BATCH_SIZE):

                batch = rows[start:start + BATCH_SIZE]

                statement = insert(model).values(batch)

                update_columns = {
                    column: statement.excluded[column]
                    for column in batch[0]
                    if column != key
                }

                if "updated_at" in model.__table__.columns:
                    update_columns["updated_at"] = now

                connection.execute(
                    statement.on_conflict_do_update(
                        index_elements=[key],
                        set_=update_columns,
                    )
                )

            counts[name] = connection.execute(
                select(func.count()).select_from(model)
            ).scalar()

    return counts


def main() -> None:

    print("\n" + "=" * 70)
    print("CUSTOMER RISK INTELLIGENCE")
    print("LOAD PROCESSED DATA INTO POSTGRESQL")
    print("=" * 70)

    counts = sync()

    for name, count in counts.items():
        print(f"{name:<14}{count:>8,} rows")


if __name__ == "__main__":
    main()
