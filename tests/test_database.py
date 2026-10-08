import pytest
from sqlalchemy import func, select, text

from src.database.connection import engine
from src.database.models import Customer


def database_available() -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not database_available(),
    reason="PostgreSQL not reachable",
)


def test_database_connection():
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        value = result.scalar()

    assert value == 1


def test_processed_data_loaded():
    # Populated by `python -m src.database.load_processed`.
    with engine.connect() as connection:
        count = connection.execute(
            select(func.count()).select_from(Customer)
        ).scalar()

    assert count >= 1000
