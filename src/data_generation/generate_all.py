"""
Generate the full synthetic raw dataset in dependency order.

    python -m src.data_generation.generate_all
"""

from src.data_generation import (
    generate_customers,
    generate_documents,
    generate_interactions,
    generate_transactions,
    inject_quality_issues,
)


def main() -> None:

    # Customers first - every other generator samples from the
    # customer master.
    generate_customers.main()
    generate_transactions.main()
    generate_interactions.main()
    generate_documents.main()

    # Dirty-data injection runs last, on top of clean raw data.
    inject_quality_issues.main()


if __name__ == "__main__":
    main()
