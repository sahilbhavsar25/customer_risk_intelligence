import os
from datetime import date

from dotenv import load_dotenv


load_dotenv()


class Settings:
    # ==============================================================
    # APPLICATION
    # ==============================================================

    APP_NAME: str = os.getenv(
        "APP_NAME",
        "Customer Risk Intelligence"
    )

    APP_ENV: str = os.getenv(
        "APP_ENV",
        "development"
    )

    DEBUG: bool = os.getenv(
        "DEBUG",
        "false"
    ).lower() == "true"

    # ==============================================================
    # DATABASE
    # ==============================================================

    DATABASE_HOST: str = os.getenv(
        "DATABASE_HOST",
        "localhost"
    )

    DATABASE_PORT: int = int(
        os.getenv(
            "DATABASE_PORT",
            "5432"
        )
    )

    DATABASE_NAME: str = os.getenv(
        "DATABASE_NAME",
        "customer_risk_db"
    )

    DATABASE_USER: str = os.getenv(
        "DATABASE_USER",
        "customer_risk_user"
    )

    DATABASE_PASSWORD: str = os.getenv(
        "DATABASE_PASSWORD",
        ""
    )

    # ==============================================================
    # OPENAI
    # ==============================================================

    OPENAI_API_KEY: str = os.getenv(
        "OPENAI_API_KEY",
        ""
    )

    # ==============================================================
    # QDRANT
    # ==============================================================

    QDRANT_URL: str = os.getenv(
        "QDRANT_URL",
        "http://localhost:6333"
    )

    QDRANT_API_KEY: str = os.getenv(
        "QDRANT_API_KEY",
        ""
    )

    # ==============================================================
    # DATASET CONFIGURATION
    # ==============================================================

    NUM_CUSTOMERS: int = 1000
    NUM_TRANSACTIONS: int = 30000
    NUM_INTERACTIONS: int = 15000
    NUM_DOCUMENTS: int = 300

    DATA_START_DATE: date = date(2025, 1, 1)
    DATA_END_DATE: date = date(2025, 12, 31)

    # Fixed date used for deterministic historical processing.
    DATA_AS_OF_DATE: date = date(2025, 12, 31)

    # ==============================================================
    # ML / FEATURE CONFIGURATION
    # ==============================================================

    # Number of historical days used to calculate features.
    FEATURE_LOOKBACK_DAYS: int = 90

    # Recent activity window.
    RECENT_ACTIVITY_DAYS: int = 30

    # Prediction horizon.
    PREDICTION_WINDOW_DAYS: int = 30

    # Threshold for model probability:
    # probability >= 0.60 -> HIGH risk.
    HIGH_RISK_THRESHOLD: float = 0.60

    # Threshold for the business-defined future risk score
    # used to construct the supervised learning target.
    TARGET_RISK_SCORE_THRESHOLD: float = 0.50

    # ==============================================================
    # DATABASE URL
    # ==============================================================

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://"
            f"{self.DATABASE_USER}:"
            f"{self.DATABASE_PASSWORD}@"
            f"{self.DATABASE_HOST}:"
            f"{self.DATABASE_PORT}/"
            f"{self.DATABASE_NAME}"
        )


settings = Settings()