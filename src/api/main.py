from contextlib import asynccontextmanager
from pathlib import Path
import json
import logging

import joblib
import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src.api.customers import router as customers_router
from src.api.intelligence import router as intelligence_router
from src.api.validation import normalize_customer_id
from src.config.settings import settings

# Import the exact feature definitions used during training.
from src.ml.train_models import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger("api")


# ==============================================================
# PATHS
# ==============================================================

MODEL_DIR = Path("models")

FEATURE_DATASET_PATH = Path(
    "data/processed/ml/customer_features.csv"
)

CUSTOMER_PATH = Path(
    "data/processed/customers/customers_clean.csv"
)

TRAINING_PATH = Path(
    "data/processed/ml/splits/train.csv"
)

MODEL_SELECTION_PATH = (
    MODEL_DIR / "model_selection.json"
)

FRONTEND_DIR = Path("frontend")


# ==============================================================
# FEATURES
# ==============================================================

FEATURE_COLUMNS = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


# ==============================================================
# REQUEST / RESPONSE MODELS
# ==============================================================

class RiskPredictionRequest(BaseModel):
    customer_id: str = Field(
        ...,
        description="Customer identifier, e.g. CUST_00004",
        examples=["CUST_00004"],
    )

    @field_validator("customer_id", mode="before")
    @classmethod
    def validate_customer_id(cls, value):
        return normalize_customer_id(value)


class RiskPredictionResponse(BaseModel):
    customer_id: str
    observation_date: str
    risk_probability: float
    risk_level: str
    risk_factors: list[str]
    model_name: str
    model_version: str


# ==============================================================
# MODEL STATE
#
# Loaded once at startup. If loading fails the API still starts
# (health + intelligence keep working) and /predict-risk
# returns 503 instead of the whole service crash-looping.
# ==============================================================

model = None
preprocessor = None
feature_data = None
customer_ids: set[str] = set()
shap_explainer = None
selected_model_name = None
model_version = "v1"
model_load_error = None


# ==============================================================
# RISK LEVEL
# ==============================================================

def get_risk_level(
    probability: float,
) -> str:

    if probability >= settings.HIGH_RISK_THRESHOLD:
        return "HIGH"

    if probability >= 0.30:
        return "MEDIUM"

    return "LOW"


# ==============================================================
# LOAD MODEL
# ==============================================================

def load_model_artifacts():

    global model
    global preprocessor
    global feature_data
    global customer_ids
    global shap_explainer
    global selected_model_name
    global model_version

    for path in [
        MODEL_SELECTION_PATH,
        FEATURE_DATASET_PATH,
        CUSTOMER_PATH,
        MODEL_DIR / "preprocessor.joblib",
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact not found: {path}"
            )

    with open(
        MODEL_SELECTION_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        selection = json.load(file)

    selected_model_name = selection["selected_model"]
    model_version = selection.get("model_version", "v1")

    model_path = (
        MODEL_DIR
        / f"{selected_model_name}.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Selected model not found: {model_path}"
        )

    preprocessor = joblib.load(
        MODEL_DIR / "preprocessor.joblib"
    )

    model = joblib.load(model_path)

    feature_data = pd.read_csv(
        FEATURE_DATASET_PATH,
        parse_dates=["observation_date"],
    )

    # Customer existence comes from the customer master, not
    # from the feature table.
    customer_ids = set(
        pd.read_csv(CUSTOMER_PATH)["customer_id"].astype(str)
    )

    # ----------------------------------------------------------
    # SHAP background. Built once at startup rather than per
    # request.
    # ----------------------------------------------------------

    if TRAINING_PATH.exists():

        X_background = preprocessor.transform(
            pd.read_csv(TRAINING_PATH)[FEATURE_COLUMNS]
        )

        shap_explainer = shap.Explainer(
            model,
            X_background[:min(100, X_background.shape[0])],
        )

    logger.info(
        "Loaded model %s (%s), %s feature rows, %s customers",
        selected_model_name,
        model_version,
        f"{len(feature_data):,}",
        f"{len(customer_ids):,}",
    )




# ==============================================================
# GENERATE RISK FACTORS
# ==============================================================

def generate_risk_factors(
    raw_row: pd.Series,
    transformed_row,
) -> list[str]:

    factors = []

    # ----------------------------------------------------------
    # SHAP explanation
    # ----------------------------------------------------------

    if shap_explainer is not None:

        shap_result = shap_explainer(
            transformed_row
        )

        shap_values = shap_result.values

        if shap_values.ndim == 3:
            shap_values = (
                shap_values[:, :, -1]
            )

        # If one row was passed, remove row dimension.
        if shap_values.ndim == 2:
            shap_values = shap_values[0]

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        ranked_indices = np.argsort(
            shap_values
        )[::-1]

        for index in ranked_indices:

            contribution = float(
                shap_values[index]
            )

            # Only features pushing toward high risk.
            if contribution <= 0:
                break

            feature_name = (
                feature_names[index]
            )

            # Only map numerical business features.
            if feature_name.startswith(
                "numeric__"
            ):

                clean_name = (
                    feature_name
                    .replace(
                        "numeric__",
                        "",
                    )
                )

                factor = (
                    business_factor_from_feature(
                        clean_name,
                        raw_row,
                    )
                )

                if factor is not None:
                    factors.append(
                        factor
                    )

            if len(factors) >= 5:
                break

    # ----------------------------------------------------------
    # Fallback business rules.
    #
    # This guarantees that an API response can still contain
    # useful explanations if SHAP doesn't generate a mapped
    # business factor.
    # ----------------------------------------------------------

    if not factors:

        if raw_row[
            "overdue_transaction_count"
        ] > 0:

            factors.append(
                "Overdue transactions"
            )

        if raw_row[
            "overdue_amount"
        ] > 0:

            factors.append(
                "Outstanding overdue amount"
            )

        if raw_row[
            "payment_failure_rate"
        ] > 0:

            factors.append(
                "Payment failures"
            )

        if raw_row[
            "complaint_count"
        ] > 0:

            factors.append(
                "Customer complaints"
            )

        if raw_row[
            "negative_sentiment_count"
        ] > 0:

            factors.append(
                "Negative customer interactions"
            )

    # ----------------------------------------------------------
    # Maximum five explanations
    # ----------------------------------------------------------

    return factors[:5]


# ==============================================================
# BUSINESS FEATURE EXPLANATIONS
# ==============================================================

def business_factor_from_feature(
    feature_name: str,
    row: pd.Series,
) -> str | None:

    # ----------------------------------------------------------
    # Payment failure rate
    # ----------------------------------------------------------

    if feature_name == "payment_failure_rate":

        value = float(
            row["payment_failure_rate"]
        )

        if value > 0:
            return (
                f"Payment failure rate is "
                f"{value:.1%}"
            )

    # ----------------------------------------------------------
    # Overdue amount
    # ----------------------------------------------------------

    if feature_name == "overdue_amount":

        value = float(
            row["overdue_amount"]
        )

        if value > 0:
            return (
                f"Overdue amount of "
                f"{value:,.2f}"
            )

    # ----------------------------------------------------------
    # Overdue count
    # ----------------------------------------------------------

    if feature_name == "overdue_transaction_count":

        value = int(
            row["overdue_transaction_count"]
        )

        if value > 0:
            return (
                f"{value} overdue transaction"
                + (
                    "s"
                    if value != 1
                    else ""
                )
            )

    # ----------------------------------------------------------
    # Failed payments
    # ----------------------------------------------------------

    if feature_name == "failed_transaction_count":

        value = int(
            row["failed_transaction_count"]
        )

        if value > 0:
            return (
                f"{value} failed payment"
                + (
                    "s"
                    if value != 1
                    else ""
                )
            )

    # ----------------------------------------------------------
    # Payment delay
    # ----------------------------------------------------------

    if feature_name == "average_payment_delay":

        value = float(
            row["average_payment_delay"]
        )

        if value > 0:
            return (
                f"Average payment delay of "
                f"{value:.1f} days"
            )

    # ----------------------------------------------------------
    # Complaints
    # ----------------------------------------------------------

    if feature_name == "complaint_count":

        value = int(
            row["complaint_count"]
        )

        if value > 0:
            return (
                f"{value} customer complaint"
                + (
                    "s"
                    if value != 1
                    else ""
                )
            )

    # ----------------------------------------------------------
    # Recent complaints
    # ----------------------------------------------------------

    if feature_name == "recent_complaint_count":

        value = int(
            row["recent_complaint_count"]
        )

        if value > 0:
            return (
                f"{value} recent complaint"
                + (
                    "s"
                    if value != 1
                    else ""
                )
            )

    # ----------------------------------------------------------
    # Negative sentiment
    # ----------------------------------------------------------

    if feature_name == "negative_sentiment_count":

        value = int(
            row["negative_sentiment_count"]
        )

        if value > 0:
            return (
                f"{value} negative customer interaction"
                + (
                    "s"
                    if value != 1
                    else ""
                )
            )

    # ----------------------------------------------------------
    # Escalations
    # ----------------------------------------------------------

    if feature_name == "escalation_count":

        value = int(
            row["escalation_count"]
        )

        if value > 0:
            return (
                f"{value} escalation"
                + (
                    "s"
                    if value != 1
                    else ""
                )
            )

    # ----------------------------------------------------------
    # Activity change
    # ----------------------------------------------------------

    if feature_name == "activity_change":

        value = float(
            row["activity_change"]
        )

        if value < -0.20:
            return (
                f"Customer activity decreased "
                f"by {abs(value):.1%}"
            )

    # ----------------------------------------------------------
    # Pending transactions
    # ----------------------------------------------------------

    if feature_name == "pending_transaction_count":

        value = int(
            row["pending_transaction_count"]
        )

        if value > 0:
            return (
                f"{value} pending transaction"
                + (
                    "s"
                    if value != 1
                    else ""
                )
            )

    return None
# ==============================================================
# APPLICATION
# ==============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global model_load_error

    try:
        load_model_artifacts()
    except Exception as exc:
        model_load_error = str(exc)
        logger.error(
            "Risk model not loaded: %s",
            exc,
        )

    yield


app = FastAPI(
    title="Customer Risk Intelligence API",
    description=(
        "Predicts 30-day high-risk probability for customers, "
        "explains the main risk factors and answers grounded "
        "questions about a customer."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(intelligence_router)
app.include_router(customers_router)


# ==============================================================
# ERROR HANDLERS
#
# Consistent {"detail": ...} bodies, never a stack trace.
# ==============================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):

    errors = [
        {
            "field": ".".join(
                str(part)
                for part in error["loc"]
                if part != "body"
                and not isinstance(part, int)
            )
            or "body",
            "message": error["msg"].removeprefix(
                "Value error, "
            ),
        }
        for error in exc.errors()
    ]

    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid request.",
            "errors": errors,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
):

    logger.exception(
        "Unhandled error on %s %s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."},
    )


# ==============================================================
# HEALTH CHECK
# ==============================================================

def check_database() -> str:

    from sqlalchemy import text

    from src.database.connection import engine

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return "ok"
    except Exception:
        return "unavailable"


def check_qdrant() -> str:

    from qdrant_client import QdrantClient

    from src.rag.retriever import COLLECTION_NAME

    try:
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
            timeout=2,
        )

        if not client.collection_exists(COLLECTION_NAME):
            return "collection_missing"

        return "ok"
    except Exception:
        return "unavailable"


@app.get("/health")
def health_check():

    components = {
        "model": (
            "ok"
            if model is not None
            else "unavailable"
        ),
        "database": check_database(),
        "vector_store": check_qdrant(),
        "llm": (
            "configured"
            if settings.OPENAI_API_KEY
            else "not_configured"
        ),
    }

    healthy = (
        components["model"] == "ok"
        and components["database"] == "ok"
        and components["vector_store"] == "ok"
        and components["llm"] == "configured"
    )

    return {
        "status": "ok" if healthy else "degraded",
        "components": components,
        "model": selected_model_name,
        "model_version": model_version,
    }


# ==============================================================
# PREDICT RISK
# ==============================================================

@app.post(
    "/predict-risk",
    response_model=RiskPredictionResponse,
    responses={
        404: {"description": "Customer not found"},
        422: {
            "description": (
                "Invalid customer_id, or customer has "
                "insufficient history for a prediction"
            )
        },
        503: {"description": "Risk model not loaded"},
    },
)
def predict_risk(
    request: RiskPredictionRequest,
):

    if model is None or feature_data is None:
        raise HTTPException(
            status_code=503,
            detail="Risk model is not available.",
        )

    customer_id = request.customer_id

    # ----------------------------------------------------------
    # Unknown customer -> 404.
    # ----------------------------------------------------------

    if customer_id not in customer_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Customer not found: {customer_id}",
        )

    customer_rows = feature_data[
        feature_data["customer_id"]
        == customer_id
    ]

    # ----------------------------------------------------------
    # Known customer without an observation snapshot (joined
    # after the last observation date). Don't make up a score.
    # ----------------------------------------------------------

    if customer_rows.empty:
        raise HTTPException(
            status_code=422,
            detail=(
                "ML risk prediction is unavailable because this "
                "customer does not have sufficient historical "
                "data for the current prediction framework."
            ),
        )

    # Latest available observation snapshot.
    row = customer_rows.sort_values(
        "observation_date"
    ).iloc[-1]

    # ----------------------------------------------------------
    # Same preprocessor as training. Target is never sent to
    # the model.
    # ----------------------------------------------------------

    X_transformed = preprocessor.transform(
        pd.DataFrame(
            [row[FEATURE_COLUMNS].to_dict()]
        )
    )

    probability = float(
        model.predict_proba(X_transformed)[0, 1]
    )

    return RiskPredictionResponse(
        customer_id=customer_id,
        observation_date=(
            row["observation_date"]
            .date()
            .isoformat()
        ),
        risk_probability=round(probability, 6),
        risk_level=get_risk_level(probability),
        risk_factors=generate_risk_factors(
            row,
            X_transformed,
        ),
        model_name=selected_model_name,
        model_version=model_version,
    )


# ==============================================================
# DEMO UI
# ==============================================================

if FRONTEND_DIR.exists():

    app.mount(
        "/ui",
        StaticFiles(
            directory=FRONTEND_DIR,
            html=True,
        ),
        name="ui",
    )

    @app.get("/", include_in_schema=False)
    def index():
        return RedirectResponse(url="/ui/")
