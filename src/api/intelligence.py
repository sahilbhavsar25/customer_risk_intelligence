import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.api.validation import (
    normalize_customer_id,
    normalize_question,
)
from src.rag.customer_intelligence import (
    STATUS_INVALID_LLM_RESPONSE,
    STATUS_LLM_UNAVAILABLE,
    STATUS_NOT_FOUND,
    CustomerIntelligenceService,
)


logger = logging.getLogger("api.intelligence")


# ==============================================================
# ROUTER
# ==============================================================

router = APIRouter(
    prefix="",
    tags=["Customer Intelligence"],
)


# ==============================================================
# REQUEST / RESPONSE MODELS
# ==============================================================

class CustomerIntelligenceRequest(BaseModel):
    customer_id: str = Field(
        ...,
        description="Customer identifier",
        examples=["CUST_00123"],
    )

    question: str = Field(
        ...,
        description="Question about the customer",
        examples=[
            "What payment problems has this customer experienced?"
        ],
    )

    @field_validator("customer_id", mode="before")
    @classmethod
    def validate_customer_id(cls, value):
        return normalize_customer_id(value)

    @field_validator("question", mode="before")
    @classmethod
    def validate_question(cls, value):
        return normalize_question(value)


class Source(BaseModel):
    document_id: str
    document_type: str | None = None
    document_date: str | None = None
    source: str | None = None


class CustomerIntelligenceResponse(BaseModel):
    customer_id: str
    question: str
    answer: str
    status: str
    grounded: bool
    sources: list[Source]
    error: str | None = None


# ==============================================================
# SERVICE
#
# Created lazily. If creation fails (no API key, Qdrant down,
# collection missing) the request gets a 503 and the next
# request tries again.
# ==============================================================

_intelligence_service = None


def get_service() -> CustomerIntelligenceService:

    global _intelligence_service

    if _intelligence_service is None:

        try:
            _intelligence_service = (
                CustomerIntelligenceService()
            )
        except Exception:
            logger.exception(
                "Customer intelligence service failed to start"
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    "Customer intelligence service is not "
                    "available."
                ),
            )

    return _intelligence_service


# ==============================================================
# CUSTOMER INTELLIGENCE ENDPOINT
# ==============================================================

STATUS_TO_HTTP = {
    STATUS_NOT_FOUND: 404,
    STATUS_LLM_UNAVAILABLE: 503,
    STATUS_INVALID_LLM_RESPONSE: 502,
}


@router.post(
    "/customer-intelligence",
    response_model=CustomerIntelligenceResponse,
    responses={
        404: {"description": "Customer not found"},
        422: {"description": "Invalid customer_id or question"},
        502: {"description": "Invalid response from the LLM"},
        503: {"description": "LLM or service unavailable"},
    },
)
def customer_intelligence(
    request: CustomerIntelligenceRequest,
):

    service = get_service()

    result = service.ask(
        customer_id=request.customer_id,
        question=request.question,
    )

    status_code = STATUS_TO_HTTP.get(result["status"])

    if status_code is not None:
        raise HTTPException(
            status_code=status_code,
            detail=result["answer"],
        )

    return CustomerIntelligenceResponse(
        customer_id=result["customer_id"],
        question=result["question"],
        answer=result["answer"],
        status=result["status"],
        grounded=result["grounded"],
        sources=result["sources"],
        error=result.get("error"),
    )
