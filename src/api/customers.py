"""
Customer history for the demo UI, served from PostgreSQL
(loaded by src.database.load_processed).
"""

import logging

from fastapi import APIRouter, HTTPException, Path
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from src.api.validation import normalize_customer_id
from src.database.connection import SessionLocal
from src.database.models import (
    Customer,
    CustomerDocument,
    Interaction,
    Transaction,
)


logger = logging.getLogger("api.customers")

RECENT_LIMIT = 15


router = APIRouter(
    prefix="/customers",
    tags=["Customers"],
)


def _date(value):
    return value.isoformat() if value is not None else None


@router.get(
    "/{customer_id}",
    responses={
        404: {"description": "Customer not found"},
        422: {"description": "Invalid customer_id"},
        503: {"description": "Database unavailable"},
    },
)
def get_customer(
    customer_id: str = Path(
        ...,
        examples=["CUST_00004"],
    ),
):

    try:
        customer_id = normalize_customer_id(customer_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )

    try:

        with SessionLocal() as session:

            customer = session.get(Customer, customer_id)

            if customer is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Customer not found: {customer_id}",
                )

            payment_summary = dict(
                session.execute(
                    select(
                        Transaction.payment_status,
                        func.count(),
                    )
                    .where(Transaction.customer_id == customer_id)
                    .group_by(Transaction.payment_status)
                ).all()
            )

            transactions = session.scalars(
                select(Transaction)
                .where(Transaction.customer_id == customer_id)
                .order_by(Transaction.transaction_date.desc())
                .limit(RECENT_LIMIT)
            ).all()

            interactions = session.scalars(
                select(Interaction)
                .where(Interaction.customer_id == customer_id)
                .order_by(Interaction.interaction_date.desc())
                .limit(RECENT_LIMIT)
            ).all()

            documents = session.scalars(
                select(CustomerDocument)
                .where(CustomerDocument.customer_id == customer_id)
                .order_by(CustomerDocument.document_date.desc())
            ).all()

            return {
                "customer": {
                    "customer_id": customer.customer_id,
                    "customer_name": customer.customer_name,
                    "customer_since": _date(customer.customer_since),
                    "customer_segment": customer.customer_segment,
                    "industry": customer.industry,
                    "country": customer.country,
                    "account_status": customer.account_status,
                },
                "payment_summary": payment_summary,
                "recent_transactions": [
                    {
                        "transaction_id": t.transaction_id,
                        "transaction_date": _date(t.transaction_date),
                        "transaction_amount": float(t.transaction_amount),
                        "payment_status": t.payment_status,
                        "due_date": _date(t.due_date),
                        "payment_date": _date(t.payment_date),
                    }
                    for t in transactions
                ],
                "recent_interactions": [
                    {
                        "interaction_id": i.interaction_id,
                        "interaction_date": _date(i.interaction_date),
                        "interaction_type": i.interaction_type,
                        "channel": i.channel,
                        "sentiment": i.sentiment,
                        "resolution_status": i.resolution_status,
                    }
                    for i in interactions
                ],
                "documents": [
                    {
                        "document_id": d.document_id,
                        "document_type": d.document_type,
                        "document_date": _date(d.document_date),
                        "source": d.source,
                        "content": d.content,
                    }
                    for d in documents
                ],
            }

    except SQLAlchemyError:
        logger.exception("Database error for %s", customer_id)
        raise HTTPException(
            status_code=503,
            detail="Database is unavailable.",
        )
