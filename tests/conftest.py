from pathlib import Path
from types import SimpleNamespace

import pytest


PROCESSED_DIR = Path("data/processed")

REQUIRED_ARTIFACTS = [
    PROCESSED_DIR / "customers" / "customers_clean.csv",
    PROCESSED_DIR / "transactions" / "transactions_clean.csv",
    PROCESSED_DIR / "interactions" / "interactions_clean.csv",
    PROCESSED_DIR / "documents" / "documents_clean.csv",
    PROCESSED_DIR / "ml" / "customer_features.csv",
    Path("models") / "model_selection.json",
]


@pytest.fixture(scope="session", autouse=True)
def require_pipeline_outputs():
    missing = [str(p) for p in REQUIRED_ARTIFACTS if not p.exists()]
    if missing:
        pytest.exit(
            "Pipeline outputs missing - run `make pipeline` first: "
            + ", ".join(missing),
            returncode=1,
        )


# --------------------------------------------------------------
# Fakes for the RAG layer, so reliability tests don't need
# OpenAI or Qdrant.
# --------------------------------------------------------------

class FakeRetriever:
    """Returns canned documents; records calls."""

    def __init__(self, documents_by_customer=None, error=None):
        self.documents_by_customer = documents_by_customer or {}
        self.error = error
        self.calls = []

    def retrieve(self, customer_id, query, plan=None):
        self.calls.append((customer_id, query))
        if self.error:
            raise self.error
        return list(self.documents_by_customer.get(customer_id, []))


class FakeLLM:
    """
    Mimics client.chat.completions.create. `reply` is either a
    string (message content) or an exception to raise.
    """

    def __init__(self, reply):
        self.reply = reply
        self.requests = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if isinstance(self.reply, Exception):
            raise self.reply
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def make_document(customer_id, document_id, document_type, content):
    return {
        "score": 0.8,
        "content": content,
        "customer_id": customer_id,
        "document_id": document_id,
        "document_type": document_type,
        "document_date": "2025-11-01",
        "source": "CRM",
    }


@pytest.fixture
def fakes():
    return SimpleNamespace(
        Retriever=FakeRetriever,
        LLM=FakeLLM,
        document=make_document,
    )
