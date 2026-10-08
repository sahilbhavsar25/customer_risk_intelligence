import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import intelligence
from src.api.main import app
from src.rag.customer_intelligence import CustomerIntelligenceService


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def customer_ids():
    customers = set(pd.read_csv("data/processed/customers/customers_clean.csv")["customer_id"])
    with_snapshot = set(pd.read_csv("data/processed/ml/customer_features.csv")["customer_id"])
    return {
        "with_snapshot": sorted(with_snapshot)[0],
        "without_snapshot": sorted(customers - with_snapshot)[0],
    }


@pytest.fixture
def fake_service(fakes, monkeypatch):
    """Swap the real (OpenAI + Qdrant) service for a fake one."""

    def install(reply):
        svc = CustomerIntelligenceService(
            retriever=fakes.Retriever(),
            llm_client=fakes.LLM(reply),
        )
        monkeypatch.setattr(intelligence, "_intelligence_service", svc)
        return svc

    return install


# --------------------------------------------------------------
# health
# --------------------------------------------------------------

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert body["components"]["model"] == "ok"


# --------------------------------------------------------------
# /predict-risk
# --------------------------------------------------------------

def test_predict_risk(client, customer_ids):
    response = client.post("/predict-risk", json={"customer_id": customer_ids["with_snapshot"]})
    assert response.status_code == 200

    body = response.json()
    assert 0 <= body["risk_probability"] <= 1
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert isinstance(body["risk_factors"], list)
    assert body["model_version"]


def test_predict_risk_normalizes_id(client, customer_ids):
    raw = f"  {customer_ids['with_snapshot'].lower()} "
    response = client.post("/predict-risk", json={"customer_id": raw})
    assert response.status_code == 200
    assert response.json()["customer_id"] == customer_ids["with_snapshot"]


def test_predict_risk_without_snapshot(client, customer_ids):
    response = client.post("/predict-risk", json={"customer_id": customer_ids["without_snapshot"]})
    assert response.status_code == 422
    assert "unavailable" in response.json()["detail"]


@pytest.mark.parametrize(
    "body, status",
    [
        ({"customer_id": "CUST_99999"}, 404),
        ({"customer_id": "not-an-id"}, 422),
        ({"customer_id": ""}, 422),
        ({}, 422),
        ({"customer_id": 123}, 422),
    ],
)
def test_predict_risk_errors(client, body, status):
    response = client.post("/predict-risk", json=body)
    assert response.status_code == status
    assert "Traceback" not in response.text


def test_invalid_json_body(client):
    response = client.post(
        "/predict-risk",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_missing_model_returns_503(client, monkeypatch):
    import src.api.main as main
    monkeypatch.setattr(main, "model", None)
    response = client.post("/predict-risk", json={"customer_id": "CUST_00001"})
    assert response.status_code == 503


# --------------------------------------------------------------
# /customer-intelligence
# --------------------------------------------------------------

def test_customer_intelligence(client, fake_service, customer_ids):
    fake_service(json.dumps({"answer": "Account is active.", "sources": [], "grounded": True}))

    response = client.post(
        "/customer-intelligence",
        json={
            "customer_id": customer_ids["with_snapshot"],
            "question": "What is the account status?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Account is active."
    assert body["grounded"] is True
    assert body["sources"] == []


@pytest.mark.parametrize(
    "body",
    [
        {"customer_id": "CUST_00001", "question": "   "},
        {"customer_id": "CUST_00001"},
        {"customer_id": "bad", "question": "hi"},
        {"customer_id": "CUST_00001", "question": "x" * 501},
    ],
)
def test_customer_intelligence_validation(client, body):
    response = client.post("/customer-intelligence", json=body)
    assert response.status_code == 422


def test_customer_intelligence_unknown_customer(client, fake_service):
    fake_service("{}")
    response = client.post(
        "/customer-intelligence",
        json={"customer_id": "CUST_99999", "question": "Anything?"},
    )
    assert response.status_code == 404


def test_customer_intelligence_llm_down(client, fake_service, customer_ids):
    fake_service(TimeoutError("timed out"))
    response = client.post(
        "/customer-intelligence",
        json={"customer_id": customer_ids["with_snapshot"], "question": "Summarize the risk"},
    )
    assert response.status_code == 503
    assert "Traceback" not in response.text


def test_customer_intelligence_invalid_llm_response(client, fake_service, customer_ids):
    fake_service("definitely not json")
    response = client.post(
        "/customer-intelligence",
        json={"customer_id": customer_ids["with_snapshot"], "question": "Summarize the risk"},
    )
    assert response.status_code == 502


def test_customer_intelligence_service_cannot_start(client, monkeypatch):
    def broken():
        raise RuntimeError("qdrant unreachable")

    monkeypatch.setattr(intelligence, "_intelligence_service", None)
    monkeypatch.setattr(intelligence, "CustomerIntelligenceService", broken)

    response = client.post(
        "/customer-intelligence",
        json={"customer_id": "CUST_00001", "question": "Summarize the risk"},
    )
    assert response.status_code == 503
    assert "qdrant" not in response.text.lower()
