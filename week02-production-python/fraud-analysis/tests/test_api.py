from fastapi.testclient import TestClient

from fraud_analysis.api import app


client = TestClient(app)

def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_transaction():
    response = client.post(
        "/transactions",
        json={
            "transaction_id": 101,
            "amount": 250.5,
            "age": 40,
            "country": "UK",
            "international": False,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "transaction_id": 101,
        "fraud": False,
        "probability": 0.15,
    }


def test_invalid_transaction_returns_422():
    response = client.post(
        "/transactions",
        json={
            "transaction_id": 101,
            "amount": -50,
            "age": 40,
            "country": "UK",
            "international": False,
        },
    )

    assert response.status_code == 422
