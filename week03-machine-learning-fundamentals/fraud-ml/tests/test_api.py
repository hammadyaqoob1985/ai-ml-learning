
import numpy as np
from fastapi.testclient import TestClient

from fraud_ml.api import app
from fraud_ml.predict import predict_transaction
from unittest.mock import patch

from fastapi.testclient import TestClient

from fraud_ml.api import app

class FakeModel:
    def predict_proba(self, X):
        return np.array([[0.1, 0.9]])


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_transaction():
    # Supply a fake artifact without loading the real model.
    app.state.artifact = {
        "model": FakeModel(),
        "threshold": 0.67,
    }

    # Override startup for this isolated unit test.
    from fraud_ml.api import predict
    from fraud_ml.schemas import TransactionRequest
    from starlette.requests import Request

    transaction = TransactionRequest(
        amount=1200.0,
        age=35,
        international=True,
        transaction_type="transfer",
        country="FR",
    )

    # Exercise the prediction function directly.
    result = predict_transaction(
        app.state.artifact,
        transaction.model_dump(),
    )

    assert result["fraud"] is True
    assert result["probability"] == 0.9


from unittest.mock import patch

from fastapi.testclient import TestClient

from fraud_ml.api import app


def test_predict_endpoint():
    fake_artifact = {
        "model": FakeModel(),
        "threshold": 0.67,
    }

    with patch("fraud_ml.api.load_model", return_value=fake_artifact):
        with TestClient(app) as client:
            response = client.post(
                "/predict",
                json={
                    "amount": 1200.0,
                    "age": 35,
                    "international": True,
                    "transaction_type": "transfer",
                    "country": "FR",
                },
            )

    assert response.status_code == 200
    assert response.json() == {
        "fraud": True,
        "probability": 0.9,
    }


def test_predict_endpoint_rejects_invalid_amount():
    fake_artifact = {
        "model": FakeModel(),
        "threshold": 0.67,
    }

    with patch("fraud_ml.api.load_model", return_value=fake_artifact):
        with TestClient(app) as client:
            response = client.post(
                "/predict",
                json={
                    "amount": -100,
                    "age": 35,
                    "international": True,
                    "transaction_type": "transfer",
                    "country": "FR",
                },
            )

    assert response.status_code == 422

# Why is an HTTP endpoint test valuable when we already have unit tests for predict_transaction() and TransactionRequest?
# E2E testing is valuable when we want to test the entire system end-to-end, including the HTTP layer.
# One distinction: these are API integration tests, rather than full end-to-end tests. We're replacing the real model with a fake one, so we're testing FastAPI, Pydantic validation, routing and response serialization without depending on the trained model.