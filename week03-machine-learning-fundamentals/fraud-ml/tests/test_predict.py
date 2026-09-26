import numpy as np

from fraud_ml.generate_data import generate_fraud_data
from fraud_ml.train import create_model
from fraud_ml.predict import save_model, load_model, predict_transaction


def test_saved_model_preserves_predictions(tmp_path):
    df = generate_fraud_data(
        number_of_transactions=1000,
        random_state=42,
    )

    X = df[
        [
            "amount",
            "age",
            "international",
            "transaction_type",
            "country",
        ]
    ]
    y = df["fraud"]

    model = create_model()
    model.fit(X, y)

    original_probabilities = model.predict_proba(X.head(5))

    artifact_path = tmp_path / "fraud_model.joblib"
    save_model(model, 0.67, artifact_path)

    artifact = load_model(artifact_path)

    loaded_probabilities = artifact["model"].predict_proba(
        X.head(5)
    )

    assert artifact["threshold"] == 0.67
    assert np.allclose(
        original_probabilities,
        loaded_probabilities,
    )

# Why is it important to save the entire fitted pipeline rather than saving only the logistic regression classifier?
# It is important to save the entire fitted pipeline because the logistic regression classifier is not fitted on the entire dataset. The pipeline may include preprocessing steps such as scaling, encoding, or feature engineering that are essential for the model to make correct predictions. Saving only the classifier would omit these steps, potentially leading to incorrect or inconsistent predictions when the model is loaded and used later.
# One small clarification: the logistic regression classifier is fitted, but it's fitted on the preprocessed features, not the original transaction data.

def test_predict_transaction():
    df = generate_fraud_data(
        number_of_transactions=1000,
        random_state=42,
    )

    X = df[
        [
            "amount",
            "age",
            "international",
            "transaction_type",
            "country",
        ]
    ]
    y = df["fraud"]

    model = create_model()
    model.fit(X, y)

    artifact = {
        "model": model,
        "threshold": 0.67,
    }

    transaction = {
        "amount": 1200.0,
        "age": 35,
        "international": True,
        "transaction_type": "transfer",
        "country": "FR",
    }

    result = predict_transaction(artifact, transaction)

    assert isinstance(result["fraud"], bool)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["fraud"] == (
            result["probability"] >= artifact["threshold"]
    )