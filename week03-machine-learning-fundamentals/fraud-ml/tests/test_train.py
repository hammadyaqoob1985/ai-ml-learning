import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fraud_ml.generate_data import generate_fraud_data
from fraud_ml.train import create_model


def test_create_model():
    model = create_model()

    assert isinstance(model, Pipeline)
    assert isinstance(
        model.named_steps["classifier"],
        LogisticRegression,
    )
    assert model.named_steps["classifier"].class_weight == "balanced"
    assert model.named_steps["classifier"].max_iter == 1000

def test_model_can_train_and_predict():
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

    probabilities = model.predict_proba(X.head(5))

    assert probabilities.shape == (5, 2)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.allclose(probabilities.sum(axis=1), 1)