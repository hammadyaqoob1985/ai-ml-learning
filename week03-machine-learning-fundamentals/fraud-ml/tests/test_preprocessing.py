import numpy as np
import pandas as pd

from fraud_ml.preprocessing import create_preprocessor


def test_preprocessor_handles_unknown_categories():
    training_data = pd.DataFrame({
        "amount": [100, 200, 300],
        "age": [25, 40, 55],
        "international": [False, True, False],
        "transaction_type": ["card", "transfer", "cash"],
        "country": ["UK", "FR", "DE"],
    })

    new_data = pd.DataFrame({
        "amount": [150],
        "age": [35],
        "international": [True],
        "transaction_type": ["crypto"],
        "country": ["IT"],
    })

    preprocessor = create_preprocessor()

    training_transformed = preprocessor.fit_transform(training_data)
    new_transformed = preprocessor.transform(new_data)

    assert new_transformed.shape == (
        1,
        training_transformed.shape[1],
    )

    assert np.isfinite(new_transformed).all()

def test_preprocessor_uses_training_statistics():
    training_data = pd.DataFrame({
        "amount": [100, 200, 300],
        "age": [20, 30, 40],
        "international": [False, True, False],
        "transaction_type": ["card", "transfer", "cash"],
        "country": ["UK", "FR", "DE"],
    })

    preprocessor = create_preprocessor()
    preprocessor.fit(training_data)

    scaler = preprocessor.named_transformers_["numerical"]

    assert np.allclose(scaler.mean_, [200, 30])

    new_data = training_data.copy()
    new_data["amount"] = [10000, 20000, 30000]

    preprocessor.transform(new_data)

    assert np.allclose(scaler.mean_, [200, 30])

# Why would it be dangerous if the scaler recalculated its mean every time your fraud-detection API received a new transaction?
# It would be dangerous because the mean would be influenced by each new transaction, potentially skewing the scaling and leading to inconsistent predictions.