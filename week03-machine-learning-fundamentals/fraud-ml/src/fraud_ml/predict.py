from pathlib import Path

import joblib
import pandas as pd

def save_model(model, threshold: float, path: str | Path) -> None:
    artifact = {
        "model": model,
        "threshold": threshold,
    }

    joblib.dump(artifact, path)


def load_model(path: str | Path) -> dict:
    return joblib.load(path)

def predict_transaction(artifact: dict, transaction: dict) -> dict:
    model = artifact["model"]
    threshold = artifact["threshold"]

    transaction_df = pd.DataFrame([transaction])

    fraud_probability = float(
        model.predict_proba(transaction_df)[0, 1]
    )

    return {
        "fraud": bool(fraud_probability >= threshold),
        "probability": fraud_probability,
    }