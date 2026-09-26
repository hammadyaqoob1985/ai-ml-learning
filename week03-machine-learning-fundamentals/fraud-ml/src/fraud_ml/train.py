from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fraud_ml.preprocessing import create_preprocessor


def create_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )