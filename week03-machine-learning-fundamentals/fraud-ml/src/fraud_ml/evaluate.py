from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)


def evaluate_model(model, X_test, y_test, threshold=0.5):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = probabilities >= threshold

    return {
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "average_precision": average_precision_score(
            y_test, probabilities
        ),
    }