import numpy as np
from fraud_ml.evaluate import evaluate_model


class FakeModel:
    def predict_proba(self, X):
        fraud_probabilities = np.array([0.2, 0.6, 0.9])
        return np.column_stack(
            [1 - fraud_probabilities, fraud_probabilities]
        )


def test_evaluation_respects_threshold():
    model = FakeModel()
    X = np.zeros((3, 1))
    y = np.array([False, True, True])

    results = evaluate_model(model, X, y, threshold=0.5)

    assert results["confusion_matrix"] == [[1, 0], [0, 2]]
    assert results["precision"] == 1.0
    assert results["recall"] == 1.0
    assert results["f1"] == 1.0

def test_higher_threshold_reduces_false_positives():
    model = FakeModel()
    X = np.zeros((3, 1))

    # Only the third transaction is actually fraudulent.
    y = np.array([False, False, True])

    low_threshold = evaluate_model(
        model, X, y, threshold=0.5
    )

    high_threshold = evaluate_model(
        model, X, y, threshold=0.8
    )

    assert low_threshold["confusion_matrix"] == [
        [1, 1],
        [0, 1],
    ]

    assert high_threshold["confusion_matrix"] == [
        [2, 0],
        [0, 1],
    ]

    assert high_threshold["precision"] > low_threshold["precision"]

# Learning question: In this particular example, recall stays at 100% when we increase the threshold. Does that mean raising a classification threshold will always preserve recall? Why or why not?
# No, raising the classification threshold will not always preserve recall. Recall measures the proportion of actual positive cases that are correctly identified. If the threshold is raised too high, some true positive cases may be classified as negative, leading to a decrease in recall. In this specific example, the model's predictions happened to align with the true labels such that increasing the threshold did not affect recall, but this is not guaranteed in general.
# One additional distinction: increasing the threshold cannot increase recall on a fixed set of scores. Recall either stays the same or decreases