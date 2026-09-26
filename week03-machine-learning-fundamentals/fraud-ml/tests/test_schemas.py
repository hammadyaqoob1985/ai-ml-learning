import pytest
from pydantic import ValidationError

from fraud_ml.schemas import TransactionRequest


def test_valid_transaction():
    transaction = TransactionRequest(
        amount=1200.0,
        age=35,
        international=True,
        transaction_type="transfer",
        country="FR",
    )

    assert transaction.amount == 1200.0
    assert transaction.transaction_type == "transfer"


@pytest.mark.parametrize(
    "invalid_field, invalid_value",
    [
        ("amount", -100),
        ("age", 15),
        ("age", 150),
        ("transaction_type", "unknown"),
        ("country", "France"),
    ],
)
def test_invalid_transaction(invalid_field, invalid_value):
    data = {
        "amount": 1200.0,
        "age": 35,
        "international": True,
        "transaction_type": "transfer",
        "country": "FR",
    }

    data[invalid_field] = invalid_value

    with pytest.raises(ValidationError):
        TransactionRequest(**data)

# Why should we validate transaction data before passing it into the ML pipeline, rather than relying on scikit-learn to reject invalid inputs?
# We should validate transaction data before passing it into the ML pipeline, as it helps to ensure that the data is in the correct format and meets the requirements of the ML pipeline. This is important because invalid data can lead to incorrect or unexpected results.

# There are three additional benefits to validating data before it reaches the ML pipeline:
# - Clear error messages: Pydantic can identify the invalid field, and FastAPI can return an HTTP 422 response instead of an unexpected scikit-learn exception.
# - Consistent input contracts: Every transaction must satisfy the same rules, whether it comes from an API, a test or another application.
# - Early rejection: Invalid transactions never reach the model, avoiding unnecessary processing.
# One important distinction: valid data isn't necessarily good data. A transaction of £1,000,000 may satisfy amount > 0 but still warrant investigation. Schema validation and data-quality monitoring solve different problems.