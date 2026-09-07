import pytest
from pydantic import ValidationError

from fraud_analysis.models import TransactionRequest


def test_valid_transaction_request():
    transaction = TransactionRequest(
        transaction_id=101,
        amount=250.50,
        age=40,
        country="UK",
        international=False,
    )

    assert transaction.transaction_id == 101
    assert transaction.amount == 250.50


def test_transaction_rejects_negative_amount():
    with pytest.raises(ValidationError):
        TransactionRequest(
            transaction_id=101,
            amount=-50.0,
            age=40,
            country="UK",
            international=False,
        )


def test_transaction_rejects_age_above_maximum():
    with pytest.raises(ValidationError):
        TransactionRequest(
            transaction_id=101,
            amount=50.0,
            age=150,
            country="UK",
            international=False,
        )
