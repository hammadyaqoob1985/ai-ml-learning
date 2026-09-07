import pytest

from fraud_analysis.analysis import calculate_average_amount


@pytest.mark.parametrize(
    "amounts, expected",
    [
        ([100.0, 200.0], 150.0),
        ([50.0, 100.0, 150.0], 100.0),
        ([10.0], 10.0),
    ],
)
def test_calculate_average_amount(amounts, expected):
    result = calculate_average_amount(amounts)

    assert result == expected


def test_calculate_average_amount_empty():
    result = calculate_average_amount([])

    assert result is None
