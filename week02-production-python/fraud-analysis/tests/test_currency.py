from unittest.mock import patch

from fraud_analysis.currency import calculate_converted_amount


def test_calculate_converted_amount():
    with patch(
            "fraud_analysis.currency.get_exchange_rate",
            return_value=0.80,
    ) as mock_get_rate:

        result = calculate_converted_amount(
            200.0,
            "USD",
            "GBP",
        )

    assert result == 160.0
    mock_get_rate.assert_called_once_with("USD", "GBP")

