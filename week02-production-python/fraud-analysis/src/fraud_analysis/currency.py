def get_exchange_rate(from_currency: str, to_currency: str) -> float:
    # Imagine this calls an external API
    raise NotImplementedError


def calculate_converted_amount(
        amount: float,
        from_currency: str,
        to_currency: str,
) -> float:
    rate = get_exchange_rate(from_currency, to_currency)

    return amount * rate