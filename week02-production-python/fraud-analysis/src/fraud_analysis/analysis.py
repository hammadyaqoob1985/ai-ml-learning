def calculate_average_amount(amounts: list[float]) -> float | None:
    if not amounts:
        return None

    return sum(amounts) / len(amounts)