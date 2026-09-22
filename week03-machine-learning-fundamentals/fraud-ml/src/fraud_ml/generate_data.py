import numpy as np
import pandas as pd


def generate_fraud_data(
        number_of_transactions: int = 10_000,
        random_state: int = 42,
) -> pd.DataFrame:

    rng = np.random.default_rng(random_state)

    amount = rng.lognormal(
        mean=4.5,
        sigma=1.0,
        size=number_of_transactions,
    )

    age = rng.integers(
        18,
        80,
        size=number_of_transactions,
    )

    transaction_type = rng.choice(
        ["card", "transfer", "cash"],
        size=number_of_transactions,
        p=[0.6, 0.3, 0.1],
    )

    country = rng.choice(
        ["UK", "FR", "DE", "ES"],
        size=number_of_transactions,
        p=[0.55, 0.15, 0.15, 0.15],
    )

    international = rng.choice(
        [False, True],
        size=number_of_transactions,
        p=[0.85, 0.15],
    )

    fraud_probability = (
            0.01
            + (amount > 500) * 0.04
            + international * 0.10
            + (transaction_type == "transfer") * 0.05
    )

    fraud_probability = np.clip(
        fraud_probability,
        0,
        0.95,
    )

    fraud = rng.random(number_of_transactions) < fraud_probability

    return pd.DataFrame(
        {
            "amount": amount.round(2),
            "age": age,
            "international": international,
            "transaction_type": transaction_type,
            "country": country,
            "fraud": fraud,
        }
    )