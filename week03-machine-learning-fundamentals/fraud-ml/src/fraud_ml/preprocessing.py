from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["country", "transaction_type"],
            ),
            (
                "numerical",
                StandardScaler(),
                ["amount", "age"],
            ),
        ],
        remainder="passthrough",
    )