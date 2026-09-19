from sklearn.preprocessing import OneHotEncoder
import pandas as pd

df = pd.DataFrame({
    "amount": [
        50, 1200, 25, 7500, 300,
        9800, 75, 2200, 150, 6100,
        45, 3200
    ],
    "age": [
        45, 22, 37, 29, 61,
        24, 52, 33, 41, 27,
        58, 31
    ],
    "international": [
        False, True, False, True, False,
        True, False, True, False, True,
        False, True
    ],
    "transaction_type": [
        "card", "transfer", "card", "transfer",
        "cash", "transfer", "card", "card",
        "cash", "transfer", "card", "transfer"
    ],
    "country": [
        "UK", "FR", "UK", "DE",
        "UK", "ES", "UK", "FR",
        "UK", "DE", "UK", "ES"
    ],
    "fraud": [
        False, True, False, True,
        False, True, False, False,
        False, True, False, True
    ]
})
encoder = OneHotEncoder(sparse_output=False)

encoded = encoder.fit_transform(df[["country", "transaction_type"]])

print(encoded)

print(
    encoder.get_feature_names_out(
        ["country", "transaction_type"]
    )
)

X = df[["amount",
        "age",
        "international",
        "transaction_type",
        "country"]]
y = df["fraud"]

encoder = OneHotEncoder(sparse_output=False)

encoded = encoder.fit_transform(df[["country", "transaction_type"]])

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)

print("\nEncoded values:")
print(encoded)

print("\nEncoded feature names:")

print(
    encoder.get_feature_names_out(
        ["country", "transaction_type"]
    )
)
# 1. Why would converting UK → 1, FR → 2, DE → 3 be potentially misleading to an ML algorithm?
# this is potentially misleading because it implies an ordinal relationship between the countries, which does not exist. The algorithm may interpret higher numbers as having more importance or value, which is not the case for categorical data like country names.

# 2. What is the difference between fit() and transform() for the encoder?
# fit() is used to learning the preprocessing rules from the training data, while transform() is used to transform new data.