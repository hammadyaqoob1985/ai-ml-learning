import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

n = 1000

data = pd.DataFrame({
    "transaction_id": range(10001, 10001 + n),

    "amount": np.round(
        np.random.lognormal(mean=4.5, sigma=1.0, size=n),
        2
    ),

    "customer_age": np.random.randint(18, 80, size=n),

    "transaction_type": np.random.choice(
        ["card", "transfer", "cash"],
        size=n,
        p=[0.6, 0.3, 0.1]
    ),

    "country": np.random.choice(
        ["UK", "France", "Germany", "Spain"],
        size=n,
        p=[0.55, 0.15, 0.15, 0.15]
    ),

    "international": np.random.choice(
        [False, True],
        size=n,
        p=[0.85, 0.15]
    )
})

fraud_probability = (
        0.01
        + (data["amount"] > 500) * 0.12
        + (data["international"]) * 0.08
        + (data["transaction_type"] == "transfer") * 0.05
)

data["fraud"] = (
        np.random.random(n) < fraud_probability
).astype(int)

data.loc[[25, 300, 725], "customer_age"] = np.nan

data.loc[[100, 450], "amount"] = np.nan

data.loc[50, "country"] = " uk "
data.loc[600, "country"] = "FRANCE"

data = pd.concat(
    [data, data.iloc[[10, 20]]],
    ignore_index=True
)
# get columns in data
print(data.columns.tolist())

########PART A################
######## PART A — UNDERSTAND THE DATASET ########

# Dataset size
print("\nDataset shape:")
print(data.shape)

# Columns and data types
print("\nData types:")
print(data.dtypes)

# Missing values
print("\nMissing values:")
print(data.isna().sum())

# Duplicate observations
print("\nDuplicate observations:")
duplicates = data[
    data.duplicated(keep=False)
].sort_values("transaction_id")

print(duplicates)

# Descriptive statistics
print("\nDescriptive statistics:")
print(data.describe())

# Categorical values
print("\nTransaction types:")
print(data["transaction_type"].value_counts())

print("\nCountries:")
print(data["country"].value_counts())
print("Unique countries:", data["country"].unique())

print("\nInternational:")
print(data["international"].value_counts())

print("\nFraud:")
print(data["fraud"].value_counts())

########PART B################
clean_data = data.copy()
# Duplicates:
# I would remove the second version of the duplicate because it has the same values as the first version, and keeping duplicates could skew the analysis.

clean_data = clean_data.drop_duplicates()
# Missing ages:
# I would calculate the median age and fill the missing values because it would provide a reasonable estimate without significantly skewing the data.
#
clean_data["customer_age"] = clean_data["customer_age"].fillna(clean_data["customer_age"].median())
# Missing amounts:
# I would calculate the median amount and fill the missing values because it would provide a reasonable estimate without significantly skewing the data.
#
clean_data["amount"] = clean_data["amount"].fillna(clean_data["amount"].median())
# Country whitespace:
# I would strip the whitespace from the country names because it would ensure consistency and prevent issues with grouping or analysis.
#
# Country case:
# I would convert the country names to uppercase because it would ensure consistency and prevent issues with grouping or analysis.

clean_data["country"] = (clean_data["country"].str.strip().str.upper())


# Large transaction amounts:
# I would investigate transactions with unusually large amounts because they could be indicative of fraud or data entry errors.

print("\nShape after cleaning:")
print(clean_data.shape)

# Columns and data types
print("\nData types cleaned:")
print(clean_data.dtypes)

# Missing values
print("\nMissing values cleaned:")
print(clean_data.isna().sum())

# Duplicate observations
print("\nDuplicate count:")
print(clean_data.duplicated().sum())

# Descriptive statistics
print("\nDescriptive statistics cleaned:")
print(clean_data.describe())

print("\nUnique countries:")
print(clean_data["country"].unique())

print("\nLargest transaction amounts:")
print(
    clean_data.sort_values(
        by="amount",
        ascending=False
    ).head(10)
)