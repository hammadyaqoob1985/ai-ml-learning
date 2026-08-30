import pandas as pd
import numpy as np

customers = pd.DataFrame({
    "customer_id": [101, 102, 103, 104, 104, 105, 106, 107],
    "age": [25, 42, np.nan, 55, 55, 28, 46, -5],
    "salary": [30000, 72000, 48000, 90000, 90000, np.nan, 68000, 5000000],
    "country": ["UK", "uk", "France", "UK", "UK", "Germany", " france ", "UK"],
    "active": ["Yes", "Yes", "No", "Yes", "Yes", "No", "Yes", "No"]
})

print(customers)

customers.info()
print(customers.isna().sum())
print(customers.duplicated())
print(customers.describe())

clean_customers = customers.copy()

#Remove duplicate rows.
clean_customers = clean_customers.drop_duplicates()

print(clean_customers)

# Convert negative ages to NaN.
clean_customers.loc[clean_customers["age"] < 0, "age"] = np.nan

print(clean_customers)

# Fill missing ages using the median age.
clean_customers["age"] = clean_customers["age"].fillna(clean_customers["age"].median())
print(clean_customers)

# Fill the missing salary using the median salary.
clean_customers["salary"] = clean_customers["salary"].fillna(clean_customers["salary"].median())
print(clean_customers)

# Remove leading/trailing whitespace from countries.
clean_customers["country"] = (
    clean_customers["country"]
    .str.strip()
    .str.upper()
)
print(clean_customers)


# Convert active from "Yes"/"No" to actual Boolean True/False.
clean_customers["active"] = clean_customers["active"].map({
    "Yes": True,
    "No": False
})
print(clean_customers)

print(clean_customers.isna().sum())

print(clean_customers.dtypes)

print(clean_customers["country"].unique())

print(clean_customers.describe())