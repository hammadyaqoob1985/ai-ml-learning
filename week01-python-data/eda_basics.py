import pandas as pd

transactions = pd.DataFrame({
    "amount": [
        25, 50, 75, 20, 500,
        35, 1200, 60, 45, 200,
        2500, 30, 90, 150, 3000,
        40, 65, 110, 1800, 55
    ],
    "customer_age": [
        25, 42, 31, 55, 28,
        36, 44, 23, 39, 50,
        34, 29, 47, 41, 32,
        58, 26, 45, 37, 52
    ],
    "transaction_type": [
        "card", "card", "transfer", "card", "transfer",
        "card", "transfer", "card", "card", "transfer",
        "transfer", "card", "card", "transfer", "transfer",
        "card", "card", "transfer", "transfer", "card"
    ],
    "international": [
        False, False, False, False, True,
        False, True, False, False, False,
        True, False, False, False, True,
        False, False, False, True, False
    ],
    "fraud": [
        0, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        1, 0, 0, 0, 1,
        0, 0, 0, 1, 0
    ]
})
# Use Pandas to answer these questions:
# How many rows and columns are there?
print(f"Rows and columns: {transactions.shape}")

# Are there any missing values?
print(f"Missing values:\n{transactions.isna().sum()}")

# Are there any duplicate rows?
print(f"Duplicate rows: {transactions.duplicated().sum()}")

# What are the descriptive statistics for the numerical columns?
print(f"Descriptive statistics:\n{transactions.describe()}")

# How many fraudulent vs legitimate transactions are there?
fraud_counts = transactions["fraud"].value_counts()
print(f"Fraudulent vs legitimate transactions:\n{fraud_counts}")

# What percentage of transactions are fraudulent?
fraud_percentage = transactions["fraud"].value_counts(normalize=True)[1] * 100
print(f"Percentage of fraudulent transactions: {fraud_percentage:.2f}%")

# What is the average transaction amount for fraudulent vs legitimate transactions?
average_amounts = transactions.groupby("fraud")["amount"].mean()
print(f"Average transaction amounts:\n{average_amounts}")

# What is the median transaction amount for fraudulent vs legitimate transactions?
median_amounts = transactions.groupby("fraud")["amount"].median()
print(f"Median transaction amounts:\n{median_amounts}")

# How does fraud break down by transaction_type?
fraud_by_type = transactions.groupby("transaction_type")["fraud"].value_counts().unstack()
print(f"Fraud breakdown by transaction type:\n{fraud_by_type}")

# How does fraud break down by international vs domestic transactions?
fraud_by_international = transactions.groupby("international")["fraud"].value_counts().unstack()
print(f"Fraud breakdown by international vs domestic transactions:\n{fraud_by_international}")

# Challenge
#
# Try producing this:
#
# count    mean_amount    median_amount
# fraud
# 0                       ...
# 1                       ...
#
# using one groupby() operation

fraud_stats = transactions.groupby("fraud").agg(
    count=("amount", "count"),
    mean_amount=("amount", "mean"),
    median_amount=("amount", "median")
)
print(f"Fraud statistics:\n{fraud_stats}")

# Then, before we introduce charts, write 3–5 observations about the dataset in plain English.
#
# For example, don't just tell me:
#
# fraud mean = X
# non-fraud mean = Y
#
# Tell me what that suggests:
#
# "Fraudulent transactions appear to have substantially higher transaction values than legitimate transactions in this sample."
#
# Observations:
# 1. The dataset is very small, with only 20 transactions, so any pattern here should be treated as a rough example rather than a strong conclusion.
# 2. Fraud seems to be associated with larger transaction amounts, because the fraudulent rows include some of the biggest payments in the sample.
# 3. International transactions appear to show up more often among fraudulent cases than domestic ones.
# 4. transfer transactions both include fraudulent examples where as cards dont.
# 5. The customer ages are spread across a wide range, which suggests age is not an obvious single-factor explanation for fraud in this sample.

