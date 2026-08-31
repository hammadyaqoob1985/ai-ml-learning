import matplotlib.pyplot as plt
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

# Histogram of transaction amounts.
plt.hist(transactions["amount"], bins=10)

plt.xlabel("Transaction amount (£)")
plt.ylabel("Frequency")
plt.title("Distribution of transaction amounts")
plt.show()

# Box plot of transaction amounts.

plt.boxplot(transactions["amount"])

plt.ylabel("Transaction amount (£)")
plt.title("Transaction amount distribution")

plt.show()

# Bar chart showing fraud vs legitimate counts.

fraud_counts = transactions["fraud"].value_counts()
plt.bar(fraud_counts.index, fraud_counts.values)

plt.xlabel("Fraud")
plt.ylabel("Count")
plt.title("Fraud vs Legitimate Transactions")
plt.xticks([0, 1], ["Legitimate", "Fraud"])
plt.show()

# Box plots comparing transaction amount between fraud and legitimate transactions.

transactions.boxplot(column="amount", by="fraud")
plt.xlabel("Fraud")
plt.ylabel("Transaction amount (£)")
plt.title("Transaction amount by Fraud Status")
plt.suptitle("")  # Suppress the automatic title
plt.xticks([1, 2], ["Legitimate", "Fraud"])
plt.show()

# Scatter plot of customer age vs transaction amount.
plt.scatter(transactions["customer_age"], transactions["amount"])
plt.xlabel("Customer Age")
plt.ylabel("Transaction amount (£)")
plt.title("Customer Age vs Transaction Amount")
plt.show()

# correlation matrix for numerical columns.

print(transactions[
    ["amount", "customer_age", "international", "fraud"]
].corr())

#
# Observations based on the charts and correlation output:
# 1. The histogram and box plot show that most transaction amounts are fairly small, but there are a few very large values that create a strong right-skew.
# 2. The fraud-vs-legitimate bar chart shows that the dataset is heavily imbalanced, with far more legitimate transactions than fraudulent ones.
# 3. The box plot by fraud status suggests fraudulent transactions tend to have much higher amounts than legitimate transactions, with several clear outliers.
# 4. The scatter plot of age vs amount does not show a clear pattern, so customer age does not appear strongly related to transaction size in this sample.
# 5. The correlation matrix suggests fraud is more strongly related to amount and international transactions than to customer age.
