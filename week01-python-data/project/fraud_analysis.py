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

######## PART C ################

# What percentage of transactions are fraudulent?
fraud_percentage = (clean_data["fraud"].sum() / clean_data.shape[0]) * 100
print(f"Percentage of fraudulent transactions: {fraud_percentage:.2f}%")

# What is the fraud rate by transaction_type?
fraud_rate_by_type = clean_data.groupby("transaction_type")["fraud"].mean() * 100
print("\nFraud rate by transaction type:")
print(fraud_rate_by_type)

# What is the fraud rate by country?
fraud_rate_by_country = clean_data.groupby("country")["fraud"].mean() * 100
print("\nFraud rate by country:")
print(fraud_rate_by_country)

# What is the fraud rate for international vs domestic transactions?
fraud_rate_by_international = clean_data.groupby("international")["fraud"].mean() * 100
print("\nFraud rate by international vs domestic transactions:")
print(fraud_rate_by_international)

# Compare fraudulent vs legitimate transactions by:
#
# mean amount
# median amount
# mean customer age
# median customer age

comparison = (
    clean_data
    .groupby("fraud")
    .agg(
        mean_amount=("amount", "mean"),
        median_amount=("amount", "median"),
        mean_customer_age=("customer_age", "mean"),
        median_customer_age=("customer_age", "median"),
    )
    .rename(index={0: "Legitimate", 1: "Fraudulent"})
)

print(comparison)

# Does transaction amount appear associated with fraud?

# Fraudulent transactions have a higher mean transaction amount than
# legitimate transactions, while the difference between their medians
# is smaller.
#
# This suggests that some high-value fraudulent transactions may be
# pulling the fraudulent mean upward.
#
# Transaction amount therefore appears potentially associated with
# fraud, but further analysis of the distribution is needed.

# Do international transactions appear associated with fraud?

# Yes—international transactions appear associated with higher fraud risk.
# Your fraud rate is about 15.7% for international vs 3.49% for domestic, which is roughly 4.5× higher for international transactions. That indicates a clear association in this dataset.

# Regarding transaction type
# there appears to be an association between transaction type and fraud in your data.
# transfer has the highest fraud rate (~7.99%), compared with card (~4.05%) and cash (~3.16%), so fraud risk varies by type. This indicates relationship/association (categorical), not a numeric correlation coefficient.

######## PART  D ################

# 1. Overall transaction amount statistics
#
# mean
# median
# standard deviation
# minimum
# Q1
# Q3
# maximum

amount_stats = clean_data["amount"].describe(percentiles=[0.25, 0.5, 0.75]).loc[
    ["mean", "50%", "std", "min", "25%", "75%", "max"]
]
amount_stats = amount_stats.rename({"50%": "median", "25%": "Q1", "75%": "Q3"})
print(amount_stats)

q1 = np.percentile(clean_data["amount"], 25)
q3 = np.percentile(clean_data["amount"], 75)
print(f"Q1 (25th percentile) of transaction amounts: {q1}")
print(f"Q3 (75th percentile) of transaction amounts: {q3}")

iqr = q3 - q1
print(f"IQR (Interquartile Range) of transaction amounts: {iqr}")

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = clean_data[(clean_data["amount"] < lower_bound) | (clean_data["amount"] > upper_bound)]
print(f"Outliers in transaction amounts:\n{outliers}")

# How many potential outliers are there?
num_outliers = outliers.shape[0]
print(f"Number of potential outliers: {num_outliers}")

# What percentage of all transactions are potential outliers?
percentage_outliers = (num_outliers / clean_data.shape[0]) * 100
print(f"Percentage of potential outliers: {percentage_outliers:.2f}%")

# Calculate the fraud rate separately for:

# IQR outliers
fraud_rate_outliers = outliers["fraud"].mean() * 100
print(f"Fraud rate for IQR outliers: {fraud_rate_outliers:.2f}%")

# non-outliers
non_outliers = clean_data[(clean_data["amount"] >= lower_bound) & (clean_data["amount"] <= upper_bound)]
fraud_rate_non_outliers = non_outliers["fraud"].mean() * 100
print(f"Fraud rate for non-outliers: {fraud_rate_non_outliers:.2f}%")

# I'd like something conceptually like:
#
#                       Transactions    Fraud rate
# IQR outliers          ???             ???%
# Non-outliers          ???             ???%

iqr_summary = (
    clean_data.assign(iqr_group=np.where(clean_data.index.isin(outliers.index), "IQR outliers", "Non-outliers"))
    .groupby("iqr_group")
    .agg(
        Transactions=("fraud", "size"),
        Fraud_rate=("fraud", "mean"),  # assumes 1=fraud, 0=not fraud
    )
)

iqr_summary["Fraud_rate"] = (iqr_summary["Fraud_rate"] * 100).round(2).astype(str) + "%"

print(iqr_summary)

clean_data["amount_outlier"] = (
        (clean_data["amount"] < lower_bound) |
        (clean_data["amount"] > upper_bound)
)

outlier_summary = (
    clean_data
    .groupby("amount_outlier")
    .agg(
        transactions=("fraud", "size"),
        fraud_rate=("fraud", "mean")
    )
)

outlier_summary["fraud_rate"] *= 100
print(f"Outlier summary:\n{outlier_summary}")

# A. Is the amount distribution skewed? What evidence supports your answer?
#
# Yes. The transaction amount distribution appears to be right-skewed.
# The mean is higher than the median, the maximum is substantially
# higher than Q3, and there are a number of unusually high-value
# transactions above the upper IQR boundary.
#
# This suggests that most transactions are relatively small, with
# a smaller number of very large transactions creating a long right tail.

# B. Why are the mean and median different?
#
# The mean is higher than the median because the amount distribution
# is right-skewed. A relatively small number of high-value transactions
# pull the mean upwards, while the median is much less affected by
# those extreme observations.
# C. Are high-value outliers more likely to be fraudulent in this dataset?

high_value_outliers = clean_data[clean_data["amount"] > upper_bound]
fraud_rate_high_value_outliers = high_value_outliers["fraud"].mean() * 100
print(f"Fraud rate for high-value outliers: {fraud_rate_high_value_outliers:.2f}%")
# I would say that high-value outliers are more likely to be fraudulent in this dataset.

# D. Should the outliers be removed before modelling?
#
# Not automatically. High-value transactions have a higher observed
# fraud rate in this dataset, so removing them could discard useful
# predictive information.
#
# These observations should first be investigated to determine whether
# they are genuine transactions or data-quality errors. If legitimate,
# they may be particularly valuable for training a fraud-detection model.



######## PART  E ################

# Histogram of transaction amounts.
plt.hist(clean_data["amount"], bins=10)

plt.xlabel("Transaction amount (£)")
plt.ylabel("Frequency")
plt.title("Distribution of transaction amounts")
plt.show()

# Box plot of transaction amounts.

plt.boxplot(clean_data["amount"])

plt.ylabel("Transaction amount (£)")
plt.title("Transaction amount distribution")

plt.show()

# Bar chart showing fraud vs legitimate counts.

fraud_counts = clean_data["fraud"].value_counts()
plt.bar(fraud_counts.index, fraud_counts.values)

plt.xlabel("Fraud")
plt.ylabel("Count")
plt.title("Fraud vs Legitimate Transactions")
plt.xticks([0, 1], ["Legitimate", "Fraud"])
plt.show()

# Box plots comparing transaction amount between fraud and legitimate transactions.

clean_data.boxplot(column="amount", by="fraud")
plt.xlabel("Fraud")
plt.ylabel("Transaction amount (£)")
plt.title("Transaction amount by Fraud Status")
plt.suptitle("")  # Suppress the automatic title
plt.xticks([1, 2], ["Legitimate", "Fraud"])
plt.show()

# Bar chart showing fraud by transaction type.

fraud_rate_by_type = (
        clean_data
        .groupby("transaction_type")["fraud"]
        .mean() * 100
)

fraud_rate_by_type.plot(kind="bar")

plt.title("Fraud Rate by Transaction Type")
plt.xlabel("Transaction Type")
plt.ylabel("Fraud Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Bar chart showing fraud by Domestic/International.
fraud_rate_location = (
        clean_data
        .groupby("international")["fraud"]
        .mean() * 100
)

fraud_rate_location.index = fraud_rate_location.index.map({
    False: "Domestic",
    True: "International"
})

fraud_rate_location.plot(kind="bar")

plt.title("Fraud Rate: Domestic vs International")
plt.xlabel("Transaction Location")
plt.ylabel("Fraud Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Amount distribution: Strongly right-skewed—most transactions are low-value, with a long tail of higher amounts.
# Outliers: Yes, clear outliers are present (including very large values around £2k–£4k+).
# Fraud imbalance: Yes, fraud is highly imbalanced (far fewer fraud cases than legitimate).
# Amounts by fraud status: Fraud transactions tend to have higher central values (higher typical amounts) than legitimate ones, though both groups are right-skewed and overlap with outliers.
#
# How does fraud differ by transaction type?: Fraud differs clearly by type: transfer and card have many more fraud cases than cash, and cash has the fewest fraud incidents.
# From your earlier rates, transfer also had the highest fraud rate (~7.99%), followed by card (~4.05%), then cash (~3.16%), so transfers look riskiest both in count and proportion.

# How does fraud differ for international/domestic transactions? based on attached graph
# pasted_image_1_1788625680024.png
# Based on this graph alone: domestic has more fraud cases in absolute count, but it also has far more total transactions.
# So the chart suggests fraud exists in both groups, but to compare risk fairly you should use fraud rate (fraud / total per group). From your earlier rates, international has the higher fraud risk.


####### Part F — Correlation #################

corr = clean_data[["amount", "customer_age", "international", "fraud"]].corr()
print(f"correlation is {corr}")

# 1.
# Strongest positive correlation with fraud: international (0.191068) is the strongest among listed variables.
# 2.
# Correlation between customer_age and fraud: about -0.000201 (essentially zero).
# 3.
# Does this support earlier findings? Yes. It aligns with your earlier result that international transactions are associated with higher fraud risk.
# 4.
# Does positive correlation imply causation? No. Correlation shows association, not cause-and-effect.
# 5.
# Why transaction_type isn’t naturally included: Pearson correlation needs numeric variables. transaction_type is categorical text, so it must be encoded (e.g., one-hot/dummies) before numeric correlation methods can use it.

######## CONCLUSION ########
# The cleaned dataset appears usable for fraud analysis: key fields were prepared and
# transformed into analysis-ready form (including numeric/boolean variables needed for plots
# and correlation checks). dataset was cleaned by removing duplicates, imputing missing values and standardising categorical values
#
# Fraud is clearly class-imbalanced (far fewer fraud cases than legitimate transactions),
# which means raw accuracy would be misleading for modelling. Model evaluation should
# prioritize recall/precision, F1, PR-AUC, and class-sensitive approaches.
#
# Transaction amount is strongly right-skewed, with many low-value transactions and a long
# tail of high-value values. There are clear outliers (including very large amounts), and
# outlier transactions show a higher fraud rate than non-outliers, so amount-related signals
# are informative for risk.
#
# International transactions show a much higher fraud rate than domestic transactions
# (earlier analysis: ~15.7% vs ~3.5%), and correlation with fraud is positive
# (international vs fraud ≈ 0.191), consistent with that pattern.
#
# Fraud also differs by transaction type: transfer has the highest fraud rate, followed by
# card, then cash. This indicates transaction_type contains meaningful predictive information.
#
# Customer age shows near-zero linear relationship with fraud (corr ≈ -0.0002), suggesting
# age alone is a weak standalone predictor in this dataset.
#
# Modelling implications:
# - handle imbalance (class weights, resampling, threshold tuning),
# - use robust/engineered amount features (e.g., log(amount), outlier flags),
# - include international and transaction_type features (proper categorical encoding),
# - use metrics suited to rare-event detection rather than accuracy alone.
#
# Overall, the strongest fraud signals in this analysis are international status, transaction
# type, and amount behavior (especially high-value/outlier patterns), while age contributes
# little on its own.