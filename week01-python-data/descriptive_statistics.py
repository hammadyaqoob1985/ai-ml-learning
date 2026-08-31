import pandas as pd
import numpy as np

transactions = pd.DataFrame({
    "amount": [
        20, 25, 30, 32, 35,
        40, 42, 45, 50, 55,
        60, 65, 70, 75, 80,
        85, 90, 100, 120, 150,
        1000
    ]
})

#1. Calculate the mean transaction amount.
mean_amount = np.mean(transactions["amount"])
print(f"Mean transaction amount: {mean_amount}")

#2. Calculate the median transaction amount.
median_amount = np.median(transactions["amount"])
print(f"Median transaction amount: {median_amount}")

#3. Calculate the variance and standard deviation.
variance_amount = np.var(transactions["amount"])
std_dev_amount = np.std(transactions["amount"])
print(f"Variance of transaction amounts: {variance_amount}")
print(f"Standard deviation of transaction amounts: {std_dev_amount}")

# Use Pandas for these.
#
# 4. Calculate Q1 and Q3.
q1 = np.percentile(transactions["amount"], 25)
q3 = np.percentile(transactions["amount"], 75)
print(f"Q1 (25th percentile) of transaction amounts: {q1}")
print(f"Q3 (75th percentile) of transaction amounts: {q3}")

# 5. Calculate the IQR.
iqr = q3 - q1
print(f"IQR (Interquartile Range) of transaction amounts: {iqr}")

# 6. Identify outliers using the 1.5*IQR rule.
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# 7. Use those boundaries to return the potential outliers.
outliers = transactions[(transactions["amount"] < lower_bound) | (transactions["amount"] > upper_bound)]
print(f"Outliers in transaction amounts:\n{outliers}")

# A. Why is the mean higher than the median?
# The mean is higher than the median because the dataset contains a significant outlier (1000), which skews the mean upwards. The median, being the middle value, is less affected by extreme values and provides a better measure of central tendency for skewed distributions.

# B. The IQR method flags £1,000 as an outlier. Does that mean we should automatically delete that transaction? Why or why not?
# Not necessarily. While the IQR method identifies £1,000 as an outlier, it doesn't mean the transaction is invalid or should be deleted. Outliers should be investigated to understand their cause. They could represent legitimate high-value transactions, data entry errors, or fraudulent activity. The decision to remove or keep outliers should be based on the context and the specific analysis goals.