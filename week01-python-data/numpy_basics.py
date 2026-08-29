import numpy as np

transactions = np.array([
    [101, 50.00],
    [102, 125.50],
    [103, 20.00],
    [104, 500.00],
    [105, 75.25],
    [106, 1000.00]
])

## 1. Print the shape of the dataset

print(transactions.shape)

# 2. Extract all transaction amounts

print(transactions[:, 1])

# 3. Calculate the mean transaction amount

print(transactions[:, 1].mean())

# 4. Return only amounts greater than £100

print(transactions[transactions[:, 1] > 100])

# 4. Return only amounts greater than £100 chat gpt solution

amounts = transactions[:, 1]

print(amounts[amounts > 100])

# 5. Increase every amount by 10% without a loop
transactions[:, 1] += transactions[:, 1] * 0.1
print(transactions)

# 5. Chatgpt solution
transactions[:, 1] *= 1.1
print(transactions)


# 2nd exercise

customers = np.array([
    [25, 30000],
    [40, 60000],
    [35, 45000],
    [50, 80000]
], dtype=float)

# 1. Shape
print(customers.shape)

# 2. Mean of each feature
means = customers.mean(axis=0)
print(means)

# 3. Centre each feature
centered_customers = customers - means
print(centered_customers)

# 4. Standard deviation of each feature
stds = customers.std(axis=0)
print(stds)

# 5. Standardise features
standardised_customers = (customers - means) / stds
print(standardised_customers)

# Bonus: matrix multiplication
weights = np.array([0.3, 0.7])
predictions = customers @ weights
print(predictions)