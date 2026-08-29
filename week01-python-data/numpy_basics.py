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
print()
