import pandas as pd

customers = pd.DataFrame({
    "customer_id": [101, 102, 103, 104, 105, 106],
    "age": [25, 42, 31, 55, 28, 46],
    "salary": [30000, 72000, 48000, 90000, 35000, 68000],
    "country": ["UK", "UK", "France", "UK", "Germany", "France"],
    "active": [True, True, False, True, False, True]
})

print(customers)

# 1. Dataset inspection
#
# Print:
#
# shape
# first 3 rows
# data types
# descriptive statistics

print(customers.shape)

print(customers.head(3))

print(customers.dtypes)

print(customers.describe())

# 2. Select only:
# age
# salary

print(customers[["age", "salary"]])

# 3. Find all customers from France. Only return the customer_id column.

print(customers[customers["country"] == "France"])

# 4. Find customers earning at least £60,000.

print(customers[customers["salary"] >= 60000])

# 5. Find customers who are:
# age > 40
# AND
# active == True

print(customers[(customers["age"] > 40) & (customers["active"] == True)])

# 6. Return only customer_id and salary for customers earning over £50,000.
#
# I'd like you to use:
#
# .loc[]

print(customers.loc[(customers["salary"] > 50000), ["customer_id", "salary"]])

# 7. Create a new column
# salary_after_raise
#
# representing a 5% salary increase.

customers["salary_after_raise"] = customers["salary"] * 1.05
print(customers)

# Bonus
#
# Find customers who are:
#
# country = UK
# OR
# salary > £70,000

print(customers[(customers["country"] == "UK") | (customers["salary"] > 70000)])
