import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.DataFrame(
    {
        "amount": [50, 1200, 25, 7500, 300, 9800],
        "age": [45, 22, 37, 29, 61, 24],
        "international": [False, True, False, True, False, True],
        "fraud": [False, True, False, True, False, True],
    }
)

X = df[["amount", "age", "international"]]
y = df["fraud"]

print(X)
print(y)

# Is this supervised or unsupervised learning?
# this is supervised learning

# Is it classification or regression?
# this is classification

# What does X represent?
# this is a matrix of features

# What does y represent?
# this is a vector of labels

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
    stratify=y
)

print("X train:")
print(X_train)

print("\nX test:")
print(X_test)

print("\ny train:")
print(y_train)

print("\ny test:")
print(y_test)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# 1. What do you think test_size=0.33 means?
# test_size=0.33 means that 33% of the data will be used for testing.

# Why do you think I've specified random_state=42?
# random_state=42 is specified to ensure reproducibility of the results.


print("Full dataset:")
print(y.value_counts())

print("\nTraining:")
print(y_train.value_counts())

print("\nTesting:")
print(y_test.value_counts())

# Why shouldn't we evaluate a model using the same data it was trained on?

# we are evaluating the model on the test data, not the training data.When we evaluate a model, we want to see how well it performs on new data that it has not seen before.

# What does random_state give us? (You've essentially already answered this.)

# random_state=42 is used to ensure reproducibility of the results.

# Why is stratify=y particularly useful for an imbalanced problem such as fraud detection?

# stratify=y helps preserve the proportion of fraud and non-fraud examples in both the training and testing datasets