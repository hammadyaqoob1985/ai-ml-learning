import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from numpy import array

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


model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print("\nProbabilities:")
print(probabilities)

print("Predictions:")
print(predictions)

print("\nActual:")
print(y_test.to_numpy())

matrix = confusion_matrix(y_test, predictions)

print("Confusion matrix:", matrix)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# We know the model missed a genuinely fraudulent transaction. If a fraud model had 99% accuracy, could it nevertheless be a terrible fraud detector? Why?
# this could be a terrible fraud detector because accuracy is not a good metric for imbalanced datasets. In fraud detection, the number of fraudulent transactions is usually much smaller than the number of non-fraudulent transactions. Therefore, a model could achieve high accuracy by simply predicting all transactions as non-fraudulent, but it would fail to identify the actual fraudulent transactions, which is the primary goal of a fraud detection system.

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)

print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

# Model flags 10 transactions as fraud
#
# ↓
#
# Actually fraud       2 ✅
# Actually legitimate  8 ❌

# Would you expect precision to be high or low?

# And why?

# Low because the model is predicting more fraudulent transactions than actually fraudulent. Precision is a measure of the accuracy of its predictions. It tells us how many of the predicted fraudulent transactions are actually fraudulent.

# | Model | Precision | Recall |
# | ----- | --------: | -----: |
# | **A** |       95% |    40% |
# | **B** |       65% |    90% |

# Don't calculate anything.
#
# I want you to think like both an ML engineer and an engineering leader.
#
# Which model would you initially prefer for a banking fraud-detection system — A or B?
#
# Explain what the business trade-off is.

# I would choose B because it has a higher recall. This means it is better at identifying actual fraudulent transactions, which is crucial in a banking fraud-detection system. The trade-off is that it has lower precision, so it may flag more legitimate transactions as fraudulent, potentially causing inconvenience to customers. Which is a good tradeoff as I would prefer to catch more fraudulent transactions even if it means some legitimate transactions are flagged incorrectly. In a banking context, the cost of missing a fraudulent transaction is typically much higher than the cost of inconveniencing a customer with a false positive.

fraud_probabilities = np.array([
    0.91,
    0.72,
    0.55,
    0.48,
    0.31,
    0.08,
])

threshold_value = 0.3

fraud = fraud_probabilities > threshold_value
print(fraud)

# Why would lowering the threshold generally increase recall, but potentially decrease precision?

# Lowering the threshold generally increases recall, but potentially decreases precision. This is because the model is now focusing on identifying fraudulent transactions, which is a more important goal than identifying legitimate transactions. However, it may also lead to a decrease in precision, as the model is now flagging more transactions as fraudulent than actually fraudulent.