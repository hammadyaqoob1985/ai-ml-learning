import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, PrecisionRecallDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
import pandas as pd
from numpy import array

from fraud_ml.evaluate import evaluate_model
from fraud_ml.generate_data import generate_fraud_data
from fraud_ml.preprocessing import create_preprocessor
from fraud_ml.train import create_model

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


df = pd.DataFrame({
    "amount": [
        50, 1200, 25, 7500, 300,
        9800, 75, 2200, 150, 6100,
        45, 3200
    ],
    "age": [
        45, 22, 37, 29, 61,
        24, 52, 33, 41, 27,
        58, 31
    ],
    "international": [
        False, True, False, True, False,
        True, False, True, False, True,
        False, True
    ],
    "transaction_type": [
        "card", "transfer", "card", "transfer",
        "cash", "transfer", "card", "card",
        "cash", "transfer", "card", "transfer"
    ],
    "country": [
        "UK", "FR", "UK", "DE",
        "UK", "ES", "UK", "FR",
        "UK", "DE", "UK", "ES"
    ],
    "fraud": [
        False, True, False, True,
        False, True, False, False,
        False, True, False, True
    ]
})
encoder = OneHotEncoder(sparse_output=False)

encoded = encoder.fit_transform(df[["country", "transaction_type"]])

print(encoded)

print(
    encoder.get_feature_names_out(
        ["country", "transaction_type"]
    )
)

X = df[["amount",
        "age",
        "international",
        "transaction_type",
        "country"]]
y = df["fraud"]

encoder = OneHotEncoder(sparse_output=False)

encoded = encoder.fit_transform(df[["country", "transaction_type"]])

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)

print("\nEncoded values:")
print(encoded)

print("\nEncoded feature names:")

print(
    encoder.get_feature_names_out(
        ["country", "transaction_type"]
    )
)
# 1. Why would converting UK → 1, FR → 2, DE → 3 be potentially misleading to an ML algorithm?
# this is potentially misleading because it implies an ordinal relationship between the countries, which does not exist. The algorithm may interpret higher numbers as having more importance or value, which is not the case for categorical data like country names.

# 2. What is the difference between fit() and transform() for the encoder?
# fit() is used to learning the preprocessing rules from the training data, while transform() is used to transform new data.

categorical_features = [
    "country",
    "transaction_type"
]

preprocessor = create_preprocessor()

X_transformed = preprocessor.fit_transform(X)

print(f"X_transformed:\n{X_transformed}")
print(f"X_transformed.shape:\n{X_transformed.shape}")

print(f"Feature names:\n{preprocessor.get_feature_names_out()}")

# We currently called fit_transform(X) before doing our train/test split. Can you think of why that might be a bad idea in a real ML project?
# This is a bad idea because test and tranin data should be kept separate to avoid data leakage. If we fit the encoder on the entire dataset before splitting, information from the test set could influence the training process, leading to overly optimistic performance metrics and a model that may not generalize well to unseen data.

X = df[["amount",
        "age",
        "international",
        "transaction_type",
        "country"]]
y = df["fraud"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

X_train_transformed = preprocessor.fit_transform(X_train)

X_test_transformed = preprocessor.transform(X_test)

print("Original:")
print("X train:", X_train.shape)
print("X test:", X_test.shape)

print("\nTransformed:")
print("X train:", X_train_transformed.shape)
print("X test:", X_test_transformed.shape)

print("\nFeatures:")
print(preprocessor.get_feature_names_out())

# 1. Why do we use fit_transform() on X_train, but only transform() on X_test?
# fit_transform() is used to learn the preprocessing rules from the training data, while transform() is used to transform new data.

# 2. In your own words, what is data leakage?
# Data leakage occurs when information from a test set is used to train a model. This can lead to overfitting or even incorrect predictions. Data leakage occurs when information that shouldn't be available during training influences the model or its training process. overly optimistic or otherwise unreliable estimate of real-world performance.

X = df[["amount",
        "age",
        "international",
        "transaction_type",
        "country"]]
y = df["fraud"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

categorical_features = [
    "country",
    "transaction_type"
]

numerical_features = [
    "amount",
    "age"
]

preprocessor = create_preprocessor()

X_train_transformed = preprocessor.fit_transform(X_train)

X_test_transformed = preprocessor.transform(X_test)

print("Features:")
print(preprocessor.get_feature_names_out())

print("\nTransformed training data:")
print(X_train_transformed)

print("\nTransformed training data shape:")
print(X_train_transformed.shape)

# Why do you think we one-hot encode country but standardise amount instead of applying the same preprocessing technique to both?
# one-hot encoding country because it is a categorical feature, while standardising amount because it is a numerical feature.

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression())
    ]
)

X = df[["amount",
        "age",
        "international",
        "transaction_type",
        "country"]]
y = df["fraud"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]

print("Predictions:")
print(predictions)

print("\nActual:")
print(y_test.to_numpy())

print("\nFraud probabilities:")
print(probabilities)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

print("\nAccuracy:", accuracy_score(y_test, predictions))
print(
    "Precision:",
    precision_score(y_test, predictions, zero_division=0)
)
print("Recall:", recall_score(y_test, predictions))
print("F1:", f1_score(y_test, predictions))

# 1. What is the advantage of putting the ColumnTransformer and LogisticRegression into a Pipeline rather than manually transforming the data?
# The advantage of putting the ColumnTransformer and LogisticRegression into a Pipeline is that it allows us to easily apply the same preprocessing techniques to both the training and testing data. This is particularly useful when we have a large dataset and want to apply the same preprocessing techniques to all features. Their biggest benefit is consistency and safety

# 2. When we call:
#
# model.predict(X_test)
#
# does the StandardScaler learn a new mean and standard deviation from X_test, or does it use what it learned from X_train? Why?
# It uses what it learned from X_train. This is because the StandardScaler is part of the pipeline, and during training (fit), it calculates the mean and standard deviation from the training data. When making predictions, it applies the same transformation to the test data using the statistics learned from the training data to ensure consistency and avoid data leakage.

df = generate_fraud_data()

print(df.head())
print()
print(df.shape)

print("\nFraud counts:")
print(df["fraud"].value_counts())

print("\nFraud rate:")
print(df["fraud"].mean())

print("\nFraud rate by international:")
print(
    df.groupby("international")["fraud"].mean()
)

print("\nFraud rate by transaction type:")
print(
    df.groupby("transaction_type")["fraud"].mean()
)

print("\nFraud rate by country:")
print(
    df.groupby("country")["fraud"].mean()
)

# Based purely on this dataset, which features appear to have a relationship with fraud, and which appear less useful?
# Expected signal
#
# international       → strong
# transaction_type    → useful
# amount              → useful
# country             → little/none
# age                 → little/none

categorical_features = [
    "country",
    "transaction_type",
]

numerical_features = [
    "amount",
    "age",
]

preprocessor = create_preprocessor()

X = df[
    [
        "amount",
        "age",
        "international",
        "transaction_type",
        "country",
    ]
]

y = df["fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression()),
    ]
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

print(
    "Accuracy:",
    accuracy_score(y_test, predictions)
)

print(
    "Precision:",
    precision_score(
        y_test,
        predictions,
        zero_division=0,
    )
)

print(
    "Recall:",
    recall_score(y_test, predictions),
)

print(
    "F1:",
    f1_score(y_test, predictions),
)

model_balanced = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced"
            ),
        ),
    ]
)

model_balanced.fit(X_train, y_train)

balanced_predictions = model_balanced.predict(X_test)

print("Confusion matrix:")
print(
    confusion_matrix(
        y_test,
        balanced_predictions
    )
)

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        balanced_predictions
    )
)

print(
    "Precision:",
    precision_score(
        y_test,
        balanced_predictions,
        zero_division=0,
    )
)

print(
    "Recall:",
    recall_score(
        y_test,
        balanced_predictions
    )
)

print(
    "F1:",
    f1_score(
        y_test,
        balanced_predictions
    )
)

fraud_probabilities = model_balanced.predict_proba(X_test)[:, 1]

thresholds = [0.3, 0.5, 0.7]

for threshold in thresholds:
    predictions = fraud_probabilities >= threshold

    print(f"\nThreshold: {threshold}")

    print(
        "Precision:",
        precision_score(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print(
        "Recall:",
        recall_score(
            y_test,
            predictions
        )
    )

    print(
        "F1:",
        f1_score(
            y_test,
            predictions
        )
    )

    print(
        "Confusion matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

precision_values, recall_values, thresholds = precision_recall_curve(
        y_test,
        fraud_probabilities
)

print("Number of thresholds:", len(thresholds))
print("Number of precision values:", len(precision_values))
print("Number of recall values:", len(recall_values))
f1_values = (
        2
        * precision_values[:-1]
        * recall_values[:-1]
        / (
                precision_values[:-1]
                + recall_values[:-1]
                + 1e-10
        )
)

best_index = np.argmax(f1_values)

print("Best threshold:", thresholds[best_index])
print("Precision:", precision_values[best_index])
print("Recall:", recall_values[best_index])
print("F1:", f1_values[best_index])

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)

print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_validation.shape, y_validation.shape)
print("Test:", X_test.shape, y_test.shape)

print("\nFraud rates:")
print("Training:", y_train.mean())
print("Validation:", y_validation.mean())
print("Test:", y_test.mean())

model_balanced.fit(X_train, y_train)

feature_names = (
    model_balanced
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

coefficients = (
    model_balanced
    .named_steps["classifier"]
    .coef_[0]
)

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients,
})

feature_importance["absolute_coefficient"] = (
    feature_importance["coefficient"].abs()
)

feature_importance = feature_importance.sort_values(
    "absolute_coefficient",
    ascending=False,
)

print("\nLOGISTIC REGRESSION COEFFICIENTS")
print(feature_importance.to_string(index=False))

validation_probabilities = model_balanced.predict_proba(
    X_validation
)[:, 1]

precision_values, recall_values, thresholds = precision_recall_curve(
    y_validation,
    validation_probabilities,
)

f1_values = (
        2
        * precision_values[:-1]
        * recall_values[:-1]
        / (
                precision_values[:-1]
                + recall_values[:-1]
                + 1e-10
        )
)

best_index = np.argmax(f1_values)

best_threshold = thresholds[best_index]

print("Best validation threshold:", best_threshold)
print("Validation precision:", precision_values[best_index])
print("Validation recall:", recall_values[best_index])
print("Validation F1:", f1_values[best_index])

test_probabilities = model_balanced.predict_proba(
    X_test
)[:, 1]

test_predictions = (
        test_probabilities >= best_threshold
)

print("\nFINAL TEST RESULTS")

print(
    "Confusion matrix:"
)
print(
    confusion_matrix(
        y_test,
        test_predictions
    )
)

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        test_predictions
    )
)

print(
    "Precision:",
    precision_score(
        y_test,
        test_predictions,
        zero_division=0,
    )
)

print(
    "Recall:",
    recall_score(
        y_test,
        test_predictions
    )
)

print(
    "F1:",
    f1_score(
        y_test,
        test_predictions
    )
)

dummy_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            DummyClassifier(
                strategy="most_frequent"
            )
        ),
    ]
)

dummy_model.fit(X_train, y_train)

dummy_predictions = dummy_model.predict(X_test)

print("\nDUMMY MODEL")

print("Confusion matrix:")
print(
    confusion_matrix(
        y_test,
        dummy_predictions
    )
)

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        dummy_predictions
    )
)

print(
    "Precision:",
    precision_score(
        y_test,
        dummy_predictions,
        zero_division=0,
    )
)

print(
    "Recall:",
    recall_score(
        y_test,
        dummy_predictions
    )
)

print(
    "F1:",
    f1_score(
        y_test,
        dummy_predictions
    )
)

tree_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)

tree_model.fit(X_train, y_train)

tree_predictions = tree_model.predict(X_validation)

print("\nDECISION TREE - VALIDATION")

print("Confusion matrix:")
print(
    confusion_matrix(
        y_validation,
        tree_predictions
    )
)

print(
    "Accuracy:",
    accuracy_score(
        y_validation,
        tree_predictions
    )
)

print(
    "Precision:",
    precision_score(
        y_validation,
        tree_predictions,
        zero_division=0,
    )
)

print(
    "Recall:",
    recall_score(
        y_validation,
        tree_predictions
    )
)

print(
    "F1:",
    f1_score(
        y_validation,
        tree_predictions
    )
)

tree_train_predictions = tree_model.predict(X_train)

print("\nDECISION TREE - TRAINING")

print(
    "Accuracy:",
    accuracy_score(
        y_train,
        tree_train_predictions
    )
)

print(
    "Precision:",
    precision_score(
        y_train,
        tree_train_predictions,
        zero_division=0,
    )
)

print(
    "Recall:",
    recall_score(
        y_train,
        tree_train_predictions
    )
)

print(
    "F1:",
    f1_score(
        y_train,
        tree_train_predictions
    )
)

for depth in [2, 3, 5, 10, None]:
    tree_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=depth,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    tree_model.fit(X_train, y_train)

    train_predictions = tree_model.predict(X_train)
    validation_predictions = tree_model.predict(X_validation)

    print(f"\nMAX DEPTH: {depth}")

    print(
        "Training F1:",
        f1_score(y_train, train_predictions)
    )

    print(
        "Validation F1:",
        f1_score(
            y_validation,
            validation_predictions
        )
    )

    print(
        "Validation precision:",
        precision_score(
            y_validation,
            validation_predictions,
            zero_division=0,
        )
    )

    print(
        "Validation recall:",
        recall_score(
            y_validation,
            validation_predictions
        )
    )

forest_model = Pipeline(
    steps=[
        ("preprocessor", create_preprocessor()),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)

forest_model.fit(X_train, y_train)

forest_predictions = forest_model.predict(
    X_validation
)

print("\nRANDOM FOREST - VALIDATION")

print(
    "Confusion matrix:"
)
print(
    confusion_matrix(
        y_validation,
        forest_predictions
    )
)

print(
    "Accuracy:",
    accuracy_score(
        y_validation,
        forest_predictions
    )
)

print(
    "Precision:",
    precision_score(
        y_validation,
        forest_predictions,
        zero_division=0,
    )
)

print(
    "Recall:",
    recall_score(
        y_validation,
        forest_predictions
    )
)

print(
    "F1:",
    f1_score(
        y_validation,
        forest_predictions
    )
)

forest_probabilities = forest_model.predict_proba(
    X_validation
)[:, 1]

for threshold in [0.1, 0.2, 0.3, 0.5]:
    predictions = forest_probabilities >= threshold

    print(f"\nThreshold: {threshold}")

    print(
        "Precision:",
        precision_score(
            y_validation,
            predictions,
            zero_division=0
        )
    )

    print(
        "Recall:",
        recall_score(
            y_validation,
            predictions,
            zero_division=0
        )
    )

    print(
        "F1:",
        f1_score(
            y_validation,
            predictions,
            zero_division=0
        )
    )

logistic_probabilities = model_balanced.predict_proba(
    X_validation
)[:, 1]

forest_probabilities = forest_model.predict_proba(
    X_validation
)[:, 1]

print(
    "Logistic Regression AP:",
    average_precision_score(
        y_validation,
        logistic_probabilities
    )
)

print(
    "Random Forest AP:",
    average_precision_score(
        y_validation,
        forest_probabilities
    )
)

PrecisionRecallDisplay.from_predictions(
    y_validation,
    logistic_probabilities,
    name="Logistic Regression"
)

PrecisionRecallDisplay.from_predictions(
    y_validation,
    forest_probabilities,
    name="Random Forest",
    ax=plt.gca()
)

plt.title("Precision-Recall Curves — Validation")
plt.show()

X_development = pd.concat(
    [X_train, X_validation]
)
y_development = pd.concat(
    [y_train, y_validation]
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

logistic_cv_scores = cross_val_score(
    model_balanced,
    X_development,
    y_development,
    cv=cv,
    scoring="average_precision"
)

forest_cv_scores = cross_val_score(
    forest_model,
    X_development,
    y_development,
    cv=cv,
    scoring="average_precision"
)

print("Logistic Regression AP:", logistic_cv_scores)
print("Logistic Regression mean:", logistic_cv_scores.mean())
print("Logistic Regression std:", logistic_cv_scores.std())

print("Random Forest AP:", forest_cv_scores)
print("Random Forest mean:", forest_cv_scores.mean())
print("Random Forest std:", forest_cv_scores.std())


X_development = pd.concat(
    [X_train, X_validation]
)

y_development = pd.concat(
    [y_train, y_validation]
)

final_model = create_model()

final_model.fit(
    X_development,
    y_development
)

final_threshold = 0.6717561057667615

results = evaluate_model(
    final_model,
    X_test,
    y_test,
    threshold=final_threshold,
)

print("\nFINAL MODEL EVALUATION")
for metric, value in results.items():
    print(f"{metric}: {value}")