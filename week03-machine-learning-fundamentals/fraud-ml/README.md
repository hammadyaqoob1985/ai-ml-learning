# Fraud ML

A small fraud-detection project for the machine-learning fundamentals exercises. It generates synthetic transaction data, preprocesses numeric and categorical features, trains a balanced logistic-regression classifier, evaluates probability thresholds, and exposes predictions through a FastAPI endpoint.

## Features

- Synthetic transaction-data generation with reproducible random seeds.
- `ColumnTransformer` preprocessing:
  - `StandardScaler` for `amount` and `age`.
  - One-hot encoding for `country` and `transaction_type`.
  - Passthrough handling for the boolean `international` feature.
- A scikit-learn `Pipeline` containing preprocessing and logistic regression.
- Model persistence with Joblib, including the decision threshold.
- Pydantic request validation and FastAPI health and prediction endpoints.
- Evaluation metrics including confusion matrix, precision, recall, F1, and average precision.

## Project layout

```text
fraud-ml/
├── models/
│   └── fraud_model.joblib       # Saved model artifact used by the API
├── Dockerfile                   # Container image definition
├── src/fraud_ml/
│   ├── api.py                   # FastAPI application
│   ├── evaluate.py              # Classification metrics
│   ├── generate_data.py         # Synthetic data generator
│   ├── predict.py               # Save, load, and predict helpers
│   ├── preprocessing.py         # Feature preprocessing
│   ├── schemas.py               # API input and output models
│   └── train.py                 # Model pipeline definition
├── tests/                       # Unit and API tests
├── example.py                   # One-hot encoding example
└── preprocessing_example.py     # Preprocessing exercise
```

## Requirements

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/)

Install the project dependencies from the project directory:

```powershell
uv sync
```

The API and test modules also require FastAPI, Uvicorn, Joblib, Pytest, and HTTPX. If these are not already available in the environment, install them with:

```powershell
uv pip install fastapi uvicorn joblib pytest httpx
```

## Run the tests

```powershell
uv run pytest
```

## Train and save a model

The training module provides the model pipeline; the following example generates data, fits the model, and writes the artifact expected by the API:

```powershell
uv run python -c "from pathlib import Path; from fraud_ml.generate_data import generate_fraud_data; from fraud_ml.train import create_model; from fraud_ml.predict import save_model; df = generate_fraud_data(); X = df[['amount', 'age', 'international', 'transaction_type', 'country']]; y = df['fraud']; model = create_model(); model.fit(X, y); Path('models').mkdir(exist_ok=True); save_model(model, 0.67, 'models/fraud_model.joblib')"
```

The saved artifact contains both the fitted preprocessing/classification pipeline and the fraud threshold. Keeping them together ensures that inference uses the same transformations as training.

## Run the API

From the project directory, start the application with:

```powershell
uv run uvicorn fraud_ml.api:app --reload
```

The API loads `models/fraud_model.joblib` during startup.

## Run with Docker

Install and start [Docker Desktop](https://www.docker.com/products/docker-desktop/) or another Docker Engine, then run these commands from the project directory:

```powershell
docker build -t fraud-ml .
docker run --rm -p 8000:8000 fraud-ml
```

The image installs the project dependencies, copies the application source and saved model artifact, and starts Uvicorn on port `8000`. The API is then available at `http://127.0.0.1:8000`.

Check the running container:

```powershell
curl http://127.0.0.1:8000/health
```

To stop a foreground container, press `Ctrl+C`. To run it in the background instead, add `-d` to `docker run` and stop it with:

```powershell
docker stop <container-id>
```

### Health check

```powershell
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

### Fraud prediction

```powershell
curl -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"amount":1200.0,"age":35,"international":true,"transaction_type":"transfer","country":"FR"}'
```

Example response:

```json
{"fraud":true,"probability":0.9}
```

Requests must use a positive amount, an age from 18 to 120, a two-character country code, and one of `card`, `transfer`, or `cash` for `transaction_type`.

## Python usage

```python
from fraud_ml.generate_data import generate_fraud_data
from fraud_ml.train import create_model

df = generate_fraud_data(number_of_transactions=10_000, random_state=42)
features = ["amount", "age", "international", "transaction_type", "country"]

model = create_model()
model.fit(df[features], df["fraud"])
probabilities = model.predict_proba(df[features].head(5))[:, 1]
```

This project uses synthetic data for learning and demonstration. The model and API are not intended for production fraud decisions without real data validation, calibration, monitoring, and domain-specific review.
