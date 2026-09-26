
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fraud_ml.predict import load_model, predict_transaction
from fraud_ml.schemas import TransactionRequest


MODEL_PATH = Path("models/fraud_model.joblib")


class PredictionResponse(BaseModel):
    fraud: bool
    probability: float = Field(ge=0, le=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.artifact = load_model(MODEL_PATH)
    yield


app = FastAPI(
    title="Fraud Detection API",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    try:
        return predict_transaction(
            app.state.artifact,
            transaction.model_dump(),
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        ) from exc

# Why do we load the model during application startup instead of loading it inside the /predict endpoint for every request?
# we load the model once during application startup and then reuse it for all requests because it is computationally expensive to load the model for each request.