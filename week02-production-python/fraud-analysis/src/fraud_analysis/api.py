from fastapi import FastAPI

from fraud_analysis.models import TransactionRequest, PredictionResponse

app = FastAPI()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/transactions", response_model=PredictionResponse)
def create_transaction(transaction: TransactionRequest):
    return PredictionResponse(
        transaction_id=transaction.transaction_id,
        fraud=False,
        probability=0.15,
    )
