from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    transaction_id: int = Field(gt=0)
    amount: float = Field(gt=0)
    age: int = Field(ge=18, le=120)
    country: str
    international: bool

class PredictionResponse(BaseModel):
    transaction_id: int
    fraud: bool
    probability: float = Field(ge=0, le=1)