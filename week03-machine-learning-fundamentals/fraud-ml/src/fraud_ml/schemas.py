from typing import Literal

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    amount: float = Field(gt=0)
    age: int = Field(ge=18, le=120)
    international: bool
    transaction_type: Literal["card", "transfer", "cash"]
    country: str = Field(min_length=2, max_length=2)