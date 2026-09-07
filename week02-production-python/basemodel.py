from pydantic import BaseModel, Field

class Transaction(BaseModel):
    transaction_id: int = Field(gt=0)
    amount: float = Field(gt=0)
    age: int = Field(ge=18, le=120)
    country: str
    international: bool