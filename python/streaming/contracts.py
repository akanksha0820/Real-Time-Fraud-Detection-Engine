"""Versioned Kafka contracts shared by producers and consumers."""
from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class TransactionPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    transaction_id: str = Field(min_length=1, max_length=100)
    event_timestamp: datetime
    customer_id: str
    account_id: str
    device_id: str
    merchant_id: str
    amount_inr: float = Field(gt=0, le=2_000_000)
    payment_channel: Literal["online", "pos"]
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

class TransactionEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = "1.0"
    event_type: Literal["transaction.created"] = "transaction.created"
    produced_at_utc: datetime
    payload: TransactionPayload

    @classmethod
    def create(cls, payload: dict):
        return cls(produced_at_utc=datetime.now(timezone.utc), payload=payload)
