import pytest
from pydantic import ValidationError
from python.streaming.contracts import TransactionEnvelope

def payload():
    return {"transaction_id":"T1","event_timestamp":"2026-01-01T00:00:00Z",
            "customer_id":"C1","account_id":"A1","device_id":"D1","merchant_id":"M1",
            "amount_inr":100,"payment_channel":"online","latitude":10,"longitude":20}

def test_transaction_envelope_round_trip():
    envelope = TransactionEnvelope.create(payload())
    restored = TransactionEnvelope.model_validate_json(envelope.model_dump_json())
    assert restored.schema_version == "1.0"
    assert restored.payload.transaction_id == "T1"

def test_transaction_envelope_rejects_invalid_amount():
    event = payload(); event["amount_inr"] = -1
    with pytest.raises(ValidationError): TransactionEnvelope.create(event)
