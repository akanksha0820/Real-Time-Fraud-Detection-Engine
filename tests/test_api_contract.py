from fastapi.testclient import TestClient

import api.main as api_module


class FakeService:
    def score(self, event):
        return {"transaction_id": event["transaction_id"], "fraud_probability": .12,
                "decision": "ALLOW", "reasons": [], "feature_snapshot": {}}

    def metrics_snapshot(self):
        return {"total_requests": 1, "unique_transactions": 1, "idempotent_replays": 0,
                "decisions": {"ALLOW": 1}, "latency_ms": {"p50": 1, "p95": 2, "p99": 2}}


def test_score_and_metrics_contract(monkeypatch):
    monkeypatch.setenv("FRAUD_AUTH_DISABLED", "true")
    monkeypatch.setenv("FRAUD_STATE_BACKEND", "memory")
    fake = FakeService()
    monkeypatch.setattr(api_module, "service", lambda: fake)
    client = TestClient(api_module.app)
    event = {"transaction_id": "LIVE1", "event_timestamp": "2026-12-15T03:14:00",
             "customer_id": "C0000001", "account_id": "A00000001", "device_id": "D000000001",
             "merchant_id": "M000001", "amount_inr": 84500, "payment_channel": "online",
             "latitude": 19.076, "longitude": 72.8777}
    response = client.post("/score", json=event)
    assert response.status_code == 200
    assert response.json()["decision"] == "ALLOW"
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert metrics.json()["unique_transactions"] == 1
