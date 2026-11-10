from fastapi.testclient import TestClient

import api.main as api_module
from api.security import InMemoryFixedWindowLimiter


class FakeService:
    state = object()

    def score(self, event):
        return {"transaction_id": event["transaction_id"], "decision": "ALLOW"}


EVENT = {
    "transaction_id": "SECURITY1", "event_timestamp": "2026-12-15T03:14:00Z",
    "customer_id": "C1", "account_id": "A1", "device_id": "D1", "merchant_id": "M1",
    "amount_inr": 100, "payment_channel": "online", "latitude": 19.076, "longitude": 72.8777,
}


def test_protected_endpoint_requires_api_key(monkeypatch):
    monkeypatch.setenv("FRAUD_AUTH_DISABLED", "false")
    monkeypatch.setenv("FRAUD_API_KEY", "test-secret")
    monkeypatch.setenv("FRAUD_STATE_BACKEND", "memory")
    monkeypatch.setattr(api_module, "service", lambda: FakeService())
    client = TestClient(api_module.app)
    assert client.post("/score", json=EVENT).status_code == 401
    response = client.post("/score", json=EVENT, headers={"X-API-Key": "test-secret"})
    assert response.status_code == 200


def test_request_id_is_returned(monkeypatch):
    monkeypatch.setenv("FRAUD_AUTH_DISABLED", "true")
    client = TestClient(api_module.app)
    response = client.get("/live", headers={"X-Request-ID": "trace-123"})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "trace-123"


def test_fixed_window_limiter_resets_between_windows():
    limiter = InMemoryFixedWindowLimiter()
    assert limiter.increment("principal", 10) == 1
    assert limiter.increment("principal", 10) == 2
    assert limiter.increment("principal", 11) == 1
