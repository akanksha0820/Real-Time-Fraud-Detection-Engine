import pytest


@pytest.fixture(autouse=True)
def safe_test_environment(monkeypatch):
    """Unit tests opt out explicitly; deployed environments remain fail-closed."""
    monkeypatch.setenv("FRAUD_AUTH_DISABLED", "true")
    monkeypatch.setenv("FRAUD_STATE_BACKEND", "memory")
