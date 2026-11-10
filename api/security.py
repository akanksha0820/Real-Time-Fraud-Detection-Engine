"""Authentication and distributed rate limiting for the HTTP boundary."""
from __future__ import annotations

import hashlib
import os
import secrets
import threading
import time
from collections import defaultdict

from fastapi import Header, HTTPException, Request, status


def _enabled(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() in {"1", "true", "yes", "on"}


def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
    """Require a constant-time API-key comparison unless tests explicitly disable auth."""
    if _enabled("FRAUD_AUTH_DISABLED"):
        return "auth-disabled"
    expected = os.getenv("FRAUD_API_KEY", "")
    if not expected:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="API authentication is not configured")
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or missing API key",
                            headers={"WWW-Authenticate": "ApiKey"})
    return x_api_key


class InMemoryFixedWindowLimiter:
    """Thread-safe local fallback; production Redis mode is shared across workers."""
    def __init__(self) -> None:
        self._windows: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
        self._lock = threading.Lock()

    def increment(self, key: str, window: int) -> int:
        with self._lock:
            previous_window, count = self._windows[key]
            count = count + 1 if previous_window == window else 1
            self._windows[key] = (window, count)
            return count


_memory_limiter = InMemoryFixedWindowLimiter()


def enforce_score_rate_limit(request: Request, api_key: str) -> None:
    """Apply a per-principal fixed-window limit to the scoring endpoint."""
    try:
        limit = max(1, int(os.getenv("FRAUD_RATE_LIMIT_PER_MINUTE", "120")))
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="Invalid rate-limit configuration") from exc

    principal = api_key if api_key != "auth-disabled" else (request.client.host if request.client else "test")
    principal_hash = hashlib.sha256(principal.encode()).hexdigest()[:24]
    window = int(time.time() // 60)
    key = f"fraud-risk:rate-limit:{principal_hash}:{window}"

    if os.getenv("FRAUD_STATE_BACKEND", "memory").lower() == "redis":
        try:
            import redis
            client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
            with client.pipeline(transaction=True) as pipeline:
                pipeline.incr(key)
                pipeline.expire(key, 120)
                count, _ = pipeline.execute()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Rate-limit service unavailable") from exc
    else:
        count = _memory_limiter.increment(principal_hash, window)

    if count > limit:
        raise HTTPException(status_code=429, detail="Scoring rate limit exceeded",
                            headers={"Retry-After": "60"})
