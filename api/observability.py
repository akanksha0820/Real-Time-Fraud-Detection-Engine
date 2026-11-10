"""PII-safe JSON request logging and correlation IDs."""
from __future__ import annotations

import json
import logging
import os
import time
from uuid import uuid4

from fastapi import Request

logger = logging.getLogger("fraud_api.access")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
logger.propagate = False


async def request_observability(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", "").strip()[:128] or str(uuid4())
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        # Do not log bodies, query strings, API keys, entity IDs, or feature values.
        logger.info(json.dumps({
            "event": "http_request",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": status_code,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        }, separators=(",", ":")))
