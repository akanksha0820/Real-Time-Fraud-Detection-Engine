"""HTTP interface for health checks and real-time scoring."""

import json
import os
from datetime import datetime
from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api.observability import request_observability
from api.security import enforce_score_rate_limit, require_api_key
from python.common import ROOT
from python.inference.feature_engine import OutOfOrderEventError
from python.inference.service import RiskService
from python.monitoring.operations import OperationsRepository

production = os.getenv("FRAUD_ENV", "development").lower() == "production"
app = FastAPI(title="Real-Time Fraud & Transaction Risk Engine", version="1.1.0",
              docs_url=None if production else "/docs",
              redoc_url=None if production else "/redoc")
app.middleware("http")(request_observability)


class TransactionEvent(BaseModel):
    transaction_id: str = Field(min_length=1, max_length=128)
    event_timestamp: datetime
    customer_id: str = Field(min_length=1, max_length=128)
    account_id: str = Field(min_length=1, max_length=128)
    device_id: str = Field(min_length=1, max_length=128)
    merchant_id: str = Field(min_length=1, max_length=128)
    amount_inr: float = Field(gt=0, le=2_000_000)
    payment_channel: str = Field(pattern="^(online|pos)$")
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

class CaseUpdate(BaseModel):
    status: Literal["OPEN", "IN_PROGRESS", "RESOLVED", "CLOSED"]
    assigned_to: str | None = Field(default=None, max_length=200)
    outcome: Literal["FRAUD", "LEGITIMATE", "INCONCLUSIVE"] | None = None
    notes: str | None = Field(default=None, max_length=4000)


@lru_cache(maxsize=1)
def service() -> RiskService:
    return RiskService()

@lru_cache(maxsize=1)
def operations() -> OperationsRepository:
    return OperationsRepository()


@app.get("/live")
def live() -> dict:
    return {"status": "alive"}


@app.get("/health")
def health() -> dict:
    """Backward-compatible liveness endpoint; does not initialize dependencies."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    checks = {"model": False, "state": False, "postgres": False}
    try:
        risk_service = service()
        checks["model"] = risk_service is not None
        state = risk_service.state
        checks["state"] = bool(state.client.ping()) if hasattr(state, "client") else True
        checks["postgres"] = operations().ping()
    except Exception:
        return JSONResponse(status_code=503, content={"status": "not_ready", "checks": checks})
    return {"status": "ready", "checks": checks}


@app.get("/model")
def model_metadata(_: str = Depends(require_api_key)) -> dict:
    with (ROOT / "artifacts/model_metadata.json").open(encoding="utf-8") as handle:
        return json.load(handle)


@app.get("/metrics")
def metrics(_: str = Depends(require_api_key)) -> dict:
    return service().metrics_snapshot()

@app.get("/operations/summary")
def operations_summary(_: str = Depends(require_api_key)) -> dict:
    return operations().summary()

@app.get("/operations/review-queue")
def review_queue(limit: int = 100, _: str = Depends(require_api_key)) -> list[dict]:
    return operations().review_queue(min(max(limit, 1), 500))

@app.get("/operations/merchants")
def merchant_risk(limit: int = 20, _: str = Depends(require_api_key)) -> list[dict]:
    return operations().merchant_risk(min(max(limit, 1), 100))

@app.get("/operations/model-governance")
def model_governance(_: str = Depends(require_api_key)) -> dict:
    return operations().model_governance()

@app.patch("/operations/cases/{case_id}")
def update_case(case_id: int, update: CaseUpdate, _: str = Depends(require_api_key)) -> dict:
    try:
        return operations().update_case(case_id, **update.model_dump())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown case: {case_id}") from exc


@app.post("/score")
def score(event: TransactionEvent, request: Request, api_key: str = Depends(require_api_key)) -> dict:
    enforce_score_rate_limit(request, api_key)
    try:
        return service().score(event.model_dump())
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"Unknown reference entity: {exc}") from exc
    except OutOfOrderEventError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
