"""Thread-safe calibrated scoring, idempotency, audit logs, and service metrics."""

from __future__ import annotations

import json
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

import joblib
import pandas as pd

from python.common import ROOT
from python.decision_engine.policy import decide
from python.inference.feature_engine import OnlineFeatureEngine
from python.inference.predict import predict_fraud_probability
from python.inference.state import StateStore, state_store_from_environment


class RiskService:
    def __init__(self, model_path: Path | None = None, state: StateStore | None = None,
                 audit_path: Path | None = ROOT / "reports/inference/decisions.jsonl"):
        self.bundle = joblib.load(model_path or ROOT / "artifacts/model.joblib")
        self.state = state or state_store_from_environment()
        self.features = OnlineFeatureEngine(self.state)
        self.audit_path = audit_path
        self.lock = RLock()
        self.started_at = datetime.now(timezone.utc)
        self.decision_counts: Counter = Counter()
        self.latencies_ms: deque[float] = deque(maxlen=10_000)
        self.total_requests = 0
        self.idempotent_replays = 0

    def _write_audit(self, event: dict, result: dict) -> None:
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"scored_at_utc": datetime.now(timezone.utc).isoformat(),
                  "transaction_id": event["transaction_id"], "customer_id": event["customer_id"],
                  "merchant_id": event["merchant_id"], "amount_inr": float(event["amount_inr"]),
                  "fraud_probability": result["fraud_probability"], "decision": result["decision"],
                  "reasons": result["reasons"]}
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def score(self, event: dict) -> dict:
        started = time.perf_counter()
        transaction_id = event["transaction_id"]
        with self.lock:
            with self.state.lock():
                self.total_requests += 1
                cached = self.state.get_decision(transaction_id)
                if cached is not None:
                    self.idempotent_replays += 1
                    return cached.copy()
                feature_snapshot = self.features.compute_and_commit(event)
                probability = predict_fraud_probability(self.bundle, pd.DataFrame([feature_snapshot]))[0]
                result = {"transaction_id": transaction_id,
                          **decide(probability, feature_snapshot, self.bundle["review_threshold"],
                                   self.bundle["block_threshold"], self.bundle.get("policy", {})),
                          "feature_snapshot": feature_snapshot}
                self.state.save_decision(transaction_id, result)
                self.decision_counts[result["decision"]] += 1
                self._write_audit(event, result)
                self.latencies_ms.append((time.perf_counter() - started) * 1000)
                return result

    def metrics_snapshot(self) -> dict:
        values = sorted(self.latencies_ms)
        percentile = lambda q: values[min(int((len(values) - 1) * q), len(values) - 1)] if values else 0.0
        return {"started_at_utc": self.started_at.isoformat(), "total_requests": self.total_requests,
                "unique_transactions": sum(self.decision_counts.values()), "idempotent_replays": self.idempotent_replays,
                "decisions": dict(self.decision_counts),
                "latency_ms": {"p50": round(percentile(.50), 3), "p95": round(percentile(.95), 3),
                               "p99": round(percentile(.99), 3)}}
