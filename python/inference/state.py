"""State-store contract plus in-memory and Redis implementations."""
from __future__ import annotations
from collections import defaultdict, deque
from contextlib import nullcontext
from dataclasses import dataclass, field
import json, os
from typing import ContextManager, Protocol
import pandas as pd

EventRecord = tuple[pd.Timestamp, float, float, float, str, str]

class StateStore(Protocol):
    def lock(self) -> ContextManager: ...
    def get_customer_events(self, key: str) -> list[EventRecord]: ...
    def get_merchant_events(self, key: str) -> list[pd.Timestamp]: ...
    def get_last_customer_event(self, key: str) -> EventRecord | None: ...
    def get_customer_device_count(self, customer: str, device: str) -> int: ...
    def get_customer_merchant_count(self, customer: str, merchant: str) -> int: ...
    def get_device_global_count(self, device: str) -> int: ...
    def get_decision(self, transaction: str) -> dict | None: ...
    def commit(self, customer: str, merchant: str, device: str, record: EventRecord,
               customer_events: list[EventRecord], merchant_events: list[pd.Timestamp]) -> None: ...
    def save_decision(self, transaction: str, decision: dict) -> None: ...
    def reset(self) -> None: ...

@dataclass
class InMemoryStateStore:
    customer_events: dict[str, deque] = field(default_factory=lambda: defaultdict(deque))
    merchant_events: dict[str, deque] = field(default_factory=lambda: defaultdict(deque))
    customer_device_counts: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    customer_merchant_counts: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    device_global_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_customer_event: dict[str, EventRecord] = field(default_factory=dict)
    decisions: dict[str, dict] = field(default_factory=dict)
    def lock(self): return nullcontext()
    def get_customer_events(self, key): return list(self.customer_events[key])
    def get_merchant_events(self, key): return list(self.merchant_events[key])
    def get_last_customer_event(self, key): return self.last_customer_event.get(key)
    def get_customer_device_count(self, customer, device): return self.customer_device_counts[(customer, device)]
    def get_customer_merchant_count(self, customer, merchant): return self.customer_merchant_counts[(customer, merchant)]
    def get_device_global_count(self, device): return self.device_global_counts[device]
    def get_decision(self, transaction):
        value = self.decisions.get(transaction)
        return value.copy() if value is not None else None
    def commit(self, customer, merchant, device, record, customer_events, merchant_events):
        self.customer_events[customer] = deque([*customer_events, record])
        self.merchant_events[merchant] = deque([*merchant_events, record[0]])
        self.last_customer_event[customer] = record
        self.customer_device_counts[(customer, device)] += 1
        self.customer_merchant_counts[(customer, merchant)] += 1
        self.device_global_counts[device] += 1
    def save_decision(self, transaction, decision): self.decisions[transaction] = decision.copy()
    def reset(self):
        self.customer_events.clear(); self.merchant_events.clear(); self.customer_device_counts.clear()
        self.customer_merchant_counts.clear(); self.device_global_counts.clear()
        self.last_customer_event.clear(); self.decisions.clear()

class RedisStateStore:
    """Shared state guarded by a Redis distributed lock for atomic scoring."""
    def __init__(self, url=None, namespace="fraud-risk"):
        try: import redis
        except ImportError as exc: raise RuntimeError("Install redis to use RedisStateStore") from exc
        self.client = redis.Redis.from_url(url or os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
        self.namespace = namespace
        self.client.ping()
    def _key(self, kind, identifier): return f"{self.namespace}:{kind}:{identifier}"
    @staticmethod
    def _pair(left, right): return f"{left}|{right}"
    @staticmethod
    def _encode(record): return [pd.Timestamp(record[0]).isoformat(), *record[1:]]
    @staticmethod
    def _decode(value): return (pd.Timestamp(value[0]), float(value[1]), float(value[2]), float(value[3]), value[4], value[5])
    def lock(self): return self.client.lock(self._key("lock", "scoring"), timeout=30, blocking_timeout=10)
    def _get(self, key, default):
        value = self.client.get(key)
        return default if value is None else json.loads(value)
    def get_customer_events(self, key): return [self._decode(x) for x in self._get(self._key("customer-events", key), [])]
    def get_merchant_events(self, key): return [pd.Timestamp(x) for x in self._get(self._key("merchant-events", key), [])]
    def get_last_customer_event(self, key):
        value = self._get(self._key("last-event", key), None)
        return None if value is None else self._decode(value)
    def _count(self, kind, field): return int(self.client.hget(self._key("counts", kind), field) or 0)
    def get_customer_device_count(self, customer, device): return self._count("customer-device", self._pair(customer, device))
    def get_customer_merchant_count(self, customer, merchant): return self._count("customer-merchant", self._pair(customer, merchant))
    def get_device_global_count(self, device): return self._count("device-global", device)
    def get_decision(self, transaction): return self._get(self._key("decision", transaction), None)
    def commit(self, customer, merchant, device, record, customer_events, merchant_events):
        pipe = self.client.pipeline(transaction=True)
        pipe.set(self._key("customer-events", customer), json.dumps([self._encode(x) for x in [*customer_events, record]]))
        pipe.set(self._key("merchant-events", merchant), json.dumps([pd.Timestamp(x).isoformat() for x in [*merchant_events, record[0]]]))
        pipe.set(self._key("last-event", customer), json.dumps(self._encode(record)))
        pipe.hincrby(self._key("counts", "customer-device"), self._pair(customer, device), 1)
        pipe.hincrby(self._key("counts", "customer-merchant"), self._pair(customer, merchant), 1)
        pipe.hincrby(self._key("counts", "device-global"), device, 1); pipe.execute()
    def save_decision(self, transaction, decision): self.client.set(self._key("decision", transaction), json.dumps(decision, default=str))
    def reset(self):
        cursor, pattern = 0, f"{self.namespace}:*"
        while True:
            cursor, keys = self.client.scan(cursor=cursor, match=pattern, count=500)
            if keys: self.client.delete(*keys)
            if cursor == 0: break

def state_store_from_environment() -> StateStore:
    backend = os.getenv("FRAUD_STATE_BACKEND", "memory").lower()
    if backend == "redis": return RedisStateStore()
    if backend != "memory": raise ValueError("FRAUD_STATE_BACKEND must be 'memory' or 'redis'")
    return InMemoryStateStore()
