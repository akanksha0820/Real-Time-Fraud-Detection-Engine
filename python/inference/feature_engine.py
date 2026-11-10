"""Point-in-time online features with an explicit mutable-state boundary."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from python.common import ROOT
from python.features.build_features import haversine_km
from python.inference.state import InMemoryStateStore, StateStore


class OutOfOrderEventError(ValueError):
    """Raised when an event predates the last committed event for its customer."""


class OnlineFeatureEngine:
    def __init__(self, state: StateStore | None = None, reference_root: Path | None = None):
        reference_root = reference_root or ROOT / "data/reference"
        self.state = state or InMemoryStateStore()
        self.customers = pd.read_parquet(reference_root / "customers.parquet").set_index("customer_id")
        self.profiles = pd.read_parquet(reference_root / "customer_profiles.parquet").set_index("customer_id")
        self.accounts = pd.read_parquet(reference_root / "accounts.parquet").set_index("account_id")
        self.merchants = pd.read_parquet(reference_root / "merchants.parquet").set_index("merchant_id")

    @staticmethod
    def _timestamp(value) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        return timestamp.tz_convert(None) if timestamp.tzinfo is not None else timestamp

    def compute_and_commit(self, event: dict) -> dict:
        timestamp = self._timestamp(event["event_timestamp"])
        customer_id, merchant_id, device_id = event["customer_id"], event["merchant_id"], event["device_id"]
        customer = self.customers.loc[customer_id]
        account = self.accounts.loc[event["account_id"]]
        if str(account.customer_id) != customer_id:
            raise ValueError(f"Account {event['account_id']} does not belong to customer {customer_id}")
        previous = self.state.get_last_customer_event(customer_id)
        if previous is not None and timestamp < previous[0]:
            raise OutOfOrderEventError(
                f"Event {event['transaction_id']} at {timestamp} predates the customer's last event at {previous[0]}"
            )
        profile = self.profiles.loc[customer_id]
        merchant = self.merchants.loc[merchant_id]
        amount = float(event["amount_inr"])

        customer_queue = [row for row in self.state.get_customer_events(customer_id)
                          if timestamp - row[0] <= pd.Timedelta("24h")]
        aggregates = {}
        for name, delta in [("10m", pd.Timedelta("10m")), ("1h", pd.Timedelta("1h")), ("24h", pd.Timedelta("24h"))]:
            prior = [row for row in customer_queue if pd.Timedelta(0) < timestamp - row[0] <= delta]
            aggregates[name] = (len(prior), sum(row[1] for row in prior))

        if previous is None:
            seconds, previous_distance = 86400 * 30, 0.0
        else:
            seconds = min(max((timestamp - previous[0]).total_seconds(), 0), 86400 * 30)
            previous_distance = float(haversine_km(previous[2], previous[3], event["latitude"], event["longitude"]))
        speed_hours = max(seconds, 1) / 3600

        merchant_queue = [row for row in self.state.get_merchant_events(merchant_id)
                          if timestamp - row <= pd.Timedelta("24h")]
        merchant_1h = sum(pd.Timedelta(0) < timestamp - prior <= pd.Timedelta("1h") for prior in merchant_queue)
        merchant_24h = sum(pd.Timedelta(0) < timestamp - prior <= pd.Timedelta("24h") for prior in merchant_queue)
        merchant_velocity = min((merchant_1h + 1) / ((merchant_24h + 1) / 24), 100)
        customer_device_uses = self.state.get_customer_device_count(customer_id, device_id)
        customer_merchant_uses = self.state.get_customer_merchant_count(customer_id, merchant_id)

        features = {
            "amount_inr": amount,
            "hour": timestamp.hour,
            "day_of_week": timestamp.dayofweek,
            "is_weekend": int(timestamp.dayofweek >= 5),
            "is_night": int(timestamp.hour <= 4),
            "merchant_risk_score": float(merchant.merchant_risk_score),
            "amount_to_typical_ratio": min(amount / max(profile.typical_amount_inr, 1), 100),
            "amount_to_income_ratio": min(amount / max(customer.annual_income_inr, 1), 1),
            "distance_from_home_km": float(haversine_km(customer.home_latitude, customer.home_longitude, event["latitude"], event["longitude"])),
            "seconds_since_previous_tx": seconds,
            "distance_from_previous_km": previous_distance,
            "travel_speed_kmh": min(previous_distance / speed_hours, 20_000),
            "customer_device_prior_uses": customer_device_uses,
            "is_new_device": int(customer_device_uses == 0),
            "customer_merchant_prior_uses": customer_merchant_uses,
            "is_new_merchant": int(customer_merchant_uses == 0),
            "tx_count_10m": aggregates["10m"][0], "amount_sum_10m": aggregates["10m"][1],
            "tx_count_1h": aggregates["1h"][0], "amount_sum_1h": aggregates["1h"][1],
            "tx_count_24h": aggregates["24h"][0], "amount_sum_24h": aggregates["24h"][1],
            "device_global_prior_uses": self.state.get_device_global_count(device_id),
            "merchant_tx_count_1h": merchant_1h,
            "merchant_tx_count_24h": merchant_24h,
            "merchant_velocity_ratio": merchant_velocity,
            "payment_channel": event["payment_channel"],
            "risk_tier": str(merchant.risk_tier),
            "merchant_category": merchant.merchant_category,
            "customer_segment": str(profile.customer_segment),
        }

        record = (timestamp, amount, float(event["latitude"]), float(event["longitude"]), device_id, merchant_id)
        self.state.commit(customer_id, merchant_id, device_id, record, customer_queue, merchant_queue)
        return features
