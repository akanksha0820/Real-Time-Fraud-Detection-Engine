"""Build point-in-time-correct transaction features without future leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from python.common import ROOT


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def prior_window_features(frame: pd.DataFrame, windows: dict[str, str]) -> pd.DataFrame:
    """Compute rolling customer features with closed='left' to exclude the current row."""
    indexed = frame.set_index("event_timestamp")
    output = pd.DataFrame(index=frame.index)
    for suffix, window in windows.items():
        grouped = indexed.groupby("customer_id", sort=False)["amount_inr"]
        counts = grouped.rolling(window, closed="left").count().reset_index(level=0, drop=True)
        sums = grouped.rolling(window, closed="left").sum().reset_index(level=0, drop=True)
        # Duplicate timestamps make a timestamp index ambiguous; positional order is authoritative.
        output[f"tx_count_{suffix}"] = counts.to_numpy()
        output[f"amount_sum_{suffix}"] = sums.to_numpy()
    return output.fillna(0)


def prior_entity_count(frame: pd.DataFrame, entity_column: str, window: str) -> pd.Series:
    """Count strictly prior events for any entity while preserving input row order."""
    ordered = frame[[entity_column, "event_timestamp", "transaction_id"]].copy()
    ordered["_row_position"] = np.arange(len(ordered))
    ordered = ordered.sort_values(
        [entity_column, "event_timestamp", "transaction_id"], kind="mergesort"
    )
    indexed = ordered.set_index("event_timestamp")
    counts = (
        indexed.groupby(entity_column, sort=False)["transaction_id"]
        .rolling(window, closed="left")
        .count()
        .reset_index(level=0, drop=True)
        .to_numpy()
    )
    output = np.zeros(len(frame), dtype=float)
    output[ordered["_row_position"].to_numpy()] = np.nan_to_num(counts, nan=0.0)
    return pd.Series(output, index=frame.index)


def build_features(transactions: pd.DataFrame) -> pd.DataFrame:
    tx = transactions.copy()
    tx["event_timestamp"] = pd.to_datetime(tx["event_timestamp"])
    tx = tx.sort_values(["customer_id", "event_timestamp", "transaction_id"], kind="mergesort").reset_index(drop=True)
    customers = pd.read_parquet(ROOT / "data/reference/customers.parquet")
    profiles = pd.read_parquet(ROOT / "data/reference/customer_profiles.parquet")
    merchants = pd.read_parquet(ROOT / "data/reference/merchants.parquet")
    tx = (tx.merge(customers[["customer_id", "home_latitude", "home_longitude", "annual_income_inr"]], on="customer_id", validate="many_to_one")
          .merge(profiles[["customer_id", "typical_amount_inr", "customer_segment"]], on="customer_id", validate="many_to_one")
          .merge(merchants[["merchant_id", "merchant_risk_score", "risk_tier", "merchant_category"]], on="merchant_id", validate="many_to_one"))

    rolling = prior_window_features(tx, {"10m": "10min", "1h": "1h", "24h": "24h"})
    tx = pd.concat([tx, rolling], axis=1)
    customer_group = tx.groupby("customer_id", sort=False)
    previous_time = customer_group["event_timestamp"].shift(1)
    previous_lat = customer_group["latitude"].shift(1)
    previous_lon = customer_group["longitude"].shift(1)
    seconds_since_previous = (tx["event_timestamp"] - previous_time).dt.total_seconds()
    distance_previous = haversine_km(previous_lat, previous_lon, tx["latitude"], tx["longitude"])

    tx["hour"] = tx["event_timestamp"].dt.hour
    tx["day_of_week"] = tx["event_timestamp"].dt.dayofweek
    tx["is_weekend"] = tx["day_of_week"].ge(5).astype("int8")
    tx["is_night"] = tx["hour"].isin([0, 1, 2, 3, 4]).astype("int8")
    tx["amount_to_typical_ratio"] = (tx["amount_inr"] / tx["typical_amount_inr"].clip(lower=1)).clip(0, 100)
    tx["amount_to_income_ratio"] = (tx["amount_inr"] / tx["annual_income_inr"].clip(lower=1)).clip(0, 1)
    tx["distance_from_home_km"] = haversine_km(tx["home_latitude"], tx["home_longitude"], tx["latitude"], tx["longitude"])
    tx["seconds_since_previous_tx"] = seconds_since_previous.fillna(86400 * 30).clip(0, 86400 * 30)
    tx["distance_from_previous_km"] = np.nan_to_num(distance_previous, nan=0.0)
    hours = tx["seconds_since_previous_tx"].clip(lower=1) / 3600
    tx["travel_speed_kmh"] = (tx["distance_from_previous_km"] / hours).clip(0, 20_000)
    tx["customer_device_prior_uses"] = tx.groupby(["customer_id", "device_id"], sort=False).cumcount()
    tx["is_new_device"] = tx["customer_device_prior_uses"].eq(0).astype("int8")
    tx["customer_merchant_prior_uses"] = tx.groupby(["customer_id", "merchant_id"], sort=False).cumcount()
    tx["is_new_merchant"] = tx["customer_merchant_prior_uses"].eq(0).astype("int8")
    chronological = tx.sort_values(["event_timestamp", "transaction_id"], kind="mergesort")
    global_device_uses = chronological.groupby("device_id", sort=False).cumcount()
    tx.loc[chronological.index, "device_global_prior_uses"] = global_device_uses.to_numpy()
    tx["device_global_prior_uses"] = tx["device_global_prior_uses"].astype("int64")
    tx["merchant_tx_count_1h"] = prior_entity_count(tx, "merchant_id", "1h")
    tx["merchant_tx_count_24h"] = prior_entity_count(tx, "merchant_id", "24h")
    hourly_baseline = (tx["merchant_tx_count_24h"] + 1.0) / 24.0
    tx["merchant_velocity_ratio"] = ((tx["merchant_tx_count_1h"] + 1.0) / hourly_baseline).clip(0, 100)
    return tx.sort_values(["event_timestamp", "transaction_id"], kind="mergesort").reset_index(drop=True)


def main() -> None:
    source = ROOT / "data/raw/transactions.parquet"
    output = ROOT / "data/processed/transaction_features.parquet"
    frame = build_features(pd.read_parquet(source))
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    print(f"[SAVED] {output.relative_to(ROOT)} | rows={len(frame):,} | columns={len(frame.columns)}")
    print(f"Fraud rate: {frame.is_fraud.mean():.3%}")
    print("Leakage guard: all rolling windows use closed='left' (current transaction excluded).")


if __name__ == "__main__":
    main()
