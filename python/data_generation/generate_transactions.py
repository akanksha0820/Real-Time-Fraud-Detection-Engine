"""Generate legitimate activity and inject scenario-driven fraud attacks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from python.common import ROOT, load_settings


@dataclass(frozen=True)
class EntityTables:
    customers: pd.DataFrame
    accounts: pd.DataFrame
    devices: pd.DataFrame
    merchants: pd.DataFrame
    profiles: pd.DataFrame


def load_entities() -> EntityTables:
    ref = ROOT / "data/reference"
    return EntityTables(*(pd.read_parquet(ref / name) for name in [
        "customers.parquet", "accounts.parquet", "devices.parquet",
        "merchants.parquet", "customer_profiles.parquet"
    ]))


def _customer_maps(entities: EntityTables) -> pd.DataFrame:
    primary_accounts = entities.accounts.loc[entities.accounts["is_primary"], ["customer_id", "account_id"]]
    primary_devices = entities.devices.loc[entities.devices["is_primary"], ["customer_id", "device_id"]]
    return (entities.customers
            .merge(entities.profiles, on="customer_id", validate="one_to_one")
            .merge(primary_accounts, on="customer_id", validate="one_to_one")
            .merge(primary_devices, on="customer_id", validate="one_to_one"))


def generate_transactions(entities: EntityTables, count: int, seed: int, start: str, end: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    people = _customer_maps(entities)
    weights = people["mean_daily_transactions"].to_numpy(float)
    chosen = rng.choice(len(people), size=count, p=weights / weights.sum())
    tx = people.iloc[chosen].reset_index(drop=True)

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
    seconds = rng.integers(0, int((end_ts - start_ts).total_seconds()), count)
    timestamps = start_ts + pd.to_timedelta(seconds, unit="s")
    # Pull most legitimate timestamps toward each customer's active hours.
    active_mask = rng.random(count) < 0.88
    active_hour = rng.integers(0, 24, count)
    spans = np.maximum(tx["active_hour_end"].to_numpy() - tx["active_hour_start"].to_numpy(), 1)
    active_hour[active_mask] = tx.loc[active_mask, "active_hour_start"].to_numpy() + (
        rng.random(active_mask.sum()) * spans[active_mask]
    ).astype(int)
    timestamps = timestamps.normalize() + pd.to_timedelta(active_hour, unit="h") + pd.to_timedelta(rng.integers(0, 3600, count), unit="s")

    merchant_idx = rng.integers(0, len(entities.merchants), count)
    merchants = entities.merchants.iloc[merchant_idx].reset_index(drop=True)
    amounts = rng.lognormal(np.log(tx["typical_amount_inr"].clip(lower=1)), tx["amount_log_sigma"])
    amounts = np.clip(amounts, 10, 500_000).round(2)
    online = merchants["channel"].eq("ecommerce").to_numpy() | (rng.random(count) < tx["online_preference"].to_numpy() * 0.25)
    use_home = rng.random(count) > tx["travel_propensity"].to_numpy()
    lat = np.where(use_home, tx["home_latitude"] + rng.normal(0, .06, count), merchants["latitude"] + rng.normal(0, .08, count))
    lon = np.where(use_home, tx["home_longitude"] + rng.normal(0, .06, count), merchants["longitude"] + rng.normal(0, .08, count))

    result = pd.DataFrame({
        "transaction_id": [f"T{i:010d}" for i in range(1, count + 1)],
        "event_timestamp": timestamps,
        "customer_id": tx["customer_id"], "account_id": tx["account_id"],
        "device_id": tx["device_id"], "merchant_id": merchants["merchant_id"],
        "amount_inr": amounts, "currency": "INR",
        "payment_channel": np.where(online, "online", "pos"),
        "latitude": lat.round(6), "longitude": lon.round(6),
        "is_fraud": 0, "fraud_scenario": "legitimate",
        "legitimate_pattern": "routine",
    })
    inject_legitimate_hard_negatives(result, entities, rng)
    inject_fraud_scenarios(result, entities, rng)
    return result.sort_values(["event_timestamp", "transaction_id"]).reset_index(drop=True)


def _repeat_identity_and_time(
    tx: pd.DataFrame,
    idx: np.ndarray,
    burst_size: int,
    rng: np.random.Generator,
    seconds: int,
    include_merchant: bool = False,
) -> None:
    """Turn selected independent rows into customer-consistent event bursts."""
    leaders = idx[::burst_size]
    leader_for_row = np.repeat(leaders, burst_size)[: len(idx)]
    columns = ["customer_id", "account_id", "device_id"]
    if include_merchant:
        columns.append("merchant_id")
    tx.loc[idx, columns] = tx.loc[leader_for_row, columns].to_numpy()
    anchors = tx.loc[leader_for_row, "event_timestamp"].to_numpy()
    tx.loc[idx, "event_timestamp"] = pd.to_datetime(anchors) + pd.to_timedelta(
        rng.integers(0, seconds, len(idx)), unit="s"
    )


def inject_legitimate_hard_negatives(
    tx: pd.DataFrame, entities: EntityTables, rng: np.random.Generator
) -> None:
    """Create genuine behavior that resembles fraud but retains a legitimate label."""
    target = max(20, int(len(tx) * 0.015))
    available = np.arange(len(tx))
    rng.shuffle(available)
    patterns = [
        ("rapid_shopping_session", 0.30),
        ("payment_retry", 0.17),
        ("festival_large_purchase", 0.20),
        ("genuine_travel", 0.16),
        ("late_night_emergency", 0.10),
        ("device_replacement", 0.07),
    ]
    secondary_devices = (
        entities.devices.loc[~entities.devices["is_primary"]]
        .groupby("customer_id")["device_id"]
        .apply(list)
        .to_dict()
    )
    cursor = allocated = 0
    for position, (pattern, share) in enumerate(patterns):
        size = target - allocated if position == len(patterns) - 1 else int(target * share)
        idx = available[cursor : cursor + size]
        cursor += size
        allocated += size
        tx.loc[idx, "legitimate_pattern"] = pattern
        if pattern == "rapid_shopping_session":
            _repeat_identity_and_time(tx, idx, burst_size=7, rng=rng, seconds=540)
            tx.loc[idx, "amount_inr"] *= rng.uniform(0.35, 1.8, size)
        elif pattern == "payment_retry":
            _repeat_identity_and_time(tx, idx, burst_size=3, rng=rng, seconds=150, include_merchant=True)
            leaders = idx[::3]
            repeated = np.repeat(tx.loc[leaders, "amount_inr"].to_numpy(), 3)[:size]
            tx.loc[idx, "amount_inr"] = repeated * rng.uniform(0.98, 1.02, size)
        elif pattern == "festival_large_purchase":
            tx.loc[idx, "amount_inr"] *= rng.uniform(4, 10, size)
        elif pattern == "genuine_travel":
            tx.loc[idx, "latitude"] = rng.uniform(8, 35, size)
            tx.loc[idx, "longitude"] = rng.uniform(68, 96, size)
        elif pattern == "late_night_emergency":
            tx.loc[idx, "event_timestamp"] = tx.loc[idx, "event_timestamp"].dt.normalize() + pd.to_timedelta(
                rng.integers(0, 5, size), unit="h"
            ) + pd.to_timedelta(rng.integers(0, 3600, size), unit="s")
            tx.loc[idx, "amount_inr"] *= rng.uniform(1.0, 3.0, size)
        elif pattern == "device_replacement":
            replacements = []
            for row_index in idx:
                customer = tx.at[row_index, "customer_id"]
                candidates = secondary_devices.get(customer)
                replacements.append(rng.choice(candidates) if candidates else tx.at[row_index, "device_id"])
            tx.loc[idx, "device_id"] = replacements
    tx["amount_inr"] = tx["amount_inr"].clip(10, 2_000_000).round(2)


def inject_fraud_scenarios(tx: pd.DataFrame, entities: EntityTables, rng: np.random.Generator) -> None:
    """Mutate selected transactions into coherent attacks; labels follow the attack mechanism."""
    n = len(tx)
    target = max(1, int(n * 0.009))
    # pandas may expose an immutable NumPy view (notably with copy-on-write).
    # shuffle mutates in place, so request an owned, writable array explicitly.
    available = tx.index[tx["legitimate_pattern"].eq("routine")].to_numpy(copy=True)
    rng.shuffle(available)
    cursor = 0
    scenarios = [("account_takeover", .28), ("card_testing", .20), ("velocity_attack", .18),
                 ("impossible_travel", .15), ("high_risk_merchant", .11), ("fraud_ring", .08)]
    high_risk_merchants = entities.merchants.loc[entities.merchants["risk_tier"].eq("high"), "merchant_id"].to_numpy()
    all_devices = entities.devices["device_id"].to_numpy()
    ring_merchants = rng.choice(high_risk_merchants, min(12, len(high_risk_merchants)), replace=False)
    allocated = 0
    for index, (scenario, share) in enumerate(scenarios):
        size = target - allocated if index == len(scenarios) - 1 else int(target * share)
        idx = available[cursor:cursor + size]
        cursor += size
        allocated += size
        tx.loc[idx, ["is_fraud", "fraud_scenario"]] = [1, scenario]
        if scenario == "account_takeover":
            new_device = idx[rng.random(size) < 0.75]
            high_amount = idx[rng.random(size) < 0.70]
            night = idx[rng.random(size) < 0.45]
            remote = idx[rng.random(size) < 0.40]
            tx.loc[new_device, "device_id"] = rng.choice(all_devices, len(new_device))
            tx.loc[high_amount, "amount_inr"] *= rng.uniform(2, 8, len(high_amount))
            tx.loc[night, "event_timestamp"] = tx.loc[night, "event_timestamp"].dt.normalize() + pd.to_timedelta(rng.integers(0, 5, len(night)), unit="h")
            tx.loc[remote, "latitude"] = rng.uniform(5, 36, len(remote))
            tx.loc[remote, "longitude"] = rng.uniform(65, 100, len(remote))
        elif scenario in {"card_testing", "velocity_attack"}:
            burst_size = 6
            _repeat_identity_and_time(tx, idx, burst_size, rng, 540)
            if scenario == "card_testing":
                tx.loc[idx, "amount_inr"] = rng.uniform(10, 250, size)
            else:
                amplified = idx[rng.random(size) < 0.70]
                tx.loc[amplified, "amount_inr"] *= rng.uniform(1.3, 4, len(amplified))
        elif scenario == "impossible_travel":
            tx.loc[idx, "latitude"] = rng.uniform(-35, 55, size)
            tx.loc[idx, "longitude"] = rng.uniform(-120, 150, size)
        elif scenario == "high_risk_merchant":
            medium_risk_merchants = entities.merchants.loc[entities.merchants["risk_tier"].eq("medium"), "merchant_id"].to_numpy()
            use_high = rng.random(size) < 0.70
            tx.loc[idx[use_high], "merchant_id"] = rng.choice(high_risk_merchants, use_high.sum())
            tx.loc[idx[~use_high], "merchant_id"] = rng.choice(medium_risk_merchants, (~use_high).sum())
            amplified = idx[rng.random(size) < 0.60]
            tx.loc[amplified, "amount_inr"] *= rng.uniform(1.5, 5, len(amplified))
            novel_device = idx[rng.random(size) < 0.30]
            tx.loc[novel_device, "device_id"] = rng.choice(all_devices, len(novel_device))
        elif scenario == "fraud_ring":
            shared_device_pool = rng.choice(all_devices, min(60, len(all_devices)), replace=False)
            tx.loc[idx, "merchant_id"] = rng.choice(ring_merchants, size)
            tx.loc[idx, "device_id"] = rng.choice(shared_device_pool, size)
            amplified = idx[rng.random(size) < 0.55]
            tx.loc[amplified, "amount_inr"] *= rng.uniform(1.2, 4, len(amplified))
    tx["amount_inr"] = tx["amount_inr"].clip(10, 2_000_000).round(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--transactions", type=int)
    args = parser.parse_args()
    settings = load_settings()
    simulation = settings["simulation"]
    count = args.transactions or (simulation["quick_transactions"] if args.quick else simulation["target_transactions"])
    output = ROOT / "data/raw/transactions.parquet"
    frame = generate_transactions(load_entities(), count, settings["seed"], simulation["start_date"], simulation["end_date"])
    frame["simulation_version"] = simulation.get("version", "unknown")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    print(f"[SAVED] {output.relative_to(ROOT)} | rows={len(frame):,} | fraud_rate={frame.is_fraud.mean():.3%}")
    print(frame.loc[frame.is_fraud.eq(1), "fraud_scenario"].value_counts().to_string())
    print("\nLegitimate hard-negative patterns:")
    print(frame.loc[frame.is_fraud.eq(0), "legitimate_pattern"].value_counts().to_string())


if __name__ == "__main__":
    main()
