"""Fail-fast integrity and plausibility checks for generated datasets."""

import pandas as pd
from python.common import ROOT


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
    print(f"[PASS] {message}")


def main() -> None:
    ref, raw = ROOT / "data/reference", ROOT / "data/raw"
    customers = pd.read_parquet(ref / "customers.parquet")
    accounts = pd.read_parquet(ref / "accounts.parquet")
    devices = pd.read_parquet(ref / "devices.parquet")
    merchants = pd.read_parquet(ref / "merchants.parquet")
    tx = pd.read_parquet(raw / "transactions.parquet")
    require(customers.customer_id.is_unique, "customer IDs are unique")
    require(accounts.account_id.is_unique, "account IDs are unique")
    require(devices.device_id.is_unique, "device IDs are unique")
    require(merchants.merchant_id.is_unique, "merchant IDs are unique")
    require(tx.transaction_id.is_unique, "transaction IDs are unique")
    require(tx.customer_id.isin(customers.customer_id).all(), "transaction customer foreign keys are valid")
    require(tx.account_id.isin(accounts.account_id).all(), "transaction account foreign keys are valid")
    require(tx.device_id.isin(devices.device_id).all(), "transaction device foreign keys are valid")
    require(tx.merchant_id.isin(merchants.merchant_id).all(), "transaction merchant foreign keys are valid")
    require(tx.amount_inr.between(0.01, 2_000_000).all(), "transaction amounts are within bounds")
    require(tx.is_fraud.mean() >= .005 and tx.is_fraud.mean() <= .015, "fraud rate is within 0.5%-1.5%")
    require(set(tx.loc[tx.is_fraud.eq(1), "fraud_scenario"]) != {"legitimate"}, "fraud labels are scenario-based")
    require("legitimate_pattern" in tx.columns, "legitimate hard-negative provenance is present")
    legitimate_patterns = tx.loc[tx.is_fraud.eq(0), "legitimate_pattern"]
    require(legitimate_patterns.ne("routine").mean() >= .01, "at least 1% of legitimate events are hard negatives")
    require(legitimate_patterns.eq("rapid_shopping_session").sum() > 0, "legitimate rapid-velocity sessions exist")
    print("Data validation completed successfully.")


if __name__ == "__main__":
    main()
