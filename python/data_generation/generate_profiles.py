"""Create latent customer behavior profiles used by transaction simulation."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from python.common import ROOT, load_settings


def build_profiles(customers: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(customers)
    income = customers["annual_income_inr"].to_numpy(dtype=float)
    segment = pd.cut(
        income,
        bins=[-np.inf, 300_000, 800_000, 2_000_000, np.inf],
        labels=["value", "mass", "affluent", "high_net_worth"],
    ).astype("string")
    income_scale = np.clip(income / 800_000, 0.25, 8.0)
    typical_amount = np.clip(rng.lognormal(np.log(850 * income_scale), 0.55), 80, 75_000)
    daily_rate = np.clip(rng.gamma(1.8, 0.75, n) * np.sqrt(income_scale), 0.05, 10.0)
    active_start = rng.choice(np.arange(6, 13), n, p=[0.03, 0.07, 0.13, 0.18, 0.20, 0.18, 0.21])
    active_hours = rng.integers(9, 17, n)

    profiles = pd.DataFrame(
        {
            "customer_id": customers["customer_id"],
            "customer_segment": segment,
            "mean_daily_transactions": daily_rate.round(3),
            "typical_amount_inr": typical_amount.round(2),
            "amount_log_sigma": rng.uniform(0.35, 0.90, n).round(3),
            "active_hour_start": active_start,
            "active_hour_end": np.minimum(active_start + active_hours, 23),
            "weekend_multiplier": rng.uniform(0.75, 1.35, n).round(3),
            "online_preference": rng.beta(2.2, 2.5, n).round(3),
            "travel_propensity": rng.beta(1.3, 8.0, n).round(3),
            "cash_preference": rng.beta(1.7, 5.5, n).round(3),
        }
    )
    if profiles["customer_id"].duplicated().any() or profiles.isna().any().any():
        raise ValueError("Behavior profiles failed uniqueness/null validation")
    return profiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    settings = load_settings()
    seed = args.seed if args.seed is not None else settings["seed"]
    source = ROOT / "data/reference/customers.parquet"
    output = ROOT / "data/reference/customer_profiles.parquet"
    profiles = build_profiles(pd.read_parquet(source), seed)
    profiles.to_parquet(output, index=False)
    print(f"[SAVED] {output.relative_to(ROOT)} | rows={len(profiles):,} | columns={len(profiles.columns)}")
    print(profiles["customer_segment"].value_counts().to_string())
    segment_share = profiles["customer_segment"].value_counts(normalize=True)
    if segment_share.max() > 0.80:
        raise ValueError(
            "Customer segmentation is implausibly concentrated: "
            f"{segment_share.idxmax()}={segment_share.max():.1%}. "
            "Inspect annual_income_inr generation before continuing."
        )


if __name__ == "__main__":
    main()
