"""Generate deterministic master entities for the fraud-risk simulation.

Outputs:
    data/reference/customers.parquet
    data/reference/accounts.parquet
    data/reference/devices.parquet
    data/reference/merchants.parquet

Run from the repository root:
    python python/data_generation/generate_entities.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker


RANDOM_SEED = 42
NUM_CUSTOMERS = 50_000
NUM_MERCHANTS = 8_000
SIMULATION_START = pd.Timestamp("2025-01-01")
SIMULATION_END = pd.Timestamp("2026-12-31")

INDIAN_CITIES = pd.DataFrame(
    [
        ("Mumbai", "Maharashtra", 19.0760, 72.8777, 0.13),
        ("Delhi", "Delhi", 28.6139, 77.2090, 0.12),
        ("Bengaluru", "Karnataka", 12.9716, 77.5946, 0.11),
        ("Hyderabad", "Telangana", 17.3850, 78.4867, 0.09),
        ("Chennai", "Tamil Nadu", 13.0827, 80.2707, 0.08),
        ("Kolkata", "West Bengal", 22.5726, 88.3639, 0.08),
        ("Pune", "Maharashtra", 18.5204, 73.8567, 0.07),
        ("Ahmedabad", "Gujarat", 23.0225, 72.5714, 0.07),
        ("Jaipur", "Rajasthan", 26.9124, 75.7873, 0.05),
        ("Lucknow", "Uttar Pradesh", 26.8467, 80.9462, 0.05),
        ("Kochi", "Kerala", 9.9312, 76.2673, 0.04),
        ("Chandigarh", "Chandigarh", 30.7333, 76.7794, 0.04),
        ("Srinagar", "Jammu and Kashmir", 34.0837, 74.7973, 0.04),
        ("Guwahati", "Assam", 26.1445, 91.7362, 0.03),
    ],
    columns=["city", "state", "latitude", "longitude", "weight"],
)

MERCHANT_CATEGORIES = pd.DataFrame(
    [
        ("5411", "Grocery Stores", 0.16, "low", 700.0),
        ("5812", "Restaurants", 0.12, "low", 900.0),
        ("5311", "Department Stores", 0.09, "low", 2_500.0),
        ("5541", "Fuel Stations", 0.08, "low", 1_800.0),
        ("5732", "Electronics", 0.08, "medium", 12_000.0),
        ("5651", "Apparel", 0.08, "low", 3_500.0),
        ("5912", "Pharmacies", 0.07, "low", 1_100.0),
        ("4814", "Telecom Services", 0.06, "medium", 1_000.0),
        ("5942", "Book Stores", 0.04, "low", 800.0),
        ("4722", "Travel Agencies", 0.05, "medium", 15_000.0),
        ("7011", "Hotels", 0.05, "medium", 9_000.0),
        ("4899", "Digital Services", 0.05, "medium", 1_500.0),
        ("5999", "Online Marketplaces", 0.04, "medium", 4_000.0),
        ("7995", "Betting and Gaming", 0.015, "high", 3_000.0),
        ("6051", "Crypto and Money Services", 0.015, "high", 10_000.0),
    ],
    columns=["merchant_category_code", "merchant_category", "weight", "base_risk", "typical_ticket"],
)


def random_dates(
    rng: np.random.Generator, start: pd.Timestamp, end: pd.Timestamp, size: int
) -> pd.DatetimeIndex:
    """Sample normalized timestamps inclusively between two dates."""
    if end < start:
        raise ValueError(f"Invalid date range: {start=} is after {end=}")
    offsets = rng.integers(0, (end - start).days + 1, size=size)
    return pd.DatetimeIndex(start + pd.to_timedelta(offsets, unit="D"))


def sample_locations(rng: np.random.Generator, size: int) -> pd.DataFrame:
    indices = rng.choice(
        len(INDIAN_CITIES), size=size, p=INDIAN_CITIES["weight"].to_numpy()
    )
    locations = INDIAN_CITIES.iloc[indices].reset_index(drop=True).copy()
    locations["latitude"] += rng.normal(0, 0.08, size)
    locations["longitude"] += rng.normal(0, 0.08, size)
    return locations.drop(columns="weight")


def generate_customers(
    rng: np.random.Generator, fake: Faker, count: int
) -> pd.DataFrame:
    customer_ids = np.array([f"C{i:07d}" for i in range(1, count + 1)])
    signup_dates = random_dates(rng, pd.Timestamp("2017-01-01"), pd.Timestamp("2024-12-31"), count)
    birth_dates = random_dates(rng, pd.Timestamp("1945-01-01"), pd.Timestamp("2006-12-31"), count)
    locations = sample_locations(rng, count)
    occupations = rng.choice(
        ["salaried", "self_employed", "student", "retired", "homemaker", "professional"],
        size=count,
        p=[0.42, 0.18, 0.10, 0.08, 0.08, 0.14],
    )
    # Parameters are on the natural-log INR scale. Medians range from roughly
    # INR 1.8 lakh for students to INR 12 lakh for professionals, with broad
    # overlap so occupation or income never becomes a trivial fraud proxy.
    income_ranges = {
        "salaried": (13.30, 0.55),
        "self_employed": (13.59, 0.75),
        "student": (12.10, 0.50),
        "retired": (12.79, 0.55),
        "homemaker": (12.39, 0.60),
        "professional": (14.00, 0.60),
    }
    annual_income = np.empty(count)
    for occupation, (mean, sigma) in income_ranges.items():
        mask = occupations == occupation
        annual_income[mask] = rng.lognormal(mean, sigma, mask.sum())
    annual_income = np.clip(annual_income, 60_000, 15_000_000).round(-2)

    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "full_name": [fake.name() for _ in range(count)],
            "date_of_birth": birth_dates,
            "gender": rng.choice(["female", "male", "non_binary"], count, p=[0.485, 0.51, 0.005]),
            "occupation_group": occupations,
            "annual_income_inr": annual_income.astype("int64"),
            "home_city": locations["city"],
            "home_state": locations["state"],
            "home_latitude": locations["latitude"].round(6),
            "home_longitude": locations["longitude"].round(6),
            "signup_date": signup_dates,
            "kyc_level": rng.choice(["basic", "standard", "enhanced"], count, p=[0.12, 0.76, 0.12]),
            "customer_status": rng.choice(["active", "dormant", "closed"], count, p=[0.94, 0.05, 0.01]),
        }
    )
    return customers.sort_values("customer_id").reset_index(drop=True)


def generate_accounts(rng: np.random.Generator, customers: pd.DataFrame) -> pd.DataFrame:
    account_counts = rng.choice([1, 2, 3], len(customers), p=[0.865, 0.12, 0.015])
    customer_ids = np.repeat(customers["customer_id"].to_numpy(), account_counts)
    customer_signup = np.repeat(customers["signup_date"].to_numpy(), account_counts)
    count = len(customer_ids)
    open_delays = rng.integers(0, 181, count)
    opened_at = pd.to_datetime(customer_signup) + pd.to_timedelta(open_delays, unit="D")
    opened_at = pd.DatetimeIndex(np.minimum(opened_at.values, SIMULATION_START.to_datetime64()))
    account_type = rng.choice(["savings", "current", "credit_card"], count, p=[0.72, 0.10, 0.18])
    credit_limits = np.where(
        account_type == "credit_card",
        np.clip(rng.lognormal(11.1, 0.75, count), 20_000, 2_000_000).round(-3),
        0,
    ).astype("int64")
    account_sequence = pd.Series(customer_ids).groupby(customer_ids).cumcount().to_numpy()
    is_primary = account_sequence == 0

    return pd.DataFrame(
        {
            "account_id": [f"A{i:08d}" for i in range(1, count + 1)],
            "customer_id": customer_ids,
            "account_type": account_type,
            "currency": "INR",
            "opened_at": opened_at,
            "account_status": rng.choice(["active", "dormant", "frozen", "closed"], count, p=[0.955, 0.03, 0.005, 0.01]),
            "credit_limit_inr": credit_limits,
            "is_primary": is_primary,
        }
    )


def generate_devices(rng: np.random.Generator, customers: pd.DataFrame) -> pd.DataFrame:
    device_counts = rng.choice([1, 2, 3, 4], len(customers), p=[0.47, 0.37, 0.13, 0.03])
    customer_ids = np.repeat(customers["customer_id"].to_numpy(), device_counts)
    signup_dates = np.repeat(customers["signup_date"].to_numpy(), device_counts)
    count = len(customer_ids)
    device_type = rng.choice(["android", "ios", "web_browser", "tablet"], count, p=[0.57, 0.25, 0.15, 0.03])
    os_version = []
    for kind in device_type:
        if kind == "android":
            os_version.append(f"Android {rng.integers(10, 15)}")
        elif kind in {"ios", "tablet"}:
            os_version.append(f"iOS {rng.integers(15, 19)}")
        else:
            os_version.append(rng.choice(["Windows 11", "macOS 15", "Linux"]))
    first_seen_delay = rng.integers(0, 1_500, count)
    first_seen = pd.to_datetime(signup_dates) + pd.to_timedelta(first_seen_delay, unit="D")
    first_seen = pd.DatetimeIndex(np.minimum(first_seen.values, SIMULATION_START.to_datetime64()))
    sequence = pd.Series(customer_ids).groupby(customer_ids).cumcount().to_numpy()

    return pd.DataFrame(
        {
            "device_id": [f"D{i:09d}" for i in range(1, count + 1)],
            "customer_id": customer_ids,
            "device_type": device_type,
            "os_version": os_version,
            "first_seen_at": first_seen,
            "is_primary": sequence == 0,
            "trust_status": rng.choice(["trusted", "unverified", "restricted"], count, p=[0.84, 0.155, 0.005]),
        }
    )


def generate_merchants(rng: np.random.Generator, fake: Faker, count: int) -> pd.DataFrame:
    category_indices = rng.choice(
        len(MERCHANT_CATEGORIES), size=count, p=MERCHANT_CATEGORIES["weight"].to_numpy()
    )
    categories = MERCHANT_CATEGORIES.iloc[category_indices].reset_index(drop=True)
    locations = sample_locations(rng, count)
    channel = rng.choice(["card_present", "ecommerce", "hybrid"], count, p=[0.49, 0.31, 0.20])
    base_to_score = {"low": 0.08, "medium": 0.28, "high": 0.62}
    risk_score = np.array([base_to_score[x] for x in categories["base_risk"]])
    risk_score = np.clip(risk_score + rng.beta(2, 10, count) * 0.22, 0.01, 0.95)
    onboarding_dates = random_dates(rng, pd.Timestamp("2016-01-01"), pd.Timestamp("2024-12-31"), count)

    return pd.DataFrame(
        {
            "merchant_id": [f"M{i:06d}" for i in range(1, count + 1)],
            "merchant_name": [fake.company() for _ in range(count)],
            "merchant_category_code": categories["merchant_category_code"].to_numpy(),
            "merchant_category": categories["merchant_category"].to_numpy(),
            "channel": channel,
            "city": locations["city"],
            "state": locations["state"],
            "latitude": locations["latitude"].round(6),
            "longitude": locations["longitude"].round(6),
            "typical_ticket_inr": categories["typical_ticket"].to_numpy(),
            "merchant_risk_score": risk_score.round(4),
            "risk_tier": pd.cut(risk_score, bins=[-np.inf, 0.25, 0.55, np.inf], labels=["low", "medium", "high"]).astype("string"),
            "onboarded_at": onboarding_dates,
            "merchant_status": rng.choice(["active", "suspended", "closed"], count, p=[0.985, 0.005, 0.01]),
        }
    )


def validate_entities(
    customers: pd.DataFrame,
    accounts: pd.DataFrame,
    devices: pd.DataFrame,
    merchants: pd.DataFrame,
) -> None:
    tables_and_keys = [
        ("customers", customers, "customer_id"),
        ("accounts", accounts, "account_id"),
        ("devices", devices, "device_id"),
        ("merchants", merchants, "merchant_id"),
    ]
    for table_name, frame, key in tables_and_keys:
        if frame.empty:
            raise ValueError(f"{table_name} is empty")
        if frame[key].isna().any() or not frame[key].is_unique:
            raise ValueError(f"{table_name}.{key} must be non-null and unique")

    customer_keys = set(customers["customer_id"])
    if not set(accounts["customer_id"]).issubset(customer_keys):
        raise ValueError("accounts contains an unknown customer_id")
    if not set(devices["customer_id"]).issubset(customer_keys):
        raise ValueError("devices contains an unknown customer_id")
    if accounts.groupby("customer_id")["is_primary"].sum().ne(1).any():
        raise ValueError("Every customer must have exactly one primary account")
    if devices.groupby("customer_id")["is_primary"].sum().ne(1).any():
        raise ValueError("Every customer must have exactly one primary device")
    if not customers["annual_income_inr"].between(60_000, 15_000_000).all():
        raise ValueError("Customer income is outside configured bounds")
    if not merchants["merchant_risk_score"].between(0, 1).all():
        raise ValueError("Merchant risk score must be between 0 and 1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customers", type=int, default=NUM_CUSTOMERS)
    parser.add_argument("--merchants", type=int, default=NUM_MERCHANTS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/reference"),
        help="Output directory relative to the current working directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.customers <= 0 or args.merchants <= 0:
        raise ValueError("Entity counts must be positive")

    rng = np.random.default_rng(args.seed)
    fake = Faker("en_IN")
    Faker.seed(args.seed)

    print(f"Generating {args.customers:,} customers...")
    customers = generate_customers(rng, fake, args.customers)
    accounts = generate_accounts(rng, customers)
    devices = generate_devices(rng, customers)
    print(f"Generating {args.merchants:,} merchants...")
    merchants = generate_merchants(rng, fake, args.merchants)

    validate_entities(customers, accounts, devices, merchants)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "customers": customers,
        "accounts": accounts,
        "devices": devices,
        "merchants": merchants,
    }
    for name, frame in outputs.items():
        path = args.output_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
        print(f"[SAVED] {path} | rows={len(frame):,} | columns={len(frame.columns)}")

    print("\nEntity generation completed successfully.")
    print(f"Accounts per customer: {len(accounts) / len(customers):.3f}")
    print(f"Devices per customer:  {len(devices) / len(customers):.3f}")
    print("Merchant risk tiers:")
    print(merchants["risk_tier"].value_counts().to_string())


if __name__ == "__main__":
    main()
