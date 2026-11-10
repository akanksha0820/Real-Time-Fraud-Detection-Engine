"""Generate model/data drift and fraud-operations monitoring outputs."""

import json
import numpy as np
import pandas as pd

from python.common import ROOT, write_json


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    edges = np.unique(reference.quantile(np.linspace(0, 1, bins + 1)).to_numpy())
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = pd.cut(reference, edges).value_counts(normalize=True, sort=False).clip(lower=1e-6)
    cur = pd.cut(current, edges).value_counts(normalize=True, sort=False).clip(lower=1e-6)
    return float(((cur - ref) * np.log(cur / ref)).sum())


def main() -> None:
    frame = pd.read_parquet(ROOT / "data/processed/transaction_features.parquet")
    frame["event_timestamp"] = pd.to_datetime(frame["event_timestamp"])
    midpoint = frame["event_timestamp"].quantile(0.75)
    reference, current = frame[frame.event_timestamp < midpoint], frame[frame.event_timestamp >= midpoint]
    monitored = ["amount_inr", "merchant_risk_score", "tx_count_1h", "distance_from_home_km"]
    psi = {column: population_stability_index(reference[column], current[column]) for column in monitored}
    report = {"reference_rows": len(reference), "current_rows": len(current),
              "reference_fraud_rate": float(reference.is_fraud.mean()), "current_fraud_rate": float(current.is_fraud.mean()),
              "feature_psi": psi, "drifted_features": [key for key, value in psi.items() if value >= 0.2],
              "scenario_counts": current.loc[current.is_fraud.eq(1), "fraud_scenario"].value_counts().to_dict()}
    write_json(report, ROOT / "reports/monitoring_report.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
