"""Replay raw events and compare online features against the offline builder."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from python.common import ROOT, write_json
from python.features.build_features import build_features
from python.inference.feature_engine import OnlineFeatureEngine
from python.inference.state import InMemoryStateStore
from python.training.train_model import FEATURES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    raw = pd.read_parquet(ROOT / "data/raw/transactions.parquet").sort_values(
        ["event_timestamp", "transaction_id"], kind="mergesort"
    ).head(args.rows).reset_index(drop=True)
    offline = build_features(raw).set_index("transaction_id")
    engine = OnlineFeatureEngine(InMemoryStateStore())
    online_rows = []
    for event in raw.to_dict("records"):
        online_rows.append({"transaction_id": event["transaction_id"], **engine.compute_and_commit(event)})
    online = pd.DataFrame(online_rows).set_index("transaction_id")
    numeric = [column for column in FEATURES if column in online and pd.api.types.is_numeric_dtype(offline[column])]
    categorical = [column for column in FEATURES if column not in numeric]
    differences = {}
    for column in numeric:
        differences[column] = float(np.nanmax(np.abs(
            offline.loc[online.index, column].to_numpy(float) - online[column].to_numpy(float)
        )))
    category_mismatches = {
        column: int((offline.loc[online.index, column].astype(str).to_numpy() != online[column].astype(str).to_numpy()).sum())
        for column in categorical
    }
    failed_numeric = {key: value for key, value in differences.items() if value > args.tolerance}
    failed_categories = {key: value for key, value in category_mismatches.items() if value > 0}
    report = {"rows": len(raw), "tolerance": args.tolerance, "status": "PASS" if not failed_numeric and not failed_categories else "FAIL",
              "maximum_absolute_difference": differences, "category_mismatches": category_mismatches,
              "failed_numeric": failed_numeric, "failed_categories": failed_categories}
    write_json(report, ROOT / "reports/inference/feature_parity.json")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise AssertionError("Offline/online feature parity failed")


if __name__ == "__main__":
    main()
