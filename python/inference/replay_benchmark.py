"""Replay chronological transactions through the live scorer and report latency."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from python.common import ROOT, write_json
from python.inference.service import RiskService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2_000)
    args = parser.parse_args()
    events = pd.read_parquet(ROOT / "data/raw/transactions.parquet").sort_values(
        ["event_timestamp", "transaction_id"], kind="mergesort"
    ).head(args.rows)
    service = RiskService(audit_path=None)
    for event in events.to_dict("records"):
        service.score(event)
    first = events.iloc[0].to_dict()
    service.score(first)  # deliberate duplicate to verify idempotency
    report = service.metrics_snapshot()
    metadata_path = ROOT / "artifacts/model_metadata.json"
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as handle:
            report["model_version"] = json.load(handle).get("model_version", "unknown")
    write_json(report, ROOT / "reports/inference/replay_benchmark.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
