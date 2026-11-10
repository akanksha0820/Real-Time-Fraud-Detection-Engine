"""Compare an archived baseline evaluation with the current model evaluation."""

from __future__ import annotations

import argparse
import json
import re

import pandas as pd

from python.common import ROOT, write_json


METRICS = [
    "pr_auc", "roc_auc", "brier_score", "log_loss", "block_precision",
    "block_recall", "investigation_capture_rate", "allowed_fraud_value_inr",
    "false_positive_blocks", "false_blocked_legitimate_value_inr",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="baseline_v1")
    args = parser.parse_args()
    baseline_path = ROOT / "artifacts/baselines" / args.baseline / "model_evaluation/evaluation_summary.json"
    current_path = ROOT / "reports/model_evaluation/evaluation_summary.json"
    metadata_path = ROOT / "artifacts/model_metadata.json"
    with baseline_path.open(encoding="utf-8") as handle:
        baseline = json.load(handle)
    with current_path.open(encoding="utf-8") as handle:
        current = json.load(handle)
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    baseline_label = re.sub(r"[^A-Za-z0-9_]+", "_", args.baseline).strip("_")
    current_label = re.sub(
        r"[^A-Za-z0-9_]+", "_", metadata.get("model_version", "current_model")
    ).strip("_")
    rows = []
    for metric in METRICS:
        before, after = baseline.get(metric), current.get(metric)
        if before is not None and after is not None:
            rows.append({"metric": metric, baseline_label: before, current_label: after, "absolute_change": after - before})
    comparison = pd.DataFrame(rows)
    output = ROOT / "reports/model_evaluation/version_comparison.csv"
    comparison.to_csv(output, index=False)
    write_json({row["metric"]: row for row in rows}, ROOT / "reports/model_evaluation/version_comparison.json")
    print(comparison.to_string(index=False))
    print(f"[SAVED] {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
