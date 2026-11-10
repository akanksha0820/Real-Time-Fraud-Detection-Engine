"""Run the reproducible offline build from entities through monitoring."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from python.common import ROOT, ensure_project_directories


def run(module: str, *arguments: str) -> None:
    command = [sys.executable, "-m", module, *arguments]
    print(f"\n[RUNNING] {' '.join(command[1:])}")
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use 50k transactions for a fast end-to-end check")
    parser.add_argument("--regenerate-entities", action="store_true")
    args = parser.parse_args()
    ensure_project_directories()
    references = [ROOT / "data/reference" / f"{name}.parquet" for name in ["customers", "accounts", "devices", "merchants"]]
    if args.regenerate_entities or not all(path.exists() for path in references):
        entity_args = ["--customers", "5000", "--merchants", "1000"] if args.quick else []
        run("python.data_generation.generate_entities", *entity_args)
    run("python.data_generation.generate_profiles")
    run("python.data_generation.generate_transactions", *(["--quick"] if args.quick else []))
    run("python.preprocessing.validate_data")
    run("python.features.build_features")
    run("python.training.train_model")
    run("python.monitoring.generate_report")
    print("\n[SUCCESS] Fraud risk engine pipeline completed.")


if __name__ == "__main__":
    main()
