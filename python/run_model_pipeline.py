"""Rebuild features and calibrated policy model without regenerating raw data."""

from __future__ import annotations

import argparse
import subprocess
import sys

from python.common import ROOT


def run(module: str, *arguments: str) -> None:
    command = [sys.executable, "-m", module, *arguments]
    print(f"\n[RUNNING] {' '.join(command[1:])}")
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="hardened_v2")
    args = parser.parse_args()
    run("python.features.build_features")
    run("python.training.train_model")
    run("python.monitoring.generate_report")
    run("python.training.evaluate_model")
    run("python.training.compare_versions", "--baseline", args.baseline)
    print("\n[SUCCESS] Calibrated model and policy pipeline completed.")


if __name__ == "__main__":
    main()
