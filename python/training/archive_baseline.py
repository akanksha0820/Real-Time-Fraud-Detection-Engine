"""Create an immutable, checksummed snapshot of a trained model and its reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from python.common import ROOT, write_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="baseline_v1")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        raise ValueError("label may contain only letters, numbers, underscores, and hyphens")
    destination = ROOT / "artifacts" / "baselines" / args.label
    if destination.exists():
        raise FileExistsError(f"Baseline already exists: {destination}")
    destination.mkdir(parents=True)
    sources = [ROOT / "artifacts/model.joblib", ROOT / "artifacts/model_metadata.json"]
    optional = [ROOT / "reports/shap_feature_importance.csv"]
    report_dir = ROOT / "reports/model_evaluation"
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(f"Required baseline artifact is missing: {source}")
    copied: list[Path] = []
    for source in sources + [path for path in optional if path.exists()]:
        target = destination / source.name
        shutil.copy2(source, target)
        copied.append(target)
    if report_dir.exists():
        shutil.copytree(report_dir, destination / "model_evaluation")
        copied.extend(path for path in (destination / "model_evaluation").rglob("*") if path.is_file())
    manifest = {
        "label": args.label,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {str(path.relative_to(destination)): sha256(path) for path in copied},
    }
    write_json(manifest, destination / "manifest.json")
    print(f"[ARCHIVED] {destination.relative_to(ROOT)} | files={len(copied)}")


if __name__ == "__main__":
    main()
