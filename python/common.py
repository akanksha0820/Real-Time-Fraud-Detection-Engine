"""Shared configuration, paths, and serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_settings(path: Path | None = None) -> dict[str, Any]:
    config_path = path or ROOT / "config" / "settings.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_project_directories() -> None:
    for relative in ["data/raw", "data/processed", "data/reference", "artifacts", "reports"]:
        (ROOT / relative).mkdir(parents=True, exist_ok=True)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

