"""Shared probability prediction for calibrated and legacy model bundles."""

from __future__ import annotations

import numpy as np
import pandas as pd


def predict_fraud_probability(bundle: dict, features: pd.DataFrame) -> np.ndarray:
    raw = bundle["pipeline"].predict_proba(features[bundle["features"]])[:, 1]
    calibrator = bundle.get("calibrator")
    if calibrator is None:
        return raw
    return np.asarray(calibrator.predict(raw), dtype=float)
