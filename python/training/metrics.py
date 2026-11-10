"""Numerically stable shared model metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss


def stable_log_loss(y_true, probability, epsilon: float = 1e-7) -> float:
    """Use an explicit clip so results do not depend on float32/float64 dtype."""
    clipped = np.clip(np.asarray(probability, dtype=float), epsilon, 1 - epsilon)
    return float(log_loss(y_true, clipped))
