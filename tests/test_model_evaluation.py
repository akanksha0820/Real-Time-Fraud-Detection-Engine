import pandas as pd
import numpy as np

from python.training.evaluate_model import assign_decision, hard_negative_report
from python.training.train_model import FEATURES
from python.training.metrics import stable_log_loss


def test_assign_decision_uses_frozen_boundaries():
    probability = pd.Series([0.10, 0.45, 0.949, 0.95])
    assert assign_decision(probability, 0.45, 0.95).tolist() == [
        "ALLOW", "REVIEW", "REVIEW", "BLOCK"
    ]


def test_simulator_provenance_is_not_a_model_feature():
    assert "is_fraud" not in FEATURES
    assert "fraud_scenario" not in FEATURES
    assert "legitimate_pattern" not in FEATURES
    assert "simulation_version" not in FEATURES


def test_hard_negative_report_counts_false_blocks():
    frame = pd.DataFrame(
        {
            "is_fraud": [0, 0],
            "is_new_device": [1, 0],
            "amount_to_typical_ratio": [1.0, 5.0],
            "is_night": [0, 0],
            "distance_from_home_km": [0.0, 0.0],
            "tx_count_10m": [0.0, 0.0],
            "merchant_risk_score": [0.1, 0.1],
            "travel_speed_kmh": [0.0, 0.0],
            "fraud_probability": [0.96, 0.20],
            "decision": ["BLOCK", "ALLOW"],
        }
    )
    report = hard_negative_report(frame).set_index("hard_negative_type")
    assert report.loc["new_device", "transactions"] == 1
    assert report.loc["new_device", "false_block_rate"] == 1.0
    assert report.loc["large_vs_customer_baseline", "false_block_rate"] == 0.0


def test_log_loss_is_dtype_stable():
    labels = np.array([0, 1])
    probability = np.array([0.0, 1.0])
    assert stable_log_loss(labels, probability.astype("float32")) == stable_log_loss(
        labels, probability.astype("float64")
    )
