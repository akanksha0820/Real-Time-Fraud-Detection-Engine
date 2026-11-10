import pandas as pd
from python.monitoring.production_performance import evaluate,expected_calibration_error

def test_monitoring_requires_matured_sample():
    frame=pd.DataFrame({"is_fraud":[0,1],"fraud_probability":[.1,.8],"decision":["ALLOW","BLOCK"]})
    metrics,alerts=evaluate(frame,minimum_labels=10)
    assert "INSUFFICIENT_MATURED_LABELS" in alerts
    assert "INSUFFICIENT_FRAUD_LABELS" in alerts
    assert metrics["metric_status"] == "PROVISIONAL"
    assert metrics["pr_auc"] == 1.0

def test_calibration_error_is_zero_for_perfect_probabilities():
    assert expected_calibration_error(pd.Series([0,1]),pd.Series([0.0,1.0])) == 0.0
