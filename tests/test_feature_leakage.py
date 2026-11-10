import pandas as pd
from python.features.build_features import prior_entity_count, prior_window_features

def test_current_transaction_is_excluded_from_rolling_window():
    frame = pd.DataFrame({"customer_id": ["C1", "C1"], "event_timestamp": pd.to_datetime(["2026-01-01 10:00", "2026-01-01 10:05"]), "amount_inr": [100., 200.]})
    features = prior_window_features(frame, {"10m": "10min"})
    assert features.loc[0, "tx_count_10m"] == 0
    assert features.loc[1, "tx_count_10m"] == 1
    assert features.loc[1, "amount_sum_10m"] == 100


def test_merchant_window_excludes_current_event():
    frame = pd.DataFrame({
        "merchant_id": ["M1", "M1", "M2"],
        "transaction_id": ["T1", "T2", "T3"],
        "event_timestamp": pd.to_datetime(["2026-01-01 10:00", "2026-01-01 10:30", "2026-01-01 10:20"]),
    })
    counts = prior_entity_count(frame, "merchant_id", "1h")
    assert counts.tolist() == [0, 1, 0]
