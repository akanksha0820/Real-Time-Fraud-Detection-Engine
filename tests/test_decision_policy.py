import pandas as pd

from python.decision_engine.policy import apply_policy_decisions, decide


def test_allow_review_block():
    assert decide(.10, {}, .35, .75)["decision"] == "ALLOW"
    assert decide(.50, {}, .35, .75)["decision"] == "REVIEW"

    # High model probability without supporting evidence
    # is downgraded from BLOCK to REVIEW.
    assert decide(.90, {}, .35, .75)["decision"] == "REVIEW"


def test_reason_codes():
    result = decide(
        .9,
        {
            "tx_count_10m": 7,
            "is_new_device": 1,
        },
        .35,
        .75,
    )

    assert "ABNORMAL_TRANSACTION_VELOCITY" in result["reasons"]
    assert "NEW_DEVICE" in result["reasons"]


def test_two_signal_guardrail_downgrades_unsupported_block():
    frame = pd.DataFrame(
        {
            "tx_count_10m": [0, 6],
            "travel_speed_kmh": [0, 0],
            "distance_from_home_km": [0, 0],
            "customer_device_prior_uses": [0, 0],
            "amount_to_typical_ratio": [1, 5],
            "merchant_risk_score": [.1, .1],
            "merchant_tx_count_1h": [0, 0],
            "merchant_velocity_ratio": [0, 0],
            "device_global_prior_uses": [0, 0],
        }
    )

    decisions = apply_policy_decisions(
        [.99, .99],
        frame,
        .5,
        .95,
        {
            "minimum_block_signals": 2,
        },
    )

    assert decisions.tolist() == ["REVIEW", "BLOCK"]


def test_single_strong_signal_promotes_allow_to_review():
    result = decide(
        .01,
        {
            "travel_speed_kmh": 1500,
            "distance_from_home_km": 100,
            "tx_count_10m": 0,
            "customer_device_prior_uses": 5,
            "amount_to_typical_ratio": 1,
            "merchant_risk_score": .1,
            "merchant_tx_count_1h": 0,
            "merchant_velocity_ratio": 0,
            "device_global_prior_uses": 5,
        },
        .35,
        .75,
        {
            "minimum_review_signals": 1,
            "minimum_block_signals": 2,
        },
    )

    assert result["decision"] == "REVIEW"


def test_multiple_strong_signals_can_block_low_model_score():
    result = decide(
        .01,
        {
            "travel_speed_kmh": 20000,
            "distance_from_home_km": 2000,
            "tx_count_10m": 6,
            "customer_device_prior_uses": 7,
            "amount_to_typical_ratio": 100,
            "merchant_risk_score": .1,
            "merchant_tx_count_1h": 7,
            "merchant_velocity_ratio": 24,
            "device_global_prior_uses": 7,
        },
        .35,
        .75,
        {
            "minimum_review_signals": 1,
            "minimum_block_signals": 2,
        },
    )

    assert result["decision"] == "BLOCK"
    assert "IMPOSSIBLE_TRAVEL" in result["reasons"]
    assert "ABNORMAL_TRANSACTION_VELOCITY" in result["reasons"]


def test_high_probability_blocks_with_supporting_evidence():
    features = {
        "tx_count_10m": 6,
        "travel_speed_kmh": 1200,
        "distance_from_home_km": 1000,
        "customer_device_prior_uses": 5,
        "amount_to_typical_ratio": 1,
        "merchant_risk_score": .1,
        "merchant_tx_count_1h": 0,
        "merchant_velocity_ratio": 0,
        "device_global_prior_uses": 5,
    }

    result = decide(
        .90,
        features,
        .35,
        .75,
        {
            "minimum_review_signals": 1,
            "minimum_block_signals": 2,
        },
    )

    assert result["decision"] == "BLOCK"