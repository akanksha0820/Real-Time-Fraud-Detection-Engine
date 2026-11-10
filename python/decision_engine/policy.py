"""Translate calibrated risk into an operational action and reason codes."""

from __future__ import annotations

import numpy as np
import pandas as pd


def evidence_count(features: dict) -> int:
    """Count independent risk families rather than correlated raw features."""
    signals = [
        features.get("tx_count_10m", 0) >= 5,

        features.get("travel_speed_kmh", 0) >= 900
        or features.get("distance_from_home_km", 0) >= 800,

        features.get("customer_device_prior_uses", 1) == 0,

        features.get("amount_to_typical_ratio", 0) >= 4,

        features.get("merchant_risk_score", 0) >= 0.55
        or (
            features.get("merchant_tx_count_1h", 0) >= 5
            and features.get("merchant_velocity_ratio", 0) >= 4
        ),

        features.get("device_global_prior_uses", 0) >= 5
        and features.get("customer_device_prior_uses", 1) == 0,
    ]
    return int(sum(signals))


def evidence_count_frame(frame: pd.DataFrame) -> pd.Series:
    """Vectorized equivalent of evidence_count used for policy optimization."""
    velocity = frame["tx_count_10m"].ge(5)

    geography = (
        frame["travel_speed_kmh"].ge(900)
        | frame["distance_from_home_km"].ge(800)
    )

    novel_device = frame["customer_device_prior_uses"].eq(0)

    unusual_amount = frame["amount_to_typical_ratio"].ge(4)

    merchant = (
        frame["merchant_risk_score"].ge(.55)
        | (
            frame["merchant_tx_count_1h"].ge(5)
            & frame["merchant_velocity_ratio"].ge(4)
        )
    )

    shared_device = (
        frame["device_global_prior_uses"].ge(5)
        & novel_device
    )

    return sum(
        signal.astype("int8")
        for signal in [
            velocity,
            geography,
            novel_device,
            unusual_amount,
            merchant,
            shared_device,
        ]
    )


def apply_policy_decisions(
    probability: np.ndarray | pd.Series,
    features: pd.DataFrame,
    review_threshold: float,
    block_threshold: float,
    policy: dict | None = None,
) -> pd.Series:
    """
    Combine model thresholds with deterministic risk evidence.

    Rules:
    - model score determines the baseline action
    - strong evidence can promote ALLOW -> REVIEW
    - multiple independent signals can promote to BLOCK
    - BLOCK still requires the configured minimum evidence
    """
    policy = policy or {}

    minimum_block_signals = int(
        policy.get("minimum_block_signals", 2)
    )

    minimum_review_signals = int(
        policy.get("minimum_review_signals", 1)
    )

    probability = pd.Series(
        np.asarray(probability),
        index=features.index
    )

    actions = pd.Series(
        np.select(
            [
                probability.ge(block_threshold),
                probability.ge(review_threshold),
            ],
            [
                "BLOCK",
                "REVIEW",
            ],
            default="ALLOW",
        ),
        index=features.index,
        dtype="string",
    )

    evidence = evidence_count_frame(features)

    # Promote low-scoring transactions when deterministic evidence is strong.
    promote_review = (
        actions.eq("ALLOW")
        & evidence.ge(minimum_review_signals)
    )
    actions.loc[promote_review] = "REVIEW"

    # Multiple independent high-risk signals can justify a block even when
    # the model score alone is below the block threshold.
    promote_block = evidence.ge(minimum_block_signals)
    actions.loc[promote_block] = "BLOCK"

    # Final safety guardrail: never leave an unsupported BLOCK.
    unsupported_block = (
        actions.eq("BLOCK")
        & evidence.lt(minimum_block_signals)
    )
    actions.loc[unsupported_block] = "REVIEW"

    return actions


def reason_codes(
    features: dict,
    probability: float,
) -> list[str]:
    rules = [
        (
            features.get("tx_count_10m", 0) >= 5,
            "ABNORMAL_TRANSACTION_VELOCITY",
        ),
        (
            features.get("travel_speed_kmh", 0) >= 900,
            "IMPOSSIBLE_TRAVEL",
        ),
        (
            features.get("distance_from_home_km", 0) >= 800,
            "GEOGRAPHIC_ANOMALY",
        ),
        (
            features.get("is_new_device", 0) == 1,
            "NEW_DEVICE",
        ),
        (
            features.get("amount_to_typical_ratio", 0) >= 4,
            "UNUSUAL_TRANSACTION_AMOUNT",
        ),
        (
            features.get("merchant_risk_score", 0) >= 0.55,
            "HIGH_RISK_MERCHANT",
        ),
        (
            features.get("is_night", 0) == 1,
            "UNUSUAL_HOUR",
        ),
    ]

    reasons = [
        code
        for condition, code in rules
        if condition
    ]

    return (
        reasons
        or (
            ["MODEL_PATTERN_RISK"]
            if probability >= 0.35
            else []
        )
    )[:5]


def decide(
    probability: float,
    features: dict,
    review_threshold: float,
    block_threshold: float,
    policy: dict | None = None,
) -> dict:
    if not 0 <= probability <= 1:
        raise ValueError(
            "probability must be between 0 and 1"
        )

    policy = policy or {}

    minimum_block_signals = int(
        policy.get("minimum_block_signals", 2)
    )

    minimum_review_signals = int(
        policy.get("minimum_review_signals", 1)
    )

    evidence = evidence_count(features)

    # Baseline model decision.
    if probability >= block_threshold:
        action = "BLOCK"
    elif probability >= review_threshold:
        action = "REVIEW"
    else:
        action = "ALLOW"

    # Deterministic evidence can escalate a low model score.
    if (
        action == "ALLOW"
        and evidence >= minimum_review_signals
    ):
        action = "REVIEW"

    if evidence >= minimum_block_signals:
        action = "BLOCK"

    # Final protection against unsupported blocks.
    if (
        action == "BLOCK"
        and evidence < minimum_block_signals
    ):
        action = "REVIEW"

    return {
        "fraud_probability": round(
            float(probability),
            6,
        ),
        "decision": action,
        "reasons": reason_codes(
            features,
            probability,
        ),
    }