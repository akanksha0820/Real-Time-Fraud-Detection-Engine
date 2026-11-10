
"""Train a time-split XGBoost fraud model and optimize decision thresholds."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from python.common import ROOT, load_settings, write_json
from python.decision_engine.policy import (
    apply_policy_decisions,
    evidence_count_frame,
)
from python.training.metrics import stable_log_loss


# ============================================================
# FEATURES
# ============================================================

NUMERIC_FEATURES = [
    "amount_inr",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_night",
    "merchant_risk_score",
    "amount_to_typical_ratio",
    "amount_to_income_ratio",
    "distance_from_home_km",
    "seconds_since_previous_tx",
    "distance_from_previous_km",
    "travel_speed_kmh",
    "customer_device_prior_uses",
    "is_new_device",
    "customer_merchant_prior_uses",
    "is_new_merchant",
    "tx_count_10m",
    "amount_sum_10m",
    "tx_count_1h",
    "amount_sum_1h",
    "tx_count_24h",
    "amount_sum_24h",
    "device_global_prior_uses",
    "merchant_tx_count_1h",
    "merchant_tx_count_24h",
    "merchant_velocity_ratio",
]

CATEGORICAL_FEATURES = [
    "payment_channel",
    "risk_tier",
    "merchant_category",
    "customer_segment",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ============================================================
# CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_by_time(
    frame: pd.DataFrame,
    validation_start: str,
    test_start: str,
):
    timestamp = pd.to_datetime(frame["event_timestamp"])

    validation_boundary = pd.Timestamp(validation_start)
    test_boundary = pd.Timestamp(test_start)

    if validation_boundary >= test_boundary:
        raise ValueError(
            "validation_start must be earlier than test_start"
        )

    train = frame[timestamp < validation_boundary].copy()

    validation = frame[
        (timestamp >= validation_boundary)
        & (timestamp < test_boundary)
    ].copy()

    test = frame[timestamp >= test_boundary].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError(
            "A chronological split is empty. "
            "Check the configured validation_start and test_start."
        )

    return train, validation, test


# ============================================================
# FRAUD DISTRIBUTION DIAGNOSTICS
# ============================================================

def print_data_distribution(train, validation, test):
    print("\n" + "=" * 60)
    print("DATASET DISTRIBUTION")
    print("=" * 60)

    for name, subset in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test),
    ]:
        fraud_count = int(subset["is_fraud"].sum())
        fraud_rate = float(subset["is_fraud"].mean())

        print(
            f"{name:15s} | "
            f"Rows: {len(subset):8,d} | "
            f"Fraud: {fraud_count:6,d} | "
            f"Rate: {fraud_rate:.3%}"
        )

    print("=" * 60)


# ============================================================
# SAFE CALIBRATION / POLICY VALIDATION SPLIT
# ============================================================

def split_calibration_policy(
    validation: pd.DataFrame,
    calibration_fraction: float = 0.5,
    min_fraud_events: int = 25,
):
    """
    Split validation data chronologically into:
      1. Calibration period
      2. Policy optimization period

    Both periods must contain at least min_fraud_events
    positive fraud examples.

    The split is adjusted to the nearest feasible chronological
    boundary without shuffling data.
    """

    validation = validation.sort_values(
        ["event_timestamp", "transaction_id"]
    ).reset_index(drop=True)

    if not 0 < calibration_fraction < 1:
        raise ValueError(
            "calibration_fraction must be between 0 and 1"
        )

    labels = validation["is_fraud"].astype(int).to_numpy()

    total_fraud = int(labels.sum())
    required_fraud = 2 * min_fraud_events

    if total_fraud < required_fraud:
        raise ValueError(
            f"Validation contains only {total_fraud} fraud events. "
            f"At least {required_fraud} are required for calibration "
            "and policy optimization. Increase the number of "
            "generated transactions or revise the configured "
            "chronological validation boundaries."
        )

    cumulative_fraud = np.cumsum(labels)

    valid_cuts = np.flatnonzero(
        (cumulative_fraud[:-1] >= min_fraud_events)
        & (
            total_fraud - cumulative_fraud[:-1]
            >= min_fraud_events
        )
    ) + 1

    # Avoid splitting transactions sharing the same timestamp.
    timestamps = pd.to_datetime(
        validation["event_timestamp"]
    ).to_numpy()

    valid_cuts = valid_cuts[
        timestamps[valid_cuts - 1] < timestamps[valid_cuts]
    ]

    if len(valid_cuts) == 0:
        raise ValueError(
            "No chronological calibration split satisfies the "
            "minimum fraud-event requirement. Increase the dataset "
            "size or adjust the time boundaries."
        )

    target_cut = int(
        len(validation) * calibration_fraction
    )

    calibration_cut = int(
        valid_cuts[
            np.argmin(np.abs(valid_cuts - target_cut))
        ]
    )

    calibration = validation.iloc[
        :calibration_cut
    ].copy()

    policy_validation = validation.iloc[
        calibration_cut:
    ].copy()

    print("\n" + "=" * 60)
    print("CALIBRATION AND POLICY SPLIT")
    print("=" * 60)

    print(
        f"Calibration rows: {len(calibration):,}"
    )
    print(
        "Calibration fraud events:",
        int(calibration["is_fraud"].sum()),
    )

    print(
        f"Policy validation rows: {len(policy_validation):,}"
    )
    print(
        "Policy validation fraud events:",
        int(policy_validation["is_fraud"].sum()),
    )

    print("=" * 60)

    return calibration, policy_validation


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def classification_metrics(
    y_true,
    probability,
    decisions: pd.Series,
) -> dict:

    prediction = decisions.eq("BLOCK")

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            prediction,
            average="binary",
            zero_division=0,
        )
    )

    return {
        "pr_auc": float(
            average_precision_score(y_true, probability)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probability)
        ),
        "brier_score": float(
            brier_score_loss(y_true, probability)
        ),
        "log_loss": stable_log_loss(
            y_true, probability
        ),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


# ============================================================
# DECISION THRESHOLD COST
# ============================================================

def threshold_cost(
    y_true: np.ndarray,
    probability: np.ndarray,
    review: float,
    block: float,
    amounts: np.ndarray,
    evidence_counts: np.ndarray,
    costs: dict,
) -> float:

    decision = np.where(
        probability >= block,
        "BLOCK",
        np.where(
            probability >= review,
            "REVIEW",
            "ALLOW",
        ),
    )

    unsupported = (
        (decision == "BLOCK")
        & (
            evidence_counts
            < int(costs.get("minimum_block_signals", 0))
        )
    )

    decision[unsupported] = "REVIEW"

    fraud = y_true.astype(bool)

    missed_loss = (
        amounts[
            (decision == "ALLOW") & fraud
        ].sum()
        * costs["fraud_loss_multiplier"]
    )

    review_cost = (
        (decision == "REVIEW").sum()
        * costs["manual_review_cost"]
    )

    reviewed_fraud_loss = (
        amounts[
            (decision == "REVIEW") & fraud
        ].sum()
        * (
            1 - costs.get(
                "review_detection_rate", 1.0
            )
        )
        * costs["fraud_loss_multiplier"]
    )

    legitimate_review_friction = (
        (
            (decision == "REVIEW")
            & ~fraud
        ).sum()
        * costs.get(
            "review_customer_friction_cost", 0.0
        )
    )

    false_block = (
        (decision == "BLOCK")
        & ~fraud
    )

    false_decline = (
        false_block.sum()
        * costs["false_decline_cost"]
        + amounts[false_block].sum()
        * costs.get(
            "false_decline_amount_pct", 0.0
        )
    )

    return float(
        missed_loss
        + reviewed_fraud_loss
        + review_cost
        + legitimate_review_friction
        + false_decline
    )


# ============================================================
# THRESHOLD CANDIDATES
# ============================================================

def _threshold_candidates(
    probability: np.ndarray,
    quantile_start: float,
) -> np.ndarray:

    quantiles = np.linspace(
        quantile_start,
        0.9999,
        28,
    )

    empirical = np.quantile(
        probability,
        quantiles,
    )

    fixed = np.array([
        0.005,
        0.01,
        0.02,
        0.05,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95,
        0.99,
    ])

    return np.unique(
        np.clip(
            np.concatenate([
                empirical,
                fixed,
            ]),
            0.001,
            0.999,
        )
    )


# ============================================================
# OPTIMIZE DECISION THRESHOLDS
# ============================================================

def optimize_thresholds(
    y_true,
    probability,
    amounts,
    features,
    costs,
) -> tuple[float, float, float]:

    best = (
        np.inf,
        0.35,
        0.75,
    )

    evidence_counts = (
        evidence_count_frame(
            features
        ).to_numpy()
    )

    review_candidates = (
        _threshold_candidates(
            probability,
            0.85,
        )
    )

    block_candidates = (
        _threshold_candidates(
            probability,
            0.95,
        )
    )

    for review in review_candidates:

        for block in block_candidates[
            block_candidates > review
        ]:

            cost = threshold_cost(
                np.asarray(y_true),
                probability,
                float(review),
                float(block),
                np.asarray(amounts),
                evidence_counts,
                costs,
            )

            if cost < best[0]:
                best = (
                    cost,
                    float(review),
                    float(block),
                )

    return best[1], best[2], best[0]


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main() -> None:

    print("\n[START] Fraud Risk Engine Model Training")

    settings = load_settings()

    data_path = (
        ROOT
        / "data/processed/transaction_features.parquet"
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {data_path}. "
            "Run feature generation first."
        )

    frame = pd.read_parquet(data_path)

    print(
        f"[LOADED] {len(frame):,} transactions | "
        f"{len(frame.columns)} columns"
    )

    required_columns = (
        FEATURES
        + [
            "event_timestamp",
            "transaction_id",
            "is_fraud",
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # STEP 1: Chronological split
    # --------------------------------------------------------

    train, validation, test = split_by_time(
        frame,
        settings["model"]["validation_start"],
        settings["model"]["test_start"],
    )

    print_data_distribution(
        train,
        validation,
        test,
    )

    train_positive = int(
        train["is_fraud"].sum()
    )

    train_negative = (
        len(train) - train_positive
    )

    if train_positive == 0 or train_negative == 0:
        raise ValueError(
            "Training requires both fraud and legitimate "
            "transactions. Increase the dataset size or "
            "adjust the chronological split boundaries."
        )

    if test["is_fraud"].nunique() < 2:
        raise ValueError(
            "Test data must contain both fraud and legitimate "
            "transactions to calculate ROC-AUC reliably."
        )

    # Validate the calibration split before fitting XGBoost.
    calibration, policy_validation = split_calibration_policy(
        validation,
        calibration_fraction=settings["model"].get(
            "calibration_fraction",
            0.5,
        ),
        min_fraud_events=25,
    )

    # --------------------------------------------------------
    # STEP 2: Handle class imbalance
    # --------------------------------------------------------

    scale = (
        train_negative
        / train_positive
    )

    print(
        f"\n[CLASS IMBALANCE] scale_pos_weight={scale:.3f}"
    )

    # --------------------------------------------------------
    # STEP 3: Feature preprocessing
    # --------------------------------------------------------

    transformer = ColumnTransformer([
        (
            "numeric",
            "passthrough",
            NUMERIC_FEATURES,
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
            ),
            CATEGORICAL_FEATURES,
        ),
    ])

    # --------------------------------------------------------
    # STEP 4: XGBoost classifier
    # --------------------------------------------------------

    classifier = XGBClassifier(
        n_estimators=450,
        max_depth=7,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=4,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale,
        random_state=settings["model"]["random_state"],
        n_jobs=-1,
    )

    pipeline = Pipeline([
        (
            "preprocessor",
            transformer,
        ),
        (
            "classifier",
            classifier,
        ),
    ])

    print("\n[TRAINING] Fitting XGBoost model...")

    pipeline.fit(
        train[FEATURES],
        train["is_fraud"],
    )

    print("[SUCCESS] XGBoost training completed")

    # --------------------------------------------------------
    # STEP 5: Isotonic probability calibration
    # --------------------------------------------------------

    print("\n[CALIBRATION] Calibrating probabilities...")

    calibration_raw_probability = (
        pipeline.predict_proba(
            calibration[FEATURES]
        )[:, 1]
    )

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    calibrator.fit(
        calibration_raw_probability,
        calibration["is_fraud"].to_numpy(),
    )

    print("[SUCCESS] Probability calibration completed")

    # --------------------------------------------------------
    # STEP 6: Policy validation
    # --------------------------------------------------------

    policy_raw_probability = (
        pipeline.predict_proba(
            policy_validation[FEATURES]
        )[:, 1]
    )

    policy_probability = calibrator.predict(
        policy_raw_probability
    )

    policy = {
        "minimum_block_signals": settings[
            "decision"
        ].get(
            "minimum_block_signals",
            1,
        )
    }

    print("\n[OPTIMIZATION] Finding decision thresholds...")

    review, block, validation_cost = optimize_thresholds(
        policy_validation["is_fraud"],
        policy_probability,
        policy_validation["amount_inr"],
        policy_validation,
        settings["decision"],
    )

    print(
        f"[THRESHOLDS] Review={review:.4f}, "
        f"Block={block:.4f}"
    )

    # --------------------------------------------------------
    # STEP 7: Final test evaluation
    # --------------------------------------------------------

    print("\n[EVALUATION] Evaluating test dataset...")

    test_raw_probability = (
        pipeline.predict_proba(
            test[FEATURES]
        )[:, 1]
    )

    test_probability = calibrator.predict(
        test_raw_probability
    )

    test_decisions = apply_policy_decisions(
        test_probability,
        test,
        review,
        block,
        policy,
    )

    metrics = classification_metrics(
        test["is_fraud"],
        test_probability,
        test_decisions,
    )

    metrics.update({
        "review_threshold": review,
        "block_threshold": block,
        "validation_cost_inr": validation_cost,
        "raw_brier_score": float(
            brier_score_loss(
                test["is_fraud"],
                test_raw_probability,
            )
        ),
        "raw_log_loss": stable_log_loss(
            test["is_fraud"],
            test_raw_probability,
        ),
        "train_rows": len(train),
        "calibration_rows": len(calibration),
        "policy_validation_rows": len(policy_validation),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "train_fraud_events": train_positive,
        "calibration_fraud_events": int(
            calibration["is_fraud"].sum()
        ),
        "policy_validation_fraud_events": int(
            policy_validation["is_fraud"].sum()
        ),
        "test_fraud_events": int(
            test["is_fraud"].sum()
        ),
        "train_fraud_rate": float(
            train["is_fraud"].mean()
        ),
        "test_fraud_rate": float(
            test["is_fraud"].mean()
        ),
        "simulation_version": settings[
            "simulation"
        ].get(
            "version",
            "unknown",
        ),
        "model_version": settings[
            "model"
        ].get(
            "version",
            "unknown",
        ),
        "minimum_block_signals": policy[
            "minimum_block_signals"
        ],
    })

    # --------------------------------------------------------
    # STEP 8: Save model and metadata
    # --------------------------------------------------------

    bundle = {
        "pipeline": pipeline,
        "calibrator": calibrator,
        "features": FEATURES,
        "review_threshold": review,
        "block_threshold": block,
        "policy": policy,
    }

    artifact = (
        ROOT
        / "artifacts/model.joblib"
    )

    artifact.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        bundle,
        artifact,
    )

    write_json(
        metrics,
        ROOT / "artifacts/model_metadata.json",
    )

    print("\n" + "=" * 60)
    print("FINAL MODEL METRICS")
    print("=" * 60)

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    print(
        f"\n[SAVED] {artifact.relative_to(ROOT)}"
    )

    print(
        "[SAVED] artifacts/model_metadata.json"
    )

    print("\n[SUCCESS] Model training pipeline completed")


if __name__ == "__main__":
    main()