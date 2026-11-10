"""Audit the trained fraud model on the untouched chronological test period.

This module never changes the model or thresholds. It produces scenario,
economic, subgroup, calibration, score-band, and legitimate hard-negative
reports under ``reports/model_evaluation``.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from python.common import ROOT, load_settings, write_json
from python.decision_engine.policy import apply_policy_decisions
from python.inference.predict import predict_fraud_probability
from python.training.metrics import stable_log_loss


REPORT_DIR = ROOT / "reports" / "model_evaluation"


def assign_decision(probability: pd.Series, review: float, block: float) -> pd.Series:
    """Apply frozen operational thresholds to model probabilities."""
    return pd.Series(
        np.select(
            [probability.ge(block), probability.ge(review)],
            ["BLOCK", "REVIEW"],
            default="ALLOW",
        ),
        index=probability.index,
        dtype="string",
    )


def safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Vectorized division that returns zero for empty denominators."""
    return numerator.div(denominator.replace(0, np.nan)).fillna(0.0)


def scenario_report(scored: pd.DataFrame) -> pd.DataFrame:
    fraud = scored.loc[scored["is_fraud"].eq(1)].copy()
    report = (
        fraud.groupby("fraud_scenario", observed=True)
        .agg(
            fraud_transactions=("transaction_id", "size"),
            blocked=("decision", lambda values: values.eq("BLOCK").sum()),
            reviewed=("decision", lambda values: values.eq("REVIEW").sum()),
            allowed=("decision", lambda values: values.eq("ALLOW").sum()),
            mean_score=("fraud_probability", "mean"),
            median_score=("fraud_probability", "median"),
            fraud_value_inr=("amount_inr", "sum"),
            blocked_value_inr=("amount_inr", lambda values: values[fraud.loc[values.index, "decision"].eq("BLOCK")].sum()),
            review_value_inr=("amount_inr", lambda values: values[fraud.loc[values.index, "decision"].eq("REVIEW")].sum()),
            missed_value_inr=("amount_inr", lambda values: values[fraud.loc[values.index, "decision"].eq("ALLOW")].sum()),
        )
        .reset_index()
    )
    report["block_recall"] = safe_rate(report["blocked"], report["fraud_transactions"])
    report["investigation_capture_rate"] = safe_rate(
        report["blocked"] + report["reviewed"], report["fraud_transactions"]
    )
    report["value_block_rate"] = safe_rate(report["blocked_value_inr"], report["fraud_value_inr"])
    return report.sort_values("block_recall").reset_index(drop=True)


def legitimate_subgroup_report(scored: pd.DataFrame, column: str) -> pd.DataFrame:
    legitimate = scored.loc[scored["is_fraud"].eq(0)].copy()
    report = (
        legitimate.groupby(column, observed=True)
        .agg(
            legitimate_transactions=("transaction_id", "size"),
            false_blocks=("decision", lambda values: values.eq("BLOCK").sum()),
            reviews=("decision", lambda values: values.eq("REVIEW").sum()),
            mean_score=("fraud_probability", "mean"),
            p95_score=("fraud_probability", lambda values: values.quantile(0.95)),
        )
        .reset_index()
    )
    report["false_block_rate"] = safe_rate(report["false_blocks"], report["legitimate_transactions"])
    report["review_rate"] = safe_rate(report["reviews"], report["legitimate_transactions"])
    return report.sort_values("false_block_rate", ascending=False).reset_index(drop=True)


def fraud_segment_report(scored: pd.DataFrame) -> pd.DataFrame:
    fraud = scored.loc[scored["is_fraud"].eq(1)].copy()
    report = (
        fraud.groupby("customer_segment", observed=True)
        .agg(
            fraud_transactions=("transaction_id", "size"),
            blocked=("decision", lambda values: values.eq("BLOCK").sum()),
            reviewed=("decision", lambda values: values.eq("REVIEW").sum()),
            fraud_value_inr=("amount_inr", "sum"),
            missed_value_inr=("amount_inr", lambda values: values[fraud.loc[values.index, "decision"].eq("ALLOW")].sum()),
        )
        .reset_index()
    )
    report["block_recall"] = safe_rate(report["blocked"], report["fraud_transactions"])
    report["investigation_capture_rate"] = safe_rate(
        report["blocked"] + report["reviewed"], report["fraud_transactions"]
    )
    return report.sort_values("block_recall").reset_index(drop=True)


def hard_negative_report(scored: pd.DataFrame) -> pd.DataFrame:
    legitimate = scored.loc[scored["is_fraud"].eq(0)].copy()
    definitions = {
        "new_device": legitimate["is_new_device"].eq(1),
        "large_vs_customer_baseline": legitimate["amount_to_typical_ratio"].ge(4),
        "late_night": legitimate["is_night"].eq(1),
        "far_from_home": legitimate["distance_from_home_km"].ge(500),
        "rapid_velocity": legitimate["tx_count_10m"].ge(5),
        "high_risk_merchant": legitimate["merchant_risk_score"].ge(0.55),
        "impossible_travel_lookalike": legitimate["travel_speed_kmh"].ge(900),
    }
    rows: list[dict] = []
    for name, mask in definitions.items():
        subset = legitimate.loc[mask]
        count = len(subset)
        rows.append(
            {
                "hard_negative_type": name,
                "transactions": count,
                "mean_score": float(subset["fraud_probability"].mean()) if count else 0.0,
                "p95_score": float(subset["fraud_probability"].quantile(0.95)) if count else 0.0,
                "reviews": int(subset["decision"].eq("REVIEW").sum()),
                "false_blocks": int(subset["decision"].eq("BLOCK").sum()),
                "review_rate": float(subset["decision"].eq("REVIEW").mean()) if count else 0.0,
                "false_block_rate": float(subset["decision"].eq("BLOCK").mean()) if count else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("false_block_rate", ascending=False).reset_index(drop=True)


def legitimate_pattern_report(scored: pd.DataFrame) -> pd.DataFrame:
    """Evaluate simulator-known legitimate behaviors without using them as model features."""
    legitimate = scored.loc[scored["is_fraud"].eq(0)].copy()
    if "legitimate_pattern" not in legitimate.columns:
        return pd.DataFrame()
    report = (
        legitimate.groupby("legitimate_pattern", observed=True)
        .agg(
            transactions=("transaction_id", "size"),
            reviews=("decision", lambda values: values.eq("REVIEW").sum()),
            false_blocks=("decision", lambda values: values.eq("BLOCK").sum()),
            mean_score=("fraud_probability", "mean"),
            p95_score=("fraud_probability", lambda values: values.quantile(.95)),
            transaction_value_inr=("amount_inr", "sum"),
        )
        .reset_index()
    )
    report["review_rate"] = safe_rate(report["reviews"], report["transactions"])
    report["false_block_rate"] = safe_rate(report["false_blocks"], report["transactions"])
    return report.sort_values("false_block_rate", ascending=False).reset_index(drop=True)


def score_band_report(scored: pd.DataFrame) -> pd.DataFrame:
    bands = pd.IntervalIndex.from_tuples(
        [(0.00, 0.10), (0.10, 0.25), (0.25, 0.45), (0.45, 0.60),
         (0.60, 0.75), (0.75, 0.90), (0.90, 0.95), (0.95, 1.00)],
        closed="right",
    )
    working = scored.copy()
    working["score_band"] = pd.cut(working["fraud_probability"], bands)
    report = (
        working.groupby("score_band", observed=False)
        .agg(
            transactions=("transaction_id", "size"),
            fraud_transactions=("is_fraud", "sum"),
            transaction_value_inr=("amount_inr", "sum"),
            fraud_value_inr=("amount_inr", lambda values: values[working.loc[values.index, "is_fraud"].eq(1)].sum()),
        )
        .reset_index()
    )
    report["observed_fraud_rate"] = safe_rate(report["fraud_transactions"], report["transactions"])
    report["score_band"] = report["score_band"].astype("string")
    return report


def overall_summary(scored: pd.DataFrame, review: float, block: float, costs: dict | None = None) -> dict:
    costs = costs or {}
    fraud = scored["is_fraud"].eq(1)
    blocked = scored["decision"].eq("BLOCK")
    reviewed = scored["decision"].eq("REVIEW")
    allowed = scored["decision"].eq("ALLOW")
    tp = int((fraud & blocked).sum())
    fp = int((~fraud & blocked).sum())
    fn = int((fraud & ~blocked).sum())
    reviewed_fraud = fraud & reviewed
    reviewed_legitimate = ~fraud & reviewed
    allowed_fraud_value = float(scored.loc[fraud & allowed, "amount_inr"].sum())
    review_fraud_value = float(scored.loc[reviewed_fraud, "amount_inr"].sum())
    false_block_value = float(scored.loc[~fraud & blocked, "amount_inr"].sum())
    review_detection_rate = float(costs.get("review_detection_rate", 1.0))
    residual_review_loss = review_fraud_value * (1 - review_detection_rate)
    review_operating_cost = int(reviewed.sum()) * float(costs.get("manual_review_cost", 0.0))
    review_friction_cost = int(reviewed_legitimate.sum()) * float(costs.get("review_customer_friction_cost", 0.0))
    false_decline_cost = int((~fraud & blocked).sum()) * float(costs.get("false_decline_cost", 0.0)) + (
        false_block_value * float(costs.get("false_decline_amount_pct", 0.0))
    )
    test_days = max((scored["event_timestamp"].max() - scored["event_timestamp"].min()).days + 1, 1)
    return {
        "test_rows": len(scored),
        "fraud_transactions": int(fraud.sum()),
        "fraud_rate": float(fraud.mean()),
        "review_threshold": review,
        "block_threshold": block,
        "pr_auc": float(average_precision_score(scored["is_fraud"], scored["fraud_probability"])),
        "roc_auc": float(roc_auc_score(scored["is_fraud"], scored["fraud_probability"])),
        "brier_score": float(brier_score_loss(scored["is_fraud"], scored["fraud_probability"])),
        "log_loss": stable_log_loss(scored["is_fraud"], scored["fraud_probability"]),
        "blocked": int(blocked.sum()),
        "reviewed": int(reviewed.sum()),
        "allowed": int(allowed.sum()),
        "true_positive_blocks": tp,
        "false_positive_blocks": fp,
        "fraud_not_blocked": fn,
        "block_precision": tp / max(tp + fp, 1),
        "block_recall": tp / max(int(fraud.sum()), 1),
        "investigation_capture_rate": float((fraud & (blocked | reviewed)).sum() / max(fraud.sum(), 1)),
        "review_fraud_transactions": int(reviewed_fraud.sum()),
        "review_legitimate_transactions": int(reviewed_legitimate.sum()),
        "review_precision": float(reviewed_fraud.sum() / max(reviewed.sum(), 1)),
        "review_queue_rate": float(reviewed.mean()),
        "average_daily_reviews": float(reviewed.sum() / test_days),
        "total_fraud_value_inr": float(scored.loc[fraud, "amount_inr"].sum()),
        "blocked_fraud_value_inr": float(scored.loc[fraud & blocked, "amount_inr"].sum()),
        "review_fraud_value_inr": review_fraud_value,
        "allowed_fraud_value_inr": allowed_fraud_value,
        "false_blocked_legitimate_value_inr": false_block_value,
        "expected_residual_review_loss_inr": residual_review_loss,
        "expected_total_fraud_loss_inr": allowed_fraud_value + residual_review_loss,
        "review_operating_cost_inr": review_operating_cost,
        "review_friction_cost_inr": review_friction_cost,
        "estimated_false_decline_cost_inr": false_decline_cost,
        "expected_total_policy_cost_inr": (
            allowed_fraud_value + residual_review_loss + review_operating_cost
            + review_friction_cost + false_decline_cost
        ),
    }


def save_report(frame: pd.DataFrame, filename: str) -> None:
    path = REPORT_DIR / filename
    frame.to_csv(path, index=False)
    print(f"[SAVED] {path.relative_to(ROOT)}")


def main() -> None:
    settings = load_settings()
    model_settings = settings["model"]
    bundle = joblib.load(ROOT / "artifacts/model.joblib")
    frame = pd.read_parquet(ROOT / "data/processed/transaction_features.parquet")
    frame["event_timestamp"] = pd.to_datetime(frame["event_timestamp"])
    test = frame.loc[frame["event_timestamp"].ge(pd.Timestamp(model_settings["test_start"]))].copy()
    if test.empty:
        raise ValueError("The configured chronological test period is empty")

    test["fraud_probability"] = predict_fraud_probability(bundle, test)
    review, block = float(bundle["review_threshold"]), float(bundle["block_threshold"])
    test["decision"] = apply_policy_decisions(
        test["fraud_probability"], test, review, block, bundle.get("policy", {})
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = scenario_report(test)
    legitimate_segments = legitimate_subgroup_report(test, "customer_segment")
    legitimate_categories = legitimate_subgroup_report(test, "merchant_category")
    fraud_segments = fraud_segment_report(test)
    hard_negatives = hard_negative_report(test)
    legitimate_patterns = legitimate_pattern_report(test)
    score_bands = score_band_report(test)
    summary = overall_summary(test, review, block, settings["decision"])

    save_report(scenarios, "scenario_performance.csv")
    save_report(legitimate_segments, "legitimate_customer_segments.csv")
    save_report(legitimate_categories, "legitimate_merchant_categories.csv")
    save_report(fraud_segments, "fraud_customer_segments.csv")
    save_report(hard_negatives, "hard_negative_performance.csv")
    if not legitimate_patterns.empty:
        save_report(legitimate_patterns, "legitimate_pattern_performance.csv")
    save_report(score_bands, "score_bands.csv")
    write_json(summary, REPORT_DIR / "evaluation_summary.json")
    test[["transaction_id", "event_timestamp", "fraud_scenario", "is_fraud", "amount_inr",
          "fraud_probability", "decision"]].to_parquet(
        ROOT / "data/processed/test_predictions.parquet", index=False
    )

    print("\n=== OVERALL TEST AUDIT ===")
    print(json.dumps(summary, indent=2))
    print("\n=== SCENARIO PERFORMANCE ===")
    print(scenarios[["fraud_scenario", "fraud_transactions", "block_recall",
                     "investigation_capture_rate", "missed_value_inr"]].to_string(index=False))
    print("\n=== LEGITIMATE HARD NEGATIVES ===")
    print(hard_negatives[["hard_negative_type", "transactions", "review_rate",
                          "false_block_rate"]].to_string(index=False))
    print("\nModel thresholds were evaluated as frozen; test data was not used for tuning.")


if __name__ == "__main__":
    main()
