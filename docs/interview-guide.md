# Interview Explanation Guide

## Thirty-second version

“I built a real-time fraud-risk platform around a million-event synthetic financial ecosystem. It combines leakage-safe batch and online features, calibrated XGBoost, economically optimized allow/review/block policy, Kafka, Redis, PostgreSQL, FastAPI, MLflow, delayed-label monitoring, and an analyst case dashboard. I validated offline/online feature parity, idempotent processing, chronological evaluation, and containerized operation.”

## Two-minute version

1. I generated stable customers, accounts, devices, merchants, and behavioral profiles.
2. I created normal activity plus six coherent fraud attacks and six legitimate hard negatives.
3. I built point-in-time features using only prior events and protected this with automated tests.
4. I trained XGBoost with chronological train/validation/test periods because random splitting would leak future behavior.
5. I calibrated probabilities with isotonic regression and selected decision thresholds on a separate policy-validation period.
6. I separated risk prediction from business policy, requiring two independent signals for blocking.
7. I implemented identical online features with Redis state and proved parity against batch features.
8. I added Kafka inbox/outbox processing, PostgreSQL audit records, DLQ handling, cases, delayed labels, MLflow governance, authenticated FastAPI, and Docker health checks.

## Questions you should be ready to answer

### Why PR-AUC instead of accuracy?

Fraud is approximately 0.9%. Predicting every event as legitimate would achieve about 99.1% accuracy while detecting no fraud. PR-AUC evaluates the precision/recall trade-off for the rare positive class.

### Why not use a random split?

The production task predicts future behavior. A random split mixes future and past customer patterns, merchant activity, and fraud strategies. A chronological split produces a more honest forward-looking evaluation.

### How did you prevent leakage?

Rolling windows use `closed="left"`; novelty counters are read before the current event is committed; labels and simulator provenance are excluded; calibration and policy tuning use separate chronological partitions; the test set remains frozen.

### Why did PR-AUC fall after hardening the simulator?

The initial data contained easier shortcuts. Adding legitimate fraud-like behavior and varying attack strength made the problem more realistic. Lower performance was accepted because the harder benchmark better measures generalization.

### Why calibrate the score?

Ranking alone is insufficient for economic decisions. Calibration makes a value such as 0.20 closer to an interpretable empirical risk, which improves threshold analysis and cost modeling.

### Why separate model and policy?

The model estimates risk; the organization decides what action to take. Review capacity, false-decline tolerance, fraud value, and evidence requirements can change without retraining the classifier.

### Why require two block signals?

One extreme feature—large amount, travel, or new device—can be legitimate. Requiring independent evidence reduces confident-looking false positives and routes ambiguity to review.

### Is the pipeline exactly once?

No. Kafka is at least once. Inbox deduplication, transaction-ID idempotency, database constraints, and a transactional outbox make repeated delivery safe. Claiming global exactly-once semantics across Kafka, Redis, and PostgreSQL would be misleading.

### Why Redis?

Online scoring requires shared, low-latency histories and counters across API workers. Process-local dictionaries would disagree after load balancing and would lose state on restart.

### What would you change at bank scale?

Partition feature ownership, remove the global lock, use managed Kafka/Redis/PostgreSQL, enforce TLS and workload identity, tokenize customer identifiers, add OpenTelemetry and Prometheus, perform fairness and stability validation, formalize model approval, and load-test under realistic concurrency.

## Metric interpretation

- `0.8038` PR-AUC is strong on this synthetic benchmark but not evidence of bank performance.
- `95.3%` block precision means few legitimate test events were blocked.
- `33.6%` block recall is intentionally conservative; review lifts total investigation capture to `85.0%`.
- `0.0031` Brier score indicates improved probability calibration.
- Production metrics with 100 labels and one fraud are correctly marked provisional.

## What not to claim

- Do not call the synthetic financial savings real business impact.
- Do not claim global exactly-once delivery.
- Do not claim the API-key implementation is equivalent to bank IAM.
- Do not present the 100-label production window as statistically reliable.
- Do not describe the Isolation Forest as an automatic blocking model.
