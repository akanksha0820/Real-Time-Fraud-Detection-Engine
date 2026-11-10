<div align="center">

# 🛡️ Real-Time Fraud & Transaction Risk Engine

### From raw payment event to explainable, economically governed fraud decision

An end-to-end financial risk platform designed and built by **Tanzeel Aftab**.

[![Python 3.12](https://img.shields.io/badge/Python-3.12-2563EB?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-0F766E)](https://xgboost.readthedocs.io/)
[![Kafka](https://img.shields.io/badge/Streaming-Kafka-111827?logo=apachekafka)](https://kafka.apache.org/)
[![PostgreSQL](https://img.shields.io/badge/Ledger-PostgreSQL-336791?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Redis](https://img.shields.io/badge/Online_State-Redis-DC382D?logo=redis&logoColor=white)](https://redis.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-21_Passing-16A34A)](#quality-gates)
[![PR-AUC](https://img.shields.io/badge/Test_PR--AUC-0.8038-0EA5E9)](#verified-project-results)

[**Architecture**](docs/architecture.md) · [**Live demo guide**](docs/demo.md) · [**Deployment**](docs/deployment.md) · [**Interview guide**](docs/interview-guide.md) · [**Tanzeel on GitHub**](https://github.com/txnzeel) · [**LinkedIn**](https://www.linkedin.com/in/txnzeel/)

</div>

The engine simulates a realistic payment ecosystem, computes leakage-safe streaming features, produces calibrated fraud probabilities, and converts them into economically governed `ALLOW`, `REVIEW`, or `BLOCK` decisions.

This is not a notebook classifier. It demonstrates how data generation, point-in-time feature engineering, machine learning, streaming infrastructure, online state, decision policy, explainability, case operations, and model monitoring fit together in a production-style system.

> All customers, transactions, fraud outcomes, and financial results are synthetic. Reported performance demonstrates engineering and modeling behavior—not verified performance at a financial institution.

## Verified project results

| Area | Result |
| --- | ---: |
| Synthetic transactions | 1,000,000 |
| Customers / merchants | 50,000 / 8,000 |
| Scenario-driven fraud rate | 0.90% |
| Test PR-AUC | 0.8038 |
| Test ROC-AUC | 0.9766 |
| Block precision | 95.30% |
| Investigation capture rate | 85.00% |
| False-positive blocks | 19 of 124,490 legitimate test events |
| Calibrated Brier score | 0.0031 |
| Replay latency | 5.1 ms p50 / 7.4 ms p95 |
| Offline/online parity | PASS at `1e-6` tolerance |
| Automated tests | 21 passing |

The calibrated policy deliberately blocks fewer transactions than the earlier high-recall model. It routes more ambiguous cases to human review, reducing false blocks from 167 to 19 while maintaining an 85.0% investigation capture rate.

<p align="center">
  <img src="docs/images/dashboard-overview.png" alt="Fraud Risk Command Center overview" width="100%">
  <br>
  <sub><strong>Fraud Risk Command Center</strong> — portfolio dashboard built by Tanzeel Aftab</sub>
</p>

## System architecture

```mermaid
flowchart TB
    subgraph EventLayer[Event layer]
        SIM[Synthetic ecosystem] --> KAFKA[Kafka transactions]
    end

    subgraph RiskLayer[Real-time risk layer]
        KAFKA --> CONSUMER[Risk consumer]
        REDIS[(Redis state)] <--> CONSUMER
        CONSUMER --> MODEL[Calibrated XGBoost]
        MODEL --> POLICY[Decision policy]
    end

    subgraph Operations[Operations and governance]
        POLICY --> PG[(PostgreSQL ledger)]
        PG --> OUTBOX[Transactional outbox]
        OUTBOX --> SCORES[Kafka risk scores]
        PG --> DASH[Analyst dashboard]
        PG --> MONITOR[Monitoring and MLflow]
    end

    API[Secured FastAPI] <--> REDIS
    API --> MODEL
    API --> PG
```

The Kafka consumer and direct API share the same feature contract, model artifact, policy, and Redis-backed online state. PostgreSQL provides durable transaction, decision, investigation, label, monitoring, and model-governance records.

Detailed design: [Architecture](docs/architecture.md) · [Deployment runbook](docs/deployment.md) · [Demo guide](docs/demo.md)

## Why this project is different

- Fraud emerges from coherent attack scenarios rather than random labels.
- Legitimate hard negatives resemble fraud: travel, device replacement, large purchases, payment retries, and rapid shopping sessions.
- Every historical feature uses events strictly before the scored transaction.
- Train, calibration, policy-validation, and test periods are chronological.
- Probability calibration and decision thresholds are evaluated separately.
- Block decisions require multiple independent risk-signal families.
- Thresholds reflect fraud loss, review cost, false declines, and customer friction.
- Offline and online feature calculations have an automated parity test.
- Redis preserves shared state and idempotency across API processes.
- Kafka processing uses inbox/outbox persistence and a dead-letter path.
- Delayed-label monitoring refuses to present tiny samples as reliable production performance.
- API authentication, distributed rate limiting, safe logs, health checks, CI, and non-root containers are included.

## Fraud and legitimate scenarios

| Fraud simulation | Legitimate hard negative |
| --- | --- |
| Account takeover | Device replacement |
| Card testing | Payment retry |
| Velocity attack | Rapid shopping session |
| Impossible travel | Genuine travel |
| High-risk merchant abuse | Festival-scale purchase |
| Coordinated fraud ring | Late-night emergency |

Scenario names and legitimate-pattern provenance exist only for simulation audits. They are excluded from the model feature contract.

## Repository map

| Location | Responsibility |
| --- | --- |
| `config/settings.yaml` | Simulation, chronological splits, economics, policy configuration |
| `python/data_generation/` | Master entities, behavioral profiles, transactions, fraud scenarios |
| `python/preprocessing/` | Referential integrity and data-quality contracts |
| `python/features/` | Leakage-safe point-in-time feature engineering |
| `python/training/` | Training, calibration, evaluation, SHAP, model registry |
| `python/inference/` | Online features, Redis state, idempotent scoring, parity benchmark |
| `python/decision_engine/` | Evidence requirements, thresholds, actions, reason codes |
| `python/streaming/` | Kafka contracts, producer, consumer, inbox/outbox and DLQ |
| `python/monitoring/` | Drift, operations health, delayed-label production evaluation |
| `api/` | Authenticated FastAPI scoring and operations endpoints |
| `dashboard/` | Streamlit fraud-operations command center |
| `sql/schema.sql` | PostgreSQL ledger, cases, labels, registry, monitoring, SQL views |
| `tests/` | Leakage, parity, policy, API, state, streaming and monitoring tests |
| `docs/` | Architecture, runbook, demo, interview guide and release checklist |

## Quick start

Requirements: Python 3.12, Git, and optionally Docker Desktop.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run a 50,000-transaction verification build:

```powershell
python -m python.run_pipeline --quick
pytest -q
```

Run the complete 1,000,000-event model pipeline:

```powershell
python -m python.run_pipeline
python -m python.training.evaluate_model
python -m python.training.explain_model
```

Regenerate the master ecosystem only when required:

```powershell
python -m python.run_pipeline --regenerate-entities
```

## Leakage and temporal validation contract

For transaction `t`, every historical feature sees only events whose timestamp is earlier than `t`:

```python
grouped.rolling("10min", closed="left")
```

This applies to transaction counts, amount sums, prior device/merchant usage, previous location, travel speed, and merchant velocity. The model never receives `is_fraud`, `fraud_scenario`, or `legitimate_pattern`.

The time split is:

| Period | Purpose |
| --- | --- |
| Before 2026-07-01 | Model training |
| First half of validation | Isotonic calibration |
| Second half of validation | Decision-policy optimization |
| From 2026-10-01 | Frozen final test |

The test set is never used to select model parameters or decision thresholds.

## Decision economics

The model produces a calibrated probability; the policy owns the business action.

```mermaid
flowchart LR
    SCORE[Calibrated score] --> EVIDENCE{Independent signals}
    EVIDENCE -->|Low risk| ALLOW[ALLOW]
    EVIDENCE -->|Ambiguous| REVIEW[REVIEW]
    EVIDENCE -->|High risk + 2 signals| BLOCK[BLOCK]
```

Policy validation accounts for:

- missed fraud value;
- imperfect review recovery;
- manual investigation expense;
- customer friction from review;
- amount-sensitive false-decline impact.

This separation allows policy changes without retraining the model.

## Full Docker runtime

Generate a local API key and launch the stack:

```powershell
$bytes = New-Object byte[] 32
$rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
$rng.GetBytes($bytes)
$rng.Dispose()
$env:FRAUD_API_KEY = ([System.BitConverter]::ToString($bytes)).Replace("-", "").ToLower()

docker compose up -d --build
docker compose ps
```

The stack includes FastAPI, Kafka, Redis, and PostgreSQL. All services expose health checks; the API container runs as a non-root user.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/live
Invoke-RestMethod http://127.0.0.1:8000/ready
```

Protected endpoints require `X-API-Key`:

```powershell
Invoke-RestMethod `
    -Uri http://127.0.0.1:8000/model `
    -Headers @{ "X-API-Key" = $env:FRAUD_API_KEY }
```

API surface:

| Endpoint | Purpose | Authentication |
| --- | --- | --- |
| `GET /live` | Process liveness | Public |
| `GET /ready` | Model, Redis and PostgreSQL readiness | Public |
| `GET /model` | Governed model metadata | API key |
| `GET /metrics` | Inference traffic and latency | API key |
| `POST /score` | Real-time transaction decision | API key + rate limit |
| `GET /operations/*` | Queue, pipeline and governance data | API key |
| `PATCH /operations/cases/{id}` | Analyst case action | API key |

Access logs include method, route, status, latency, and correlation ID—but never request bodies, API keys, identifiers, or feature values.

## Streaming workflow

```powershell
$env:FRAUD_STATE_BACKEND = "redis"
$env:REDIS_URL = "redis://localhost:6379/0"
$env:POSTGRES_DSN = "postgresql://fraud:fraud@localhost:5433/fraud"
$env:KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"

python -m python.streaming.init_infrastructure
python -m python.streaming.consumer
python -m python.streaming.outbox_publisher
python -m python.streaming.producer --rows 100
```

The consumer records every Kafka message in an inbox, stores one durable transaction and decision per transaction ID, and writes outbound events to a transactional outbox. The publisher marks messages only after Kafka acknowledges delivery. Invalid messages go to `transactions-dlq`.

This provides idempotent processing and at-least-once delivery semantics without making a false end-to-end exactly-once claim.

## Monitoring and governance

```powershell
python -m python.monitoring.operational_report
python -m python.monitoring.reconcile_labels --delay-days 30
python -m python.monitoring.production_performance
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Monitoring covers feature PSI, pipeline reconciliation, dead letters, outbox backlog, fraud rate, PR-AUC, calibration, investigation capture, segment performance, and matured-label sufficiency.

A window with too few labels, too few fraud examples, or only one class is marked `PROVISIONAL`. Diagnostic metrics may still be shown, but promotion gates cannot treat them as reliable evidence.

## Analyst dashboard

```powershell
$env:POSTGRES_DSN = "postgresql://fraud:fraud@localhost:5433/fraud"
streamlit run dashboard/app.py
```

The dashboard shows pipeline reconciliation, decision volume and value, review cases, merchant concentration, champion/challenger status, monitoring runs, simulated fraud trends, and scenario context.

### Visual walkthrough

<table>
  <tr>
    <td width="50%" valign="top">
      <strong>Decision traffic</strong><br>
      <sub>Decision mix, transaction value, and hourly throughput.</sub><br><br>
      <img src="docs/images/decision-traffic.png" alt="Decision traffic dashboard">
    </td>
    <td width="50%" valign="top">
      <strong>Analyst investigations</strong><br>
      <sub>Prioritized review workload and case operations.</sub><br><br>
      <img src="docs/images/review-queue.png" alt="Analyst review queue">
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <strong>Merchant exposure</strong><br>
      <sub>Concentration of volume and risk across merchants.</sub><br><br>
      <img src="docs/images/merchant-exposure.png" alt="Merchant risk exposure dashboard">
    </td>
    <td width="50%" valign="top">
      <strong>Simulation behavior</strong><br>
      <sub>Rolling fraud rate and transaction amount patterns.</sub><br><br>
      <img src="docs/images/simulation-trends.png" alt="Fraud simulation trends">
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <strong>Attack scenarios</strong><br>
      <sub>Scenario frequency and simulated financial exposure.</sub><br><br>
      <img src="docs/images/attack-scenarios.png" alt="Fraud attack scenarios">
    </td>
    <td width="50%" valign="top">
      <strong>Model governance</strong><br>
      <sub>Hard-negative behavior and governed model status.</sub><br><br>
      <img src="docs/images/model-governance.png" alt="Model governance dashboard">
    </td>
  </tr>
</table>

## Quality gates

```powershell
pip install -r requirements-dev.txt
pytest -q
ruff check api tests/conftest.py tests/test_api_contract.py tests/test_api_security.py
bandit -q -r api
```

GitHub Actions repeats the tests, hardened-boundary linting, security scan, and container build for pushes and pull requests.

## Key artifacts

| Artifact | Meaning |
| --- | --- |
| `data/reference/*.parquet` | Customers, accounts, devices, merchants and profiles |
| `data/raw/transactions.parquet` | Event history and simulation-only ground truth |
| `data/processed/transaction_features.parquet` | Point-in-time model features |
| `artifacts/model.joblib` | Preprocessor, model, calibrator, feature contract and policy |
| `artifacts/model_metadata.json` | Frozen evaluation results and thresholds |
| `reports/model_evaluation/` | Scenario, segment, score-band and hard-negative audits |
| `reports/shap_feature_importance.csv` | Global SHAP importance |
| `reports/monitoring_report.json` | Feature drift report |
| `mlflow.db` | Local experiment and monitoring tracking backend |

Large generated data and trained binaries are intentionally excluded from Git. Reproduce them with the documented pipelines.

## Honest limitations

- Synthetic behavior cannot reproduce every adaptive adversary or institutional bias.
- The benchmark runs on one development machine, not a distributed load-test environment.
- The API key mechanism demonstrates service authentication; real deployments should use a secrets manager, TLS/mTLS, key rotation, and gateway identity.
- Redis uses a global scoring lock for correctness in this portfolio implementation; higher throughput would partition state ownership.
- PostgreSQL and Redis credentials in Compose are local-development defaults.
- Real deployment requires encrypted/tokenized identifiers, RBAC/SSO, immutable audit retention, backups, disaster recovery, network policies, vulnerability management, and formal model-risk approval.

## Interview resources

- [Five-minute demonstration](docs/demo.md)
- [Architecture deep dive](docs/architecture.md)
- [Deployment and operations runbook](docs/deployment.md)
- [Interview explanation guide](docs/interview-guide.md)
- [Screenshot and release checklist](docs/release-checklist.md)

## About the builder

**Tanzeel Aftab** designed and built this end-to-end fraud risk engine to demonstrate production-oriented data science, analytics engineering, machine learning, and real-time fraud decisioning.

The system covers the complete fraud detection lifecycle—from synthetic transaction generation and feature engineering to streaming inference, policy-based decisions, analyst workflows, observability, and model monitoring.

Connect with Tanzeel: [GitHub](https://github.com/txnzeel) · [LinkedIn](https://www.linkedin.com/in/txnzeel/)

## License and responsible use

This repository is an educational portfolio system. It must not be used to make real financial decisions without independent security, legal, compliance, fairness, validation, and model-risk review.

<p align="center"><sub>Designed and built by <strong>Tanzeel Aftab</strong> · 2026</sub></p>
