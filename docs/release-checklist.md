# Screenshot and GitHub Release Checklist

## Captured dashboard evidence

Store final images in `docs/images/` with these names:

| Filename | Evidence |
| --- | --- |
| `dashboard-overview.png` | Pipeline state, KPIs and operational alert |
| `decision-traffic.png` | Decision mix, value and hourly throughput |
| `review-queue.png` | Workload KPIs and case priority |
| `merchant-exposure.png` | Risk concentration and investigation table |
| `simulation-trends.png` | Million-event summary and temporal behavior |
| `attack-scenarios.png` | Scenario volume and synthetic exposure |
| `model-governance.png` | Hard negatives and governance section |

Never capture or commit API keys, environment files, passwords, personal paths, private tokens, or unsanitized identifiers.

## Screenshot quality

- Use a 16:9 or wide browser viewport.
- Crop unused terminal space.
- Keep text readable at GitHub width.
- Use consistent dark or light theme.
- Show the system result, not installation logs.
- Sanitize local usernames when practical.

The main README uses one overview image and keeps the remaining views in a collapsible gallery. This gives recruiters immediate visual evidence without overwhelming the technical narrative.

## Repository cleanup

```powershell
git status
git check-ignore .env
git check-ignore artifacts/model.joblib
git check-ignore data/raw/transactions.parquet
pytest -q
ruff check api tests/conftest.py tests/test_api_contract.py tests/test_api_security.py
bandit -q -r api
```

Confirm the repository contains no `.env`, API keys, database backups, generated million-row datasets, trained binary models, or local MLflow database.

## Suggested GitHub metadata

Description:

> Production-style real-time fraud risk engine with Kafka, Redis, PostgreSQL, calibrated XGBoost, point-in-time features, decision economics, FastAPI, MLflow, monitoring, and Docker.

Topics:

`fraud-detection`, `machine-learning`, `data-engineering`, `kafka`, `redis`, `postgresql`, `fastapi`, `xgboost`, `mlflow`, `shap`, `streaming`, `decision-science`, `docker`

## Final release

```powershell
git add .
git status
git commit -m "Complete production-grade fraud risk engine"
git push origin main
git tag -a v1.0.0 -m "Portfolio release: real-time fraud risk engine"
git push origin v1.0.0
```

Review `git status` and the staged diff before committing. If the default branch is not `main`, use the actual branch name.

Suggested release title:

> v1.0.0 — Real-Time Fraud & Transaction Risk Engine

Suggested release summary:

> End-to-end portfolio release featuring a one-million-event synthetic financial ecosystem, leakage-safe offline and online features, calibrated XGBoost risk scoring, economic ALLOW/REVIEW/BLOCK policy, Kafka inbox/outbox processing, Redis state, PostgreSQL operations, delayed-label monitoring, MLflow governance, authenticated FastAPI, analyst dashboard, Docker health checks, and automated quality gates.
