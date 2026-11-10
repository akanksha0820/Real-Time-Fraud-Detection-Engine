# Contributing

## Development setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
```

## Before opening a pull request

```powershell
pytest -q
ruff check api tests/conftest.py tests/test_api_contract.py tests/test_api_security.py
bandit -q -r api
docker compose config
```

Changes to feature engineering must preserve the point-in-time contract and include a leakage or parity test. Changes to thresholds must be selected on policy-validation data, never the test period. Changes to streaming persistence must preserve idempotency and inbox/outbox reconciliation.

Do not commit generated datasets, model binaries, `.env`, database backups, local MLflow stores, or secrets.

## Pull request summary

Describe:

1. the problem being solved;
2. the design and trade-offs;
3. validation performed;
4. data-leakage, customer-friction, or operational implications;
5. any migration or rollback requirement.
