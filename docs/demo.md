# Five-Minute Demonstration

## Goal

Show the complete journey from transaction event to features, probability, business action, durable record, and analyst workflow without spending the demo waiting for a million-row pipeline.

## Before the demo

```powershell
$docker = "C:\Users\tanze\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe"
$env:POSTGRES_DSN = "postgresql://fraud:fraud@localhost:5433/fraud"

# Set the same FRAUD_API_KEY used when the API container was created.
& $docker compose up -d
& $docker compose ps
pytest -q
```

Open these tabs:

1. repository README;
2. Streamlit dashboard;
3. MLflow at `http://127.0.0.1:5000` if running;
4. a PowerShell terminal.

## Demo narrative

### 1. Establish credibility — 30 seconds

“This system generates one million synthetic financial transactions across 50,000 customers. Fraud is produced through six coherent attack scenarios, while legitimate hard negatives make classification non-trivial. Every historical feature is point-in-time correct.”

Show the verified-results table and architecture diagram.

### 2. Prove the service is operational — 30 seconds

```powershell
Invoke-RestMethod http://127.0.0.1:8000/ready | ConvertTo-Json -Depth 10
```

Explain that readiness checks the model artifact, shared Redis state, and PostgreSQL—not merely whether a process exists.

### 3. Score a suspicious event — 90 seconds

```powershell
$event = @{
    transaction_id = "PORTFOLIO_DEMO_001"
    event_timestamp = "2026-12-31T03:14:00Z"
    customer_id = "C0000001"
    account_id = "A00000001"
    device_id = "DEMO_NEW_DEVICE"
    merchant_id = "M000001"
    amount_inr = 84500
    payment_channel = "online"
    latitude = 19.0760
    longitude = 72.8777
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri http://127.0.0.1:8000/score `
    -Method Post `
    -ContentType "application/json" `
    -Headers @{ "X-API-Key" = $env:FRAUD_API_KEY } `
    -Body $event | ConvertTo-Json -Depth 10
```

Discuss the probability, action, reason codes, and point-in-time feature snapshot. Do not promise a specific action because the customer's prior Redis state can affect the result.

### 4. Demonstrate idempotency — 30 seconds

Send the same request again and show the same decision. Then inspect metrics:

```powershell
Invoke-RestMethod `
    -Uri http://127.0.0.1:8000/metrics `
    -Headers @{ "X-API-Key" = $env:FRAUD_API_KEY } |
    ConvertTo-Json -Depth 10
```

Point out that request count increases, but unique transaction count does not.

### 5. Show operations and governance — 60 seconds

```powershell
python -m python.monitoring.operational_report
streamlit run dashboard/app.py
```

Show reconciliation, the review queue, dead letters, merchant concentration, and model registry. Explain that review decisions generate cases and delayed confirmed labels later measure realized performance.

### 6. Close with engineering evidence — 60 seconds

Show:

- 21 passing tests;
- the leakage test;
- offline/online parity output;
- chronological split and calibration;
- the honest limitations section.

Closing sentence:

“The key design choice is that the classifier does not directly decide who gets blocked. It supplies a calibrated probability to an auditable policy that balances fraud loss, review capacity, evidence strength, and customer friction.”

## Demo reset

Use a new transaction ID for each live demonstration. To discard all local containers while preserving volumes:

```powershell
& $docker compose down
```

Do not use `docker compose down -v` unless intentionally deleting the PostgreSQL and Redis data volumes.
