# Deployment and Operations Runbook

## Local environment variables

| Variable | Example | Purpose |
| --- | --- | --- |
| `FRAUD_API_KEY` | generated 64-character hex value | Protected API access |
| `FRAUD_STATE_BACKEND` | `redis` | Shared online state |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `POSTGRES_DSN` | `postgresql://fraud:fraud@localhost:5433/fraud` | Durable ledger |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Event broker |
| `FRAUD_RATE_LIMIT_PER_MINUTE` | `120` | Per-principal scoring limit |
| `LOG_LEVEL` | `INFO` | API log verbosity |

Copy `.env.example` to `.env` for local configuration, but never commit `.env`.

## Startup

```powershell
docker compose up -d --build
docker compose ps
python -m python.streaming.init_infrastructure
```

Healthy startup requires `api`, `postgres`, `redis`, and `kafka` to report healthy.

## Health interpretation

| Check | Meaning | Operator response |
| --- | --- | --- |
| `/live` fails | API process unavailable | Inspect API container and restart policy |
| Model readiness false | Artifact missing/corrupt or load failed | Rebuild image and verify model artifact |
| State readiness false | Redis unavailable | Inspect Redis health and connectivity |
| PostgreSQL readiness false | Ledger unavailable | Inspect database health, credentials, migrations |
| Outbox backlog rising | Publisher cannot publish | Inspect Kafka and outbox publisher |
| Dead letters present | Invalid or unprocessable events | Inspect payload contract and DLQ records |
| Transaction/decision mismatch | Persistence pipeline incomplete | Stop traffic and reconcile before replay |

## Useful commands

```powershell
docker compose logs api --tail 100
docker compose logs kafka --tail 100
docker compose exec redis redis-cli ping
docker compose exec postgres psql -U fraud -d fraud -c "SELECT * FROM fraud.v_pipeline_health;"
python -m python.monitoring.operational_report
```

## Database initialization

`sql/schema.sql` is applied automatically to an empty PostgreSQL volume. For an existing volume, apply compatible schema changes with:

```powershell
python -m python.streaming.init_infrastructure
```

Never delete a persistent volume merely to apply a migration.

## Backup guidance

For this local portfolio environment:

```powershell
docker compose exec postgres pg_dump -U fraud -d fraud -Fc -f /tmp/fraud.backup
docker cp fraud-risk-engine-postgres-1:/tmp/fraud.backup .\fraud.backup
```

A real environment requires automated encrypted backups, restore drills, retention policy, and point-in-time recovery.

## Secret rotation

Generate a new key in PowerShell, update the environment or secret store, and recreate only the API:

```powershell
$bytes = New-Object byte[] 32
$rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
$rng.GetBytes($bytes)
$rng.Dispose()
$env:FRAUD_API_KEY = ([System.BitConverter]::ToString($bytes)).Replace("-", "").ToLower()
docker compose up -d --force-recreate api
```

## Shutdown

```powershell
docker compose down
```

This preserves named volumes. Adding `-v` deletes local PostgreSQL and Redis data and should be treated as destructive.

## Production gap

The Compose stack is a reproducible development deployment. Production would use managed secrets, TLS, private networking, workload identity, orchestration, centralized logs and traces, managed databases, backups, autoscaling, vulnerability scanning, incident response, and formal governance.
