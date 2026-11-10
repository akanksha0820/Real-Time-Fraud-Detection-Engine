CREATE SCHEMA IF NOT EXISTS fraud;
CREATE TABLE IF NOT EXISTS fraud.transactions (
    transaction_id text PRIMARY KEY, event_timestamp timestamptz NOT NULL,
    customer_id text NOT NULL, account_id text NOT NULL, device_id text NOT NULL,
    merchant_id text NOT NULL, amount_inr numeric(14,2) NOT NULL CHECK (amount_inr > 0),
    payment_channel text NOT NULL, latitude double precision, longitude double precision
);
ALTER TABLE fraud.transactions DROP COLUMN IF EXISTS is_fraud;
ALTER TABLE fraud.transactions DROP COLUMN IF EXISTS fraud_scenario;
CREATE INDEX IF NOT EXISTS idx_transactions_customer_time ON fraud.transactions(customer_id, event_timestamp);
CREATE INDEX IF NOT EXISTS idx_transactions_merchant_time ON fraud.transactions(merchant_id, event_timestamp);

CREATE TABLE IF NOT EXISTS fraud.risk_decisions (
    transaction_id text PRIMARY KEY REFERENCES fraud.transactions(transaction_id),
    scored_at timestamptz NOT NULL DEFAULT now(), fraud_probability double precision NOT NULL,
    decision text NOT NULL CHECK (decision IN ('ALLOW','REVIEW','BLOCK')),
    reason_codes jsonb NOT NULL, model_version text NOT NULL
);
ALTER TABLE fraud.risk_decisions ADD COLUMN IF NOT EXISTS feature_snapshot jsonb;

CREATE TABLE IF NOT EXISTS fraud.fraud_labels (
    transaction_id text PRIMARY KEY REFERENCES fraud.transactions(transaction_id),
    is_fraud boolean NOT NULL, fraud_scenario text,
    label_source text NOT NULL CHECK (label_source IN ('CHARGEBACK','ANALYST','SIMULATION')),
    observed_at timestamptz NOT NULL, confidence double precision NOT NULL DEFAULT 1.0
        CHECK (confidence>=0 AND confidence<=1),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS fraud.model_registry (
    model_version text PRIMARY KEY,
    stage text NOT NULL CHECK (stage IN ('CHAMPION','CHALLENGER','ARCHIVED')),
    artifact_path text NOT NULL, artifact_sha256 text NOT NULL,
    offline_metrics jsonb NOT NULL, registered_at timestamptz NOT NULL DEFAULT now(),
    approved_by text, approved_at timestamptz,
    CHECK ((stage='CHAMPION' AND approved_by IS NOT NULL AND approved_at IS NOT NULL) OR stage<>'CHAMPION')
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_one_champion ON fraud.model_registry(stage) WHERE stage='CHAMPION';

CREATE TABLE IF NOT EXISTS fraud.model_monitoring_runs (
    monitoring_run_id bigserial PRIMARY KEY, model_version text NOT NULL,
    window_start timestamptz, window_end timestamptz, labeled_transactions integer NOT NULL,
    metrics jsonb NOT NULL, segment_metrics jsonb NOT NULL, alerts jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS fraud.kafka_inbox (
    topic text NOT NULL, partition_id integer NOT NULL, offset_id bigint NOT NULL,
    message_id text UNIQUE NOT NULL, transaction_id text,
    processed_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (topic, partition_id, offset_id)
);

CREATE TABLE IF NOT EXISTS fraud.kafka_outbox (
    outbox_id bigserial PRIMARY KEY, event_key text NOT NULL, target_topic text NOT NULL,
    payload jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
    published_at timestamptz, attempts integer NOT NULL DEFAULT 0,
    UNIQUE (event_key, target_topic)
);
CREATE INDEX IF NOT EXISTS idx_kafka_outbox_pending
ON fraud.kafka_outbox(outbox_id) WHERE published_at IS NULL;

CREATE TABLE IF NOT EXISTS fraud.investigation_cases (
    case_id bigserial PRIMARY KEY,
    transaction_id text UNIQUE NOT NULL REFERENCES fraud.risk_decisions(transaction_id),
    status text NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN','IN_PROGRESS','RESOLVED','CLOSED')),
    priority text NOT NULL CHECK (priority IN ('LOW','MEDIUM','HIGH','CRITICAL')),
    assigned_to text, outcome text CHECK (outcome IS NULL OR outcome IN ('FRAUD','LEGITIMATE','INCONCLUSIVE')),
    notes text, created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
    resolved_at timestamptz
);
CREATE INDEX IF NOT EXISTS idx_cases_work_queue
ON fraud.investigation_cases(status, priority, created_at);

INSERT INTO fraud.investigation_cases(transaction_id,priority)
SELECT d.transaction_id,
       CASE WHEN d.fraud_probability>=0.35 OR t.amount_inr>=100000 THEN 'CRITICAL'
            WHEN d.fraud_probability>=0.15 OR t.amount_inr>=50000 THEN 'HIGH'
            ELSE 'MEDIUM' END
FROM fraud.risk_decisions d JOIN fraud.transactions t USING(transaction_id)
WHERE d.decision='REVIEW'
ON CONFLICT (transaction_id) DO NOTHING;

CREATE OR REPLACE VIEW fraud.v_pipeline_health AS
SELECT
    (SELECT COUNT(*) FROM fraud.kafka_inbox) AS consumed_messages,
    (SELECT COUNT(DISTINCT transaction_id) FROM fraud.kafka_inbox WHERE transaction_id IS NOT NULL) AS unique_valid_inputs,
    (SELECT COUNT(*) FROM fraud.transactions) AS persisted_transactions,
    (SELECT COUNT(*) FROM fraud.risk_decisions) AS persisted_decisions,
    (SELECT COUNT(*) FROM fraud.kafka_outbox WHERE published_at IS NULL) AS pending_outbox,
    (SELECT COUNT(*) FROM fraud.kafka_outbox WHERE target_topic='transactions-dlq') AS dead_letters,
    (SELECT MAX(processed_at) FROM fraud.kafka_inbox) AS last_consumed_at;

CREATE OR REPLACE VIEW fraud.v_hourly_decision_metrics AS
SELECT date_trunc('hour', d.scored_at) AS metric_hour, d.decision,
       COUNT(*) AS transactions, SUM(t.amount_inr) AS amount_inr,
       AVG(d.fraud_probability) AS average_probability
FROM fraud.risk_decisions d JOIN fraud.transactions t USING (transaction_id)
GROUP BY 1,2;

CREATE OR REPLACE VIEW fraud.v_review_queue AS
SELECT c.case_id,c.transaction_id,c.status,c.priority,c.assigned_to,c.outcome,c.created_at,
       EXTRACT(EPOCH FROM (now()-c.created_at))/60 AS age_minutes,
       d.fraud_probability,d.reason_codes,t.event_timestamp,t.customer_id,t.merchant_id,t.amount_inr
FROM fraud.investigation_cases c
JOIN fraud.risk_decisions d USING (transaction_id)
JOIN fraud.transactions t USING (transaction_id)
WHERE c.status IN ('OPEN','IN_PROGRESS');

CREATE OR REPLACE VIEW fraud.v_merchant_risk_concentration AS
SELECT t.merchant_id,COUNT(*) AS transactions,SUM(t.amount_inr) AS amount_inr,
       COUNT(*) FILTER (WHERE d.decision='BLOCK') AS blocks,
       COUNT(*) FILTER (WHERE d.decision='REVIEW') AS reviews,
       AVG(d.fraud_probability) AS average_probability
FROM fraud.transactions t JOIN fraud.risk_decisions d USING (transaction_id)
GROUP BY t.merchant_id;
