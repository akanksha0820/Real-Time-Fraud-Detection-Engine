"""PostgreSQL inbox/outbox persistence for recoverable Kafka processing."""
from __future__ import annotations
import json, os
from pathlib import Path
from python.common import ROOT

class PostgresLedger:
    def __init__(self, dsn: str | None = None):
        try: import psycopg
        except ImportError as exc: raise RuntimeError("Install psycopg[binary] for streaming persistence") from exc
        self.psycopg = psycopg
        self.dsn = dsn or os.getenv("POSTGRES_DSN", "postgresql://fraud:fraud@localhost:5433/fraud")

    def initialize(self, schema_path: Path | None = None):
        sql = (schema_path or ROOT / "sql/schema.sql").read_text(encoding="utf-8")
        with self.psycopg.connect(self.dsn) as connection:
            connection.execute(sql)

    def persist_decision(self, envelope: dict, result: dict, topic: str, partition: int,
                         offset: int, output_topic: str) -> bool:
        event = envelope["payload"]
        message_id = f"{topic}:{partition}:{offset}"
        with self.psycopg.connect(self.dsn) as connection:
            already = connection.execute(
                "SELECT 1 FROM fraud.kafka_inbox WHERE topic=%s AND partition_id=%s AND offset_id=%s",
                (topic, partition, offset),
            ).fetchone()
            if already: return False
            connection.execute("""
                INSERT INTO fraud.transactions
                (transaction_id,event_timestamp,customer_id,account_id,device_id,merchant_id,
                 amount_inr,payment_channel,latitude,longitude)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (transaction_id) DO NOTHING
            """, (event["transaction_id"], event["event_timestamp"], event["customer_id"],
                  event["account_id"], event["device_id"], event["merchant_id"], event["amount_inr"],
                  event["payment_channel"], event["latitude"], event["longitude"]))
            connection.execute("""
                INSERT INTO fraud.risk_decisions
                (transaction_id,fraud_probability,decision,reason_codes,model_version,feature_snapshot)
                VALUES (%s,%s,%s,%s::jsonb,%s,%s::jsonb) ON CONFLICT (transaction_id) DO NOTHING
            """, (event["transaction_id"], result["fraud_probability"], result["decision"],
                  json.dumps(result["reasons"]), result.get("model_version", "calibrated_policy_v3"),
                  json.dumps(result.get("feature_snapshot", {}), default=str)))
            if result["decision"] == "REVIEW":
                probability, amount = float(result["fraud_probability"]), float(event["amount_inr"])
                priority = "CRITICAL" if probability >= .35 or amount >= 100000 else "HIGH" if probability >= .15 or amount >= 50000 else "MEDIUM"
                connection.execute("""INSERT INTO fraud.investigation_cases(transaction_id,priority)
                    VALUES (%s,%s) ON CONFLICT (transaction_id) DO NOTHING""",
                    (event["transaction_id"], priority))
            connection.execute("""
                INSERT INTO fraud.kafka_outbox(event_key,target_topic,payload)
                VALUES (%s,%s,%s::jsonb) ON CONFLICT (event_key,target_topic) DO NOTHING
            """, (event["transaction_id"], output_topic, json.dumps(result, default=str)))
            connection.execute("""
                INSERT INTO fraud.kafka_inbox(topic,partition_id,offset_id,message_id,transaction_id)
                VALUES (%s,%s,%s,%s,%s)
            """, (topic, partition, offset, message_id, event["transaction_id"]))
        return True

    def persist_dead_letter(self, raw_value: bytes | None, error: str, topic: str, partition: int,
                            offset: int, dlq_topic: str) -> bool:
        message_id = f"{topic}:{partition}:{offset}"
        payload = {"source_topic": topic, "partition": partition, "offset": offset,
                   "error": error[:2000], "raw_value": (raw_value or b"").decode("utf-8", errors="replace")[:10000]}
        with self.psycopg.connect(self.dsn) as connection:
            exists = connection.execute("SELECT 1 FROM fraud.kafka_inbox WHERE message_id=%s", (message_id,)).fetchone()
            if exists: return False
            connection.execute("""INSERT INTO fraud.kafka_outbox(event_key,target_topic,payload)
                VALUES (%s,%s,%s::jsonb) ON CONFLICT (event_key,target_topic) DO NOTHING""",
                (message_id, dlq_topic, json.dumps(payload)))
            connection.execute("""INSERT INTO fraud.kafka_inbox
                (topic,partition_id,offset_id,message_id) VALUES (%s,%s,%s,%s)""",
                (topic, partition, offset, message_id))
        return True

    def pending_outbox(self, limit=100):
        with self.psycopg.connect(self.dsn) as connection:
            return connection.execute("""SELECT outbox_id,event_key,target_topic,payload
                FROM fraud.kafka_outbox WHERE published_at IS NULL ORDER BY outbox_id LIMIT %s""", (limit,)).fetchall()

    def mark_published(self, outbox_id: int):
        with self.psycopg.connect(self.dsn) as connection:
            connection.execute("""UPDATE fraud.kafka_outbox SET published_at=now(), attempts=attempts+1
                WHERE outbox_id=%s""", (outbox_id,))
