"""Create Kafka topics and apply idempotent PostgreSQL schema migrations."""
import os
from confluent_kafka.admin import AdminClient, NewTopic
from python.streaming.persistence import PostgresLedger

def main():
    servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    names = [os.getenv("KAFKA_TRANSACTION_TOPIC", "transactions"),
             os.getenv("KAFKA_SCORE_TOPIC", "risk-scores"),
             os.getenv("KAFKA_DLQ_TOPIC", "transactions-dlq")]
    admin = AdminClient({"bootstrap.servers": servers})
    existing = admin.list_topics(timeout=10).topics
    missing = [name for name in names if name not in existing]
    if missing:
        futures = admin.create_topics([
            NewTopic(name, num_partitions=3, replication_factor=1) for name in missing
        ])
        for _, future in futures.items():
            future.result()
        print(f"[CREATED] Kafka topics={missing}")
    else:
        print("[EXISTS] All Kafka topics already exist")
    PostgresLedger().initialize()
    print(f"[READY] topics={names} PostgreSQL schema applied")

if __name__ == "__main__": main()
