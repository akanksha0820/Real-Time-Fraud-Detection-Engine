"""Reliably publish committed PostgreSQL outbox records to Kafka."""
import json, os, time
from confluent_kafka import Producer
from python.streaming.persistence import PostgresLedger

def main():
    producer = Producer({"bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
                         "enable.idempotence": True, "acks": "all"})
    ledger = PostgresLedger()
    print("[OUTBOX] publishing pending decisions and dead letters")
    try:
        while True:
            rows = ledger.pending_outbox()
            if not rows:
                time.sleep(1); continue
            for outbox_id, event_key, topic, payload in rows:
                errors = []
                producer.produce(topic, key=event_key, value=json.dumps(payload, default=str),
                                 callback=lambda error, _: errors.append(str(error)) if error else None)
                remaining = producer.flush(30)
                if remaining or errors: raise RuntimeError(f"Outbox delivery failed: {errors}")
                ledger.mark_published(outbox_id)
    except KeyboardInterrupt:
        print("\n[STOPPED] outbox publisher")

if __name__ == "__main__": main()
