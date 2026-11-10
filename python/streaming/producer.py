"""Replay a bounded chronological sample as versioned Kafka events."""
import argparse, os, time
import pandas as pd
from confluent_kafka import Producer
from python.common import ROOT
from python.streaming.contracts import TransactionEnvelope

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-per-second", type=float, default=20)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--start-row", type=int, default=0)
    args = parser.parse_args()
    producer = Producer({"bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
                         "enable.idempotence": True, "acks": "all"})
    topic = os.getenv("KAFKA_TRANSACTION_TOPIC", "transactions")
    frame = pd.read_parquet(ROOT / "data/raw/transactions.parquet").sort_values("event_timestamp")
    frame = frame.iloc[args.start_row:args.start_row + args.limit]
    failures = []
    def delivered(error, _):
        if error: failures.append(str(error))
    for record in frame.to_dict("records"):
        envelope = TransactionEnvelope.create(record)
        producer.produce(topic, key=envelope.payload.customer_id,
                         value=envelope.model_dump_json(), callback=delivered)
        producer.poll(0)
        if args.events_per_second > 0: time.sleep(1 / args.events_per_second)
    remaining = producer.flush(30)
    if remaining or failures: raise RuntimeError(f"Kafka delivery failed: remaining={remaining}, errors={failures[:3]}")
    print(f"[PRODUCED] topic={topic} events={len(frame):,}")

if __name__ == "__main__": main()
