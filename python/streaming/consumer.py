"""Validate, score, persist atomically, then manually commit Kafka offsets."""
import json, os
from pydantic import ValidationError
from confluent_kafka import Consumer
from python.inference.service import RiskService
from python.streaming.contracts import TransactionEnvelope
from python.streaming.persistence import PostgresLedger

def main():
    servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    source = os.getenv("KAFKA_TRANSACTION_TOPIC", "transactions")
    target = os.getenv("KAFKA_SCORE_TOPIC", "risk-scores")
    dlq = os.getenv("KAFKA_DLQ_TOPIC", "transactions-dlq")
    consumer = Consumer({"bootstrap.servers": servers, "group.id": "fraud-scorers-v1",
                         "auto.offset.reset": "earliest", "enable.auto.commit": False})
    ledger, service = PostgresLedger(), RiskService()
    consumer.subscribe([source])
    print(f"[CONSUMING] {source} -> PostgreSQL outbox -> {target}")
    try:
        while True:
            message = consumer.poll(1)
            if message is None: continue
            if message.error(): raise RuntimeError(message.error())
            try:
                envelope = TransactionEnvelope.model_validate_json(message.value())
                event = envelope.payload.model_dump(mode="json")
                result = service.score(event)
                result["model_version"] = service.bundle.get("model_version", "calibrated_policy_v3")
                ledger.persist_decision(envelope.model_dump(mode="json"), result, message.topic(),
                                        message.partition(), message.offset(), target)
            except (ValidationError, json.JSONDecodeError, UnicodeDecodeError, KeyError, ValueError) as exc:
                ledger.persist_dead_letter(message.value(), repr(exc), message.topic(),
                                           message.partition(), message.offset(), dlq)
            consumer.commit(message=message, asynchronous=False)
    except KeyboardInterrupt:
        print("\n[STOPPED] consumer")
    finally:
        consumer.close()

if __name__ == "__main__": main()
