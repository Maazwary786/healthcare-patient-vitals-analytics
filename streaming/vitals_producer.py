
"""Publish synthetic patient vital signs to Kafka."""

import argparse
import hashlib
import json
import logging
import time
import uuid

from datetime import datetime, timezone

from kafka import KafkaProducer
from kafka.errors import KafkaError


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def patient_key(number: int) -> str:
    """Generate a stable pseudonymous key for a synthetic demo patient."""
    return hashlib.sha256(
        f"synthetic-demo-patient-{number}".encode("utf-8")
    ).hexdigest()


def build_event(number: int, mode: str) -> dict:
    """Build one synthetic vitals event."""
    spo2 = 87.0 if mode == "low" else 97.0

    return {
        "event_id": str(uuid.uuid4()),
        "patient_key": patient_key(number),
        "event_time": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z"),
        "heart_rate": 76.0,
        "respiratory_rate": 16.0,
        "spo2": spo2,
        "source": "synthetic_demo",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--topic", default="vitals_stream")
    parser.add_argument("--patient-number", type=int, default=1)
    parser.add_argument("--mode", choices=["normal", "low"], default="normal")
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument("--max-events", type=int, default=0)
    args = parser.parse_args()

    if args.patient_number < 1:
        parser.error("--patient-number must be positive")
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    if args.max_events < 0:
        parser.error("--max-events cannot be negative")

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_servers.split(","),
        key_serializer=lambda key: key.encode("utf-8"),
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        acks="all",
        retries=10,
        max_in_flight_requests_per_connection=1,
    )

    sent = 0

    try:
        logger.info(
            "Publishing to topic=%s mode=%s",
            args.topic,
            args.mode,
        )

        while args.max_events == 0 or sent < args.max_events:
            event = build_event(args.patient_number, args.mode)

            try:
                metadata = producer.send(
                    args.topic,
                    key=event["patient_key"],
                    value=event,
                ).get(timeout=15)
            except KafkaError:
                logger.exception("Failed to publish Kafka event")
                raise

            sent += 1
            logger.info(
                "sent=%d event_id=%s spo2=%.1f partition=%d offset=%d",
                sent,
                event["event_id"],
                event["spo2"],
                metadata.partition,
                metadata.offset,
            )

            if args.max_events == 0 or sent < args.max_events:
                time.sleep(args.interval_seconds)

    except KeyboardInterrupt:
        logger.info("Producer interrupted")
    finally:
        producer.flush(timeout=15)
        producer.close(timeout=15)
        logger.info("Producer closed; sent=%d", sent)


if __name__ == "__main__":
    main()
