"""Idempotent JSON publisher for Kafka."""

import json
from collections.abc import Callable

from confluent_kafka import Producer

PRODUCER_CONFIG = {
    "acks": "all",
    "enable.idempotence": True,
    "compression.type": "zstd",
    "linger.ms": 50,
}


class DeliveryError(RuntimeError):
    """Some messages were not acknowledged by the broker."""


class KafkaPublisher:
    def __init__(
        self, bootstrap_servers: str, producer_factory: Callable[[dict], Producer] = Producer
    ):
        self._producer = producer_factory(
            {"bootstrap.servers": bootstrap_servers, **PRODUCER_CONFIG}
        )
        self._failures: list[str] = []
        self.published = 0

    def _on_delivery(self, err, msg) -> None:
        if err is not None:
            self._failures.append(f"{msg.topic()}[{msg.key()!r}]: {err}")

    def publish(self, topic: str, key: str, value: dict) -> None:
        data = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        while True:
            try:
                self._producer.produce(
                    topic, key=key.encode("utf-8"), value=data, on_delivery=self._on_delivery
                )
                break
            except BufferError:
                # Local queue is full: serve delivery callbacks to drain it, then retry.
                self._producer.poll(1.0)
        self._producer.poll(0)
        self.published += 1

    def flush(self, timeout: float = 60.0) -> None:
        """Wait for every message to be acknowledged; raise if any were not."""
        remaining = self._producer.flush(timeout)
        if remaining or self._failures:
            raise DeliveryError(
                f"{remaining} messages undelivered, {len(self._failures)} failed: "
                f"{self._failures[:5]}"
            )
