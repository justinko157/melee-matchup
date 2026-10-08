import json

import pytest

from pipeline.ingest.kafka_publisher import DeliveryError, KafkaPublisher


class FakeMessage:
    def __init__(self, topic, key):
        self._topic, self._key = topic, key

    def topic(self):
        return self._topic

    def key(self):
        return self._key


class FakeProducer:
    def __init__(self, config, *, fail_deliveries=False, remaining=0, buffer_full_times=0):
        self.config = config
        self.produced = []
        self.polls = []
        self._callbacks = []
        self._fail = fail_deliveries
        self._remaining = remaining
        self._buffer_full_times = buffer_full_times

    def produce(self, topic, key, value, on_delivery):
        if self._buffer_full_times:
            self._buffer_full_times -= 1
            raise BufferError("queue full")
        self.produced.append((topic, key, value))
        self._callbacks.append((on_delivery, FakeMessage(topic, key)))

    def poll(self, timeout):
        self.polls.append(timeout)
        return 0

    def flush(self, timeout):
        for callback, msg in self._callbacks:
            callback("broker down" if self._fail else None, msg)
        return self._remaining


def make(publisher_kwargs=None, **kwargs):
    holder = {}

    def factory(config):
        holder["producer"] = FakeProducer(config, **kwargs)
        return holder["producer"]

    publisher = KafkaPublisher("kafka:9092", producer_factory=factory, **(publisher_kwargs or {}))
    return publisher, holder


def test_producer_is_idempotent_with_acks_all():
    _, holder = make()
    config = holder["producer"].config
    assert config["bootstrap.servers"] == "kafka:9092"
    assert config["acks"] == "all"
    assert config["enable.idempotence"] is True


def test_publish_encodes_key_and_compact_json():
    publisher, holder = make()
    publisher.publish("t", "42", {"b": 1, "name": "Zaín"})
    topic, key, value = holder["producer"].produced[0]
    assert (topic, key) == ("t", b"42")
    assert json.loads(value.decode("utf-8")) == {"b": 1, "name": "Zaín"}
    assert b" " not in value
    assert publisher.published == 1


def test_publish_waits_when_local_queue_is_full():
    publisher, holder = make(buffer_full_times=2)
    publisher.publish("t", "1", {})
    assert len(holder["producer"].produced) == 1
    assert holder["producer"].polls[:2] == [1.0, 1.0]


def test_flush_succeeds_when_all_delivered():
    publisher, _ = make()
    publisher.publish("t", "1", {})
    publisher.flush()


def test_flush_raises_on_delivery_errors():
    publisher, _ = make(fail_deliveries=True)
    publisher.publish("t", "1", {})
    with pytest.raises(DeliveryError, match="broker down"):
        publisher.flush()


def test_flush_raises_when_messages_remain():
    publisher, _ = make(remaining=3)
    publisher.publish("t", "1", {})
    with pytest.raises(DeliveryError, match="3 messages undelivered"):
        publisher.flush()


def test_publish_gives_up_when_queue_stays_full_past_buffer_timeout():
    now = [0.0]

    def clock():
        now[0] += 1.0
        return now[0]

    publisher, holder = make({"buffer_timeout": 5.0, "clock": clock}, buffer_full_times=10**9)
    with pytest.raises(DeliveryError) as info:
        publisher.publish("melee.raw.test", "1", {"secret_payload": "do-not-log"})
    assert "do-not-log" not in str(info.value)
    assert "melee.raw.test" in str(info.value)
    assert holder["producer"].produced == []
    assert 1 <= len(holder["producer"].polls) <= 6
    assert publisher.published == 0
