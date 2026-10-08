import json
from datetime import UTC, datetime

import pytest

from pipeline.ingest.envelope import build_envelope
from pipeline.topics import SETS_TOPIC, TOURNAMENTS_TOPIC

pytestmark = pytest.mark.spark
pytest.importorskip("pyspark")

from pyspark.sql.types import (  # noqa: E402
    BinaryType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from pipeline.spark_jobs.bronze import bronze_rows, parse_envelopes, reject_rows  # noqa: E402

KAFKA_SCHEMA = StructType(
    [
        StructField("key", BinaryType()),
        StructField("value", BinaryType()),
        StructField("topic", StringType()),
        StructField("partition", IntegerType()),
        StructField("offset", LongType()),
        StructField("timestamp", TimestampType()),
        StructField("timestampType", IntegerType()),
    ]
)
KAFKA_TS = datetime(2025, 1, 13, 6, 0, tzinfo=UTC)


def envelope(entity="set", entity_id=100, payload=None):
    return build_envelope(
        entity=entity,
        entity_id=entity_id,
        payload=payload if payload is not None else {"id": entity_id, "slots": [{"seed": 1}]},
        run_id="run-1",
        partition_week="2025-01-06",
        query_name="EventSets",
        query_variables={"eventId": 10, "perPage": 15},
        ingested_at=datetime(2025, 1, 13, 5, 59, 30, tzinfo=UTC),
    )


def kafka_df(spark, rows):
    """rows: list of (topic, value) where value is a dict, str, bytes or None."""
    data = []
    for offset, (topic, value) in enumerate(rows):
        if isinstance(value, dict):
            value = json.dumps(value).encode()
        elif isinstance(value, str):
            value = value.encode()
        data.append((b"k", value, topic, 0, offset, KAFKA_TS, 0))
    return spark.createDataFrame(data, KAFKA_SCHEMA)


def reasons(spark, rows):
    return [r.reject_reason for r in reject_rows(parse_envelopes(kafka_df(spark, rows))).collect()]


def test_valid_set_becomes_bronze_row(spark):
    parsed = parse_envelopes(kafka_df(spark, [(SETS_TOPIC, envelope())]))
    (row,) = bronze_rows(parsed, "set").collect()
    assert row.entity == "set"
    assert row.entity_id == "100"
    assert row.kafka_topic == SETS_TOPIC
    assert row.kafka_offset == 0
    assert row.kafka_key == "k"
    assert row.dagster_run_id == "run-1"
    assert row.partition_week == "2025-01-06"
    assert row.ingested_at.replace(tzinfo=UTC) == datetime(2025, 1, 13, 5, 59, 30, tzinfo=UTC)
    assert json.loads(row.payload) == {"id": 100, "slots": [{"seed": 1}]}
    assert json.loads(row.query_variables) == {"eventId": 10}
    assert reject_rows(parsed).count() == 0


def test_bronze_rows_filters_by_entity(spark):
    parsed = parse_envelopes(
        kafka_df(spark, [(TOURNAMENTS_TOPIC, envelope(entity="tournament", entity_id=1))])
    )
    assert bronze_rows(parsed, "set").count() == 0
    assert bronze_rows(parsed, "tournament").count() == 1


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (None, "null_value"),
        ("not json at all", "unparseable_envelope"),
        ("[1, 2]", "unparseable_envelope"),
        ({"entity": "set", "payload": {"id": 1}}, "missing_required_field"),
        ({**envelope(), "payload": None}, "missing_required_field"),
        ({**envelope(), "ingested_at": "yesterday"}, "invalid_ingested_at"),
    ],
)
def test_bad_messages_are_rejected_with_reason(spark, value, reason):
    assert reasons(spark, [(SETS_TOPIC, value)]) == [reason]


def test_entity_on_wrong_topic_is_rejected(spark):
    assert reasons(spark, [(TOURNAMENTS_TOPIC, envelope(entity="set"))]) == [
        "entity_topic_mismatch"
    ]


def test_rejects_keep_raw_bytes(spark):
    (row,) = reject_rows(parse_envelopes(kafka_df(spark, [(SETS_TOPIC, "garbage")]))).collect()
    assert bytes(row.raw_value) == b"garbage"
    assert bytes(row.raw_key) == b"k"
    assert row.rejected_at is not None


def test_rejected_rows_never_reach_bronze(spark):
    parsed = parse_envelopes(kafka_df(spark, [(SETS_TOPIC, "garbage"), (SETS_TOPIC, envelope())]))
    assert bronze_rows(parsed, "set").count() == 1


def test_unknown_topic_is_rejected(spark):
    assert reasons(spark, [("some.other.topic", envelope())]) == ["entity_topic_mismatch"]


def test_every_input_row_lands_in_exactly_one_place(spark):
    rows = [
        (SETS_TOPIC, envelope()),
        (TOURNAMENTS_TOPIC, envelope(entity="tournament", entity_id=1)),
        (SETS_TOPIC, "garbage"),
        (SETS_TOPIC, None),
        (TOURNAMENTS_TOPIC, envelope(entity="set")),
        ("some.other.topic", envelope()),
        ("some.other.topic", envelope(entity="tournament", entity_id=2)),
    ]
    parsed = parse_envelopes(kafka_df(spark, rows))
    accepted = bronze_rows(parsed, "set").count() + bronze_rows(parsed, "tournament").count()
    assert accepted == 2
    assert accepted + reject_rows(parsed).count() == len(rows)
