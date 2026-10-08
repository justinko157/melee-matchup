from datetime import UTC, datetime, timedelta, timezone

import pytest

from pipeline.ingest.envelope import build_dlq_envelope, build_envelope, strip_pagination

COMMON = dict(
    run_id="run-1",
    partition_week="2025-01-06",
    query_name="EventSets",
    query_variables={"eventId": 10, "page": 3, "perPage": 15},
)


def test_strip_pagination_removes_page_fields_only():
    assert strip_pagination({"eventId": 1, "page": 2, "perPage": 15}) == {"eventId": 1}


def test_build_envelope_has_exact_fields():
    env = build_envelope(
        entity="set",
        entity_id=123,
        payload={"id": 123},
        ingested_at=datetime(2025, 1, 6, 12, 0, tzinfo=UTC),
        **COMMON,
    )
    assert env == {
        "entity": "set",
        "entity_id": "123",
        "ingested_at": "2025-01-06T12:00:00+00:00",
        "dagster_run_id": "run-1",
        "partition_week": "2025-01-06",
        "query_name": "EventSets",
        "query_variables": {"eventId": 10},
        "payload": {"id": 123},
    }


def test_ingested_at_is_normalized_to_utc():
    pacific = timezone(timedelta(hours=-8))
    env = build_envelope(
        entity="set",
        entity_id="preview_1_2",
        payload={},
        ingested_at=datetime(2025, 1, 6, 4, 0, tzinfo=pacific),
        **COMMON,
    )
    assert env["ingested_at"] == "2025-01-06T12:00:00+00:00"
    assert env["entity_id"] == "preview_1_2"


def test_naive_ingested_at_is_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        build_envelope(
            entity="set", entity_id=1, payload={}, ingested_at=datetime(2025, 1, 6), **COMMON
        )


def test_dlq_envelope_has_error_fields_and_null_payload():
    env = build_dlq_envelope(
        entity="event",
        entity_id=10,
        error=TimeoutError("took too long"),
        ingested_at=datetime(2025, 1, 6, tzinfo=UTC),
        **COMMON,
    )
    assert env == {
        "entity": "event",
        "entity_id": "10",
        "ingested_at": "2025-01-06T00:00:00+00:00",
        "dagster_run_id": "run-1",
        "partition_week": "2025-01-06",
        "query_name": "EventSets",
        "query_variables": {"eventId": 10},
        "payload": None,
        "error_type": "TimeoutError",
        "error_message": "took too long",
    }
    assert env["payload"] is None


def test_none_entity_id_is_rejected():
    with pytest.raises(ValueError, match="entity_id"):
        build_envelope(
            entity="set",
            entity_id=None,
            payload={},
            ingested_at=datetime(2025, 1, 6, tzinfo=UTC),
            **COMMON,
        )
    with pytest.raises(ValueError, match="entity_id"):
        build_dlq_envelope(
            entity="event",
            entity_id=None,
            error=TimeoutError("x"),
            ingested_at=datetime(2025, 1, 6, tzinfo=UTC),
            **COMMON,
        )
