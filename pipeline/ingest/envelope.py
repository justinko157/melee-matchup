"""Message envelopes for the raw and dead-letter Kafka topics."""

from datetime import UTC, datetime

PAGINATION_FIELDS = frozenset({"page", "perPage"})


def strip_pagination(variables: dict) -> dict:
    """Drop page/perPage so variables identify a query, not one page of it."""
    return {k: v for k, v in variables.items() if k not in PAGINATION_FIELDS}


def build_envelope(
    *,
    entity: str,
    entity_id: object,
    payload: dict | None,
    run_id: str,
    partition_week: str,
    query_name: str,
    query_variables: dict,
    ingested_at: datetime,
) -> dict:
    """Wrap one API node with the metadata every raw message carries."""
    if entity_id is None:
        raise ValueError(f"entity_id is required for a {entity} envelope")
    if ingested_at.tzinfo is None:
        raise ValueError("ingested_at must be timezone-aware")
    return {
        "entity": entity,
        "entity_id": str(entity_id),
        "ingested_at": ingested_at.astimezone(UTC).isoformat(),
        "dagster_run_id": run_id,
        "partition_week": partition_week,
        "query_name": query_name,
        "query_variables": strip_pagination(query_variables),
        "payload": payload,
    }


def build_dlq_envelope(
    *,
    entity: str,
    entity_id: object,
    error: Exception,
    run_id: str,
    partition_week: str,
    query_name: str,
    query_variables: dict,
    ingested_at: datetime,
) -> dict:
    """Envelope for an entity that could not be fetched after retries."""
    envelope = build_envelope(
        entity=entity,
        entity_id=entity_id,
        payload=None,
        run_id=run_id,
        partition_week=partition_week,
        query_name=query_name,
        query_variables=query_variables,
        ingested_at=ingested_at,
    )
    envelope["error_type"] = type(error).__name__
    envelope["error_message"] = str(error)
    return envelope
