# Ingestion Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest start.gg Melee data week by week through Kafka into Iceberg bronze tables, orchestrated by Dagster, running in a local Docker Compose stack.

**Architecture:** A Dagster asset partitioned by week calls the start.gg GraphQL API and publishes one JSON envelope per tournament and per set to Kafka. A Spark Structured Streaming job (`availableNow` trigger, local mode) lands the topics into Iceberg tables registered in a Lakekeeper REST catalog, with data files on Silo (S3-compatible). Trino queries the tables. Failed events go to a dead-letter topic; malformed messages go to a rejects table; both surface as Dagster asset checks.

**Tech Stack:** Python 3.12, uv, Dagster 1.13, confluent-kafka 2.16, PySpark 4.0.4, Iceberg 1.11.0, Apache Kafka 4.3.1 (KRaft), Lakekeeper v0.14.0, Silo (MinIO fork), Trino 483, Postgres 17, Docker Compose, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-07-ingestion-foundation-design.md`

## Global Constraints

- Python is pinned to 3.12 (`.python-version`), `requires-python = ">=3.12,<3.13"`.
- Use `uv` for everything (`uv add`, `uv sync`, `uv run`). Never pip.
- Do not modify anything under `src/`, `app.py` or the existing `tests/test_*.py` files. The existing 20 tests must keep passing.
- Kafka topics: `startgg.tournaments.raw`, `startgg.sets.raw`, `startgg.ingest.dlq`. 3 partitions each, `retention.ms=-1`, `cleanup.policy=delete`. Auto-creation disabled.
- Producer settings: `acks=all`, `enable.idempotence=true`.
- Envelope fields, exactly: `entity`, `entity_id`, `ingested_at`, `dagster_run_id`, `partition_week`, `query_name`, `query_variables`, `payload`. DLQ adds `error_type`, `error_message`; DLQ `payload` is `null`; DLQ `entity` is `event`.
- Bronze tables: `bronze.startgg_tournaments`, `bronze.startgg_sets`, `bronze.startgg_rejects`, in the Lakekeeper warehouse `melee`. Bronze is append-only and keeps duplicates.
- Weekly partitions start Monday 2018-01-01 (UTC). Backfill target is 2025-01-06 onward.
- Default `min_attendees` is 50. Only offline tournaments; only Melee events (videogame ID 1).
- The start.gg token must never appear in logs, exceptions, envelopes or Dagster config shown in the UI.
- Before every commit, run `uv run ruff format pipeline tests/pipeline tests/smoke scripts` and `uv run ruff check pipeline tests scripts`. Code in this plan may be reflowed by the formatter (line length 100); that is expected.
- Commit messages must not include a `Co-Authored-By: Claude` trailer (user's global instruction).
- Pinned images: `postgres:17.11`, `quay.io/lakekeeper/catalog:v0.14.0`, `pgsty/silo:RELEASE.2026-09-16T00-00-00Z`, `apache/kafka:4.3.1`, `kafbat/kafka-ui:v1.5.0`, `trinodb/trino:483`, `curlimages/curl:8.22.0`, `python:3.12-slim-bookworm`, `ghcr.io/astral-sh/uv:0.12.3`.
- Pinned Spark packages: `org.apache.iceberg:iceberg-spark-runtime-4.0_2.13:1.11.0`, `org.apache.iceberg:iceberg-aws-bundle:1.11.0`, `org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.4`.

## Review Focus

1. **Empty or missing token.** `STARTGG_API_TOKEN` unset or blank should fail with a message naming the variable, not a stack trace from `requests`. Pinned by `test_empty_token_raises_invalid_token` (Task 3).
2. **A week with no tournaments.** Early-January or pandemic weeks can return nothing; the asset should succeed with zero counts and publish nothing. Pinned by `test_empty_week_publishes_nothing` (Task 4).
3. **Null nested fields from the API.** start.gg returns `events: null`, `videogame: null` and `nodes: null`; these should be treated as empty, not crash. Pinned by `test_null_events_and_videogame_are_skipped` (Task 4) and `test_paginate_null_nodes_is_empty` (Task 3).
4. **Concurrent bronze runs.** A 2025 backfill produces ~90 successful ingestion runs; each triggers the sensor. Two Spark streaming queries sharing a checkpoint would corrupt it. Bronze runs must be serialized and queued requests collapsed. Pinned by the `spark_bronze` pool (Task 10) and `test_sensor_skips_when_bronze_run_already_queued` (Task 10).
5. **Re-running the bronze job with nothing new.** It should report zero rows and pass its check, not fail on a null sum. Pinned by the empty-topic run in Task 9, Step 4, and by `added_rows_since` using `coalesce`.

---

## File Structure

```
.gitattributes                         LF line endings (containers and Git Bash need them)
.python-version                        3.12
pyproject.toml                         adds `pipeline` extra, pytest config
pipeline/
  __init__.py
  topics.py                            topic name constants
  ingest/
    __init__.py
    windows.py                         partition key -> week bounds
    envelope.py                        raw and DLQ envelopes, strip_pagination
    queries.py                         GraphQL queries (copied from src/queries.py)
    startgg.py                         client: rate limit, retry, error classes, paginate
    collect.py                         collect_week: discovery, filters, publish, DLQ
    fixtures.py                        RecordingClient, FixtureClient
    kafka_publisher.py                 KafkaPublisher
  spark_jobs/
    __init__.py
    bronze.py                          pure transforms + DDL
    bronze_job.py                      spark-submit entrypoint
  dagster_defs/
    __init__.py                        `defs`
    partitions.py
    resources.py
    assets.py
    automation.py                      jobs, sensor, schedule
scripts/
  record_startgg_fixture.py
  smoke.sh
infra/
  pipeline.Dockerfile
  docker-compose.yml
  docker-compose.smoke.yml
  postgres/init-databases.sql
  lakekeeper/create-warehouse.json
  spark/spark-defaults.conf
  spark/warm_ivy_cache.py
  spark/check_lakehouse.py
  trino/catalog/lakekeeper.properties
  dagster/dagster.yaml
  dagster/workspace.yaml
tests/
  pipeline/
    __init__.py
    fakes.py                           FakePaginator, FakePublisher
    conftest.py                        spark session fixture
    test_windows.py
    test_envelope.py
    test_startgg.py
    test_collect.py
    test_fixtures.py
    test_kafka_publisher.py
    test_infra_config.py
    test_bronze_transform.py           (marked spark)
    test_dagster_defs.py
  smoke/
    __init__.py
    check_bronze.py
  fixtures/startgg/week_2025-01-06.json
```

---

### Task 1: Python 3.12 pin, dependencies and package skeleton

**Files:**
- Create: `.gitattributes`, `.python-version`, `pipeline/__init__.py`, `pipeline/ingest/__init__.py`, `pipeline/spark_jobs/__init__.py`, `pipeline/dagster_defs/__init__.py`, `tests/pipeline/__init__.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: importable `pipeline` package; `uv sync --all-extras` installs dagster, confluent-kafka, pyspark 4.0.4, trino.

- [ ] **Step 1: Add `.gitattributes`**

```gitattributes
# Shell scripts, configs and SQL are read inside Linux containers and by Git
# Bash, which both break on CRLF line endings.
* text=auto eol=lf
*.png binary
*.ipynb text eol=lf
```

- [ ] **Step 2: Renormalize existing files and commit separately**

```bash
git add .gitattributes
git add --renormalize .
git commit -m "Normalize line endings to LF"
```

- [ ] **Step 3: Pin Python**

Create `.python-version`:

```
3.12
```

- [ ] **Step 4: Edit `pyproject.toml`**

Change `requires-python = ">=3.10"` to:

```toml
requires-python = ">=3.12,<3.13"
```

Add this extra under `[project.optional-dependencies]`, after `app`:

```toml
pipeline = [
    "confluent-kafka>=2.16",
    "dagster>=1.13.25",
    "dagster-postgres>=0.29.25",
    "dagster-webserver>=1.13.25",
    "pyspark==4.0.4",
    "trino>=0.340",
]
```

Change `target-version = "py310"` to `target-version = "py312"`, and append:

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
addopts = "-m 'not spark'"
markers = [
    "spark: needs a local SparkSession (Java). Run with `make test-spark`.",
]
```

- [ ] **Step 5: Create the empty package files**

Create each of these with a one-line docstring:

- `pipeline/__init__.py`: `"""Data engineering pipeline: start.gg -> Kafka -> Iceberg."""`
- `pipeline/ingest/__init__.py`: `"""start.gg ingestion: API client, envelopes and Kafka publishing."""`
- `pipeline/spark_jobs/__init__.py`: `"""Spark jobs that land and transform lakehouse tables."""`
- `pipeline/dagster_defs/__init__.py`: `"""Dagster definitions for the pipeline."""`
- `tests/pipeline/__init__.py`: empty file.

- [ ] **Step 6: Sync and verify**

Run:

```bash
uv sync --all-extras
uv run python -c "import sys, dagster, confluent_kafka, pyspark, trino; print(sys.version.split()[0], dagster.__version__, pyspark.__version__)"
uv run pytest -q
```

Expected: `3.12.x 1.13.25 4.0.4` (dagster may be a newer 1.13.x patch), then `20 passed`.

- [ ] **Step 7: Commit**

```bash
git add .python-version pyproject.toml uv.lock pipeline tests/pipeline
git commit -m "Pin Python 3.12 and add pipeline dependencies and package skeleton"
```

---

### Task 2: Week windows and message envelopes

**Files:**
- Create: `pipeline/ingest/windows.py`, `pipeline/ingest/envelope.py`
- Test: `tests/pipeline/test_windows.py`, `tests/pipeline/test_envelope.py`

**Interfaces:**
- Produces:
  - `week_bounds(partition_key: str) -> tuple[int, int]`
  - `strip_pagination(variables: dict) -> dict`
  - `build_envelope(*, entity: str, entity_id: object, payload: dict | None, run_id: str, partition_week: str, query_name: str, query_variables: dict, ingested_at: datetime) -> dict`
  - `build_dlq_envelope(*, entity: str, entity_id: object, error: Exception, run_id: str, partition_week: str, query_name: str, query_variables: dict, ingested_at: datetime) -> dict`

- [ ] **Step 1: Write the failing tests**

`tests/pipeline/test_windows.py`:

```python
import pytest

from pipeline.ingest.windows import week_bounds


def test_week_bounds_cover_monday_to_sunday_utc():
    # 2025-01-06 00:00:00 UTC = 1736121600; next Monday = 1736726400
    assert week_bounds("2025-01-06") == (1736121600, 1736726399)


def test_adjacent_weeks_do_not_overlap():
    _, first_end = week_bounds("2025-01-06")
    second_start, _ = week_bounds("2025-01-13")
    assert second_start == first_end + 1


def test_rejects_non_monday():
    with pytest.raises(ValueError, match="not a Monday"):
        week_bounds("2025-01-07")


def test_rejects_malformed_key():
    with pytest.raises(ValueError):
        week_bounds("2025-W02")
```

`tests/pipeline/test_envelope.py`:

```python
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
    assert env["payload"] is None
    assert env["entity"] == "event"
    assert env["error_type"] == "TimeoutError"
    assert env["error_message"] == "took too long"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_windows.py tests/pipeline/test_envelope.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.ingest.windows'`.

- [ ] **Step 3: Implement**

`pipeline/ingest/windows.py`:

```python
"""Weekly partition windows for start.gg ingestion."""

from datetime import UTC, datetime, timedelta

WEEK = timedelta(days=7)


def week_bounds(partition_key: str) -> tuple[int, int]:
    """Return (after_date, before_date) unix timestamps for a weekly partition.

    The window runs from Monday 00:00:00 UTC to the following Sunday 23:59:59
    UTC, so adjacent weeks never overlap.
    """
    start = datetime.strptime(partition_key, "%Y-%m-%d").replace(tzinfo=UTC)
    if start.weekday() != 0:
        raise ValueError(f"Partition key {partition_key!r} is not a Monday")
    end = start + WEEK
    return int(start.timestamp()), int(end.timestamp()) - 1
```

`pipeline/ingest/envelope.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_windows.py tests/pipeline/test_envelope.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add pipeline/ingest/windows.py pipeline/ingest/envelope.py tests/pipeline/test_windows.py tests/pipeline/test_envelope.py
git commit -m "Add weekly partition windows and Kafka message envelopes"
```

---

### Task 3: start.gg client with error classification

Rewrites `src/api_client.py`'s logic into `pipeline/ingest/startgg.py` with injectable session, clock and sleep. Behavior changes versus `src/api_client.py`, all deliberate:
- 5xx responses are retried.
- An invalid token fails immediately (no retries).
- Hitting the 10,000-result cap raises `PaginationCapError` instead of returning partial results.
- When a complexity error halves the page size, pagination restarts at page 1. The old code kept its page number, which skips or repeats records because page boundaries move when the page size changes.

**Files:**
- Create: `pipeline/ingest/startgg.py`, `pipeline/ingest/queries.py`
- Test: `tests/pipeline/test_startgg.py`

**Interfaces:**
- Produces:
  - Exceptions: `StartGGError` (base), `InvalidTokenError`, `RetriesExhaustedError`, `RequestRejectedError`, `GraphQLError`, `ComplexityError(GraphQLError)`, `PaginationCapError(GraphQLError)`
  - `operation_name(query: str) -> str`
  - `RateLimiter(max_requests=80, window_seconds=60.0, clock=time.monotonic, sleep=time.sleep)` with `.acquire() -> None`
  - `StartGGClient(token: str, session=None, rate_limiter=None, sleep=time.sleep)` with `.query(query: str, variables: dict) -> dict`, `.paginate(query: str, variables: dict, data_path: list[str]) -> list[dict]`, attribute `.api_calls: int`
  - Constants: `API_URL`, `MELEE_VIDEOGAME_ID = 1`, `MAX_ATTEMPTS = 4`
  - `pipeline.ingest.queries.TOURNAMENTS_BY_GAME`, `pipeline.ingest.queries.EVENT_SETS`

- [ ] **Step 1: Copy the queries**

```bash
cp src/queries.py pipeline/ingest/queries.py
```

Then change its docstring line to:

```python
"""GraphQL query definitions for the start.gg API.

Copied from src/queries.py, which is retired in sub-project 2.
"""
```

- [ ] **Step 2: Write the failing tests**

`tests/pipeline/test_startgg.py`:

```python
import pytest
import requests

from pipeline.ingest.startgg import (
    MAX_ATTEMPTS,
    ComplexityError,
    GraphQLError,
    InvalidTokenError,
    PaginationCapError,
    RateLimiter,
    RequestRejectedError,
    RetriesExhaustedError,
    StartGGClient,
    operation_name,
)

TOKEN = "secret-token-value"
QUERY = "query EventSets($eventId: ID!) { event(id: $eventId) { id } }"
PAGED_QUERY = "query EventSets($eventId: ID!, $page: Int!, $perPage: Int!) { x }"


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body
        self.text = text

    def json(self):
        return self._body


class FakeSession:
    """Returns queued responses (or raises queued exceptions) in order."""

    def __init__(self, *responses):
        self.headers = {}
        self._responses = list(responses)
        self.calls = []

    def post(self, url, json, timeout):
        self.calls.append(json)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def ok(data):
    return FakeResponse(200, {"data": data})


def make_client(*responses):
    session = FakeSession(*responses)
    sleeps = []
    client = StartGGClient(
        TOKEN,
        session=session,
        rate_limiter=RateLimiter(max_requests=10_000),
        sleep=sleeps.append,
    )
    return client, session, sleeps


def test_operation_name():
    assert operation_name(QUERY) == "EventSets"


def test_empty_token_raises_invalid_token():
    with pytest.raises(InvalidTokenError, match="STARTGG_API_TOKEN"):
        StartGGClient("")


def test_query_returns_data_and_sets_auth_header():
    client, session, _ = make_client(ok({"event": {"id": 1}}))
    assert client.query(QUERY, {"eventId": 1}) == {"event": {"id": 1}}
    assert session.headers["Authorization"] == f"Bearer {TOKEN}"
    assert client.api_calls == 1


def test_invalid_token_400_fails_fast_without_retry():
    client, session, _ = make_client(
        FakeResponse(400, text='{"message":"Invalid authentication token"}')
    )
    with pytest.raises(InvalidTokenError):
        client.query(QUERY, {})
    assert len(session.calls) == 1


def test_401_fails_fast():
    client, session, _ = make_client(FakeResponse(401, text="Unauthorized"))
    with pytest.raises(InvalidTokenError):
        client.query(QUERY, {})
    assert len(session.calls) == 1


@pytest.mark.parametrize(
    "first",
    [
        FakeResponse(429),
        FakeResponse(502),
        requests.exceptions.Timeout(),
        requests.exceptions.ConnectionError(),
    ],
)
def test_transient_failures_are_retried(first):
    client, session, sleeps = make_client(first, ok({"event": None}))
    assert client.query(QUERY, {}) == {"event": None}
    assert len(session.calls) == 2
    assert sleeps == [2.0]


def test_gives_up_after_max_attempts():
    client, session, sleeps = make_client(*[FakeResponse(503)] * MAX_ATTEMPTS)
    with pytest.raises(RetriesExhaustedError, match="EventSets"):
        client.query(QUERY, {})
    assert len(session.calls) == MAX_ATTEMPTS
    assert sleeps == [2.0, 4.0, 8.0]


def test_other_4xx_is_not_retried():
    client, session, _ = make_client(FakeResponse(404, text="not found"))
    with pytest.raises(RequestRejectedError, match="404"):
        client.query(QUERY, {})
    assert len(session.calls) == 1


@pytest.mark.parametrize(
    ("message", "error_class"),
    [
        ("Your query complexity is too high", ComplexityError),
        ("Cannot query more than 10,000th entry", PaginationCapError),
        ("Something else broke", GraphQLError),
    ],
)
def test_graphql_errors_are_classified(message, error_class):
    client, _, _ = make_client(FakeResponse(200, {"errors": [{"message": message}]}))
    with pytest.raises(error_class):
        client.query(QUERY, {})


def test_error_messages_never_contain_the_token():
    client, _, _ = make_client(FakeResponse(401, text="Unauthorized"))
    with pytest.raises(InvalidTokenError) as invalid:
        client.query(QUERY, {})
    client, _, _ = make_client(*[FakeResponse(500)] * MAX_ATTEMPTS)
    with pytest.raises(RetriesExhaustedError) as exhausted:
        client.query(QUERY, {})
    assert TOKEN not in str(invalid.value)
    assert TOKEN not in str(exhausted.value)


def test_rate_limiter_sleeps_until_window_frees():
    now = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(max_requests=2, window_seconds=10.0, clock=lambda: now[0], sleep=sleep)
    limiter.acquire()
    limiter.acquire()
    limiter.acquire()
    assert sleeps == [10.0]


def page(nodes, page_num, total_pages):
    return ok({"event": {"sets": {"nodes": nodes, "pageInfo": {"totalPages": total_pages, "page": page_num}}}})


def test_paginate_concatenates_pages():
    client, session, _ = make_client(page([{"id": 1}], 1, 2), page([{"id": 2}], 2, 2))
    nodes = client.paginate(PAGED_QUERY, {"eventId": 5, "perPage": 15}, ["event", "sets"])
    assert nodes == [{"id": 1}, {"id": 2}]
    assert [c["variables"]["page"] for c in session.calls] == [1, 2]


def test_paginate_halves_page_size_and_restarts_on_complexity():
    complexity = FakeResponse(200, {"errors": [{"message": "query complexity is too high"}]})
    client, session, _ = make_client(
        page([{"id": 1}], 1, 2), complexity, page([{"id": 1}], 1, 1)
    )
    nodes = client.paginate(PAGED_QUERY, {"eventId": 5, "perPage": 20}, ["event", "sets"])
    assert nodes == [{"id": 1}]
    assert [(c["variables"]["page"], c["variables"]["perPage"]) for c in session.calls] == [
        (1, 20),
        (2, 20),
        (1, 10),
    ]


def test_paginate_raises_on_pagination_cap():
    cap = FakeResponse(200, {"errors": [{"message": "Cannot query more than 10,000th entry"}]})
    client, _, _ = make_client(page([{"id": 1}], 1, 3), cap)
    with pytest.raises(PaginationCapError):
        client.paginate(PAGED_QUERY, {"eventId": 5, "perPage": 15}, ["event", "sets"])


def test_paginate_null_nodes_is_empty():
    client, _, _ = make_client(page(None, 1, 1))
    assert client.paginate(PAGED_QUERY, {"eventId": 5}, ["event", "sets"]) == []


def test_paginate_missing_path_is_empty():
    client, _, _ = make_client(ok({"event": None}))
    assert client.paginate(PAGED_QUERY, {"eventId": 5}, ["event", "sets"]) == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_startgg.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.ingest.startgg'`.

- [ ] **Step 4: Implement**

`pipeline/ingest/startgg.py`:

```python
"""start.gg GraphQL client: rate limiting, retries, error classification, pagination."""

import logging
import re
import time
from collections import deque
from collections.abc import Callable

import requests

logger = logging.getLogger(__name__)

API_URL = "https://api.start.gg/gql/alpha"
MELEE_VIDEOGAME_ID = 1

# start.gg allows 80 requests per 60 seconds
MAX_REQUESTS_PER_WINDOW = 80
WINDOW_SECONDS = 60.0
MAX_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 2.0  # waits 2s, 4s, 8s between attempts
MIN_PAGE_SIZE = 5
TOKEN_HELP = "Create a new token at https://start.gg/admin/profile/developer"


class StartGGError(Exception):
    """Base class for start.gg failures."""


class InvalidTokenError(StartGGError):
    """The token is missing or rejected. Never retried; fails the run."""


class RetriesExhaustedError(StartGGError):
    """Transient failures (429, 5xx, timeouts) persisted past MAX_ATTEMPTS."""


class RequestRejectedError(StartGGError):
    """A non-retryable HTTP 4xx other than an auth failure."""


class GraphQLError(StartGGError):
    """The response contained GraphQL errors."""


class ComplexityError(GraphQLError):
    """The query asked for too many objects per page."""


class PaginationCapError(GraphQLError):
    """start.gg refuses to page past its 10,000th result."""


def operation_name(query: str) -> str:
    """Return the GraphQL operation name, e.g. 'EventSets'."""
    match = re.search(r"\bquery\s+(\w+)", query)
    if not match:
        raise ValueError("GraphQL query has no operation name")
    return match.group(1)


def _classify_graphql_errors(errors: list) -> GraphQLError:
    text = str(errors)
    if "complexity" in text.lower():
        return ComplexityError(text)
    if "10,000" in text or "10000" in text:
        return PaginationCapError(text)
    return GraphQLError(text)


class RateLimiter:
    """Sliding-window limiter: at most max_requests per window_seconds."""

    def __init__(
        self,
        max_requests: int = MAX_REQUESTS_PER_WINDOW,
        window_seconds: float = WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._max = max_requests
        self._window = window_seconds
        self._clock = clock
        self._sleep = sleep
        self._timestamps: deque[float] = deque()

    def _evict(self, now: float) -> None:
        while self._timestamps and now - self._timestamps[0] >= self._window:
            self._timestamps.popleft()

    def acquire(self) -> None:
        now = self._clock()
        self._evict(now)
        if len(self._timestamps) >= self._max:
            wait = self._window - (now - self._timestamps[0])
            logger.info("Rate limit reached, waiting %.1fs", wait)
            self._sleep(wait)
            now = self._clock()
            self._evict(now)
        self._timestamps.append(now)


class StartGGClient:
    """GraphQL client for the start.gg API."""

    def __init__(
        self,
        token: str,
        session: requests.Session | None = None,
        rate_limiter: RateLimiter | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not token:
            raise InvalidTokenError(f"STARTGG_API_TOKEN is empty. Set it in .env. {TOKEN_HELP}")
        self._session = session or requests.Session()
        self._session.headers.update(
            {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        )
        self._rate_limiter = rate_limiter or RateLimiter()
        self._sleep = sleep
        self.api_calls = 0

    def query(self, query: str, variables: dict) -> dict:
        """Run one GraphQL request and return its 'data'."""
        payload = {"query": query, "variables": variables}
        last_error: object = None
        for attempt in range(MAX_ATTEMPTS):
            if attempt:
                wait = BACKOFF_BASE_SECONDS**attempt
                logger.warning(
                    "Retrying %s in %.0fs after: %s", operation_name(query), wait, last_error
                )
                self._sleep(wait)
            self._rate_limiter.acquire()
            self.api_calls += 1
            try:
                resp = self._session.post(API_URL, json=payload, timeout=30)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_error = repr(exc)
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}"
                continue
            if resp.status_code == 401 or (
                resp.status_code == 400 and "authentication token" in resp.text.lower()
            ):
                raise InvalidTokenError(
                    f"start.gg rejected STARTGG_API_TOKEN (HTTP {resp.status_code}). {TOKEN_HELP}"
                )
            if resp.status_code >= 400:
                raise RequestRejectedError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            body = resp.json()
            if body.get("errors"):
                raise _classify_graphql_errors(body["errors"])
            return body["data"]
        raise RetriesExhaustedError(
            f"{operation_name(query)} failed after {MAX_ATTEMPTS} attempts: {last_error}"
        )

    def paginate(self, query: str, variables: dict, data_path: list[str]) -> list[dict]:
        """Fetch every page of a paginated field and return all nodes.

        On a complexity error the page size is halved and pagination restarts
        at page 1, because page boundaries move when the page size changes.
        """
        nodes: list[dict] = []
        page = 1
        per_page = variables.get("perPage", 50)
        while True:
            try:
                data = self.query(query, {**variables, "page": page, "perPage": per_page})
            except ComplexityError:
                if per_page <= MIN_PAGE_SIZE:
                    raise
                per_page = max(MIN_PAGE_SIZE, per_page // 2)
                logger.warning("Query too complex, restarting with page size %d", per_page)
                nodes, page = [], 1
                continue
            obj = data
            for key in data_path:
                obj = (obj or {}).get(key)
            if obj is None:
                return nodes
            nodes.extend(obj.get("nodes") or [])
            total_pages = (obj.get("pageInfo") or {}).get("totalPages") or 1
            if page >= total_pages:
                return nodes
            page += 1
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_startgg.py -v`
Expected: 21 passed (including parametrized cases).

- [ ] **Step 6: Commit**

```bash
git add pipeline/ingest/startgg.py pipeline/ingest/queries.py tests/pipeline/test_startgg.py
git commit -m "Add start.gg client with retries, error classification and safe pagination"
```

---

### Task 4: Weekly collection with dead-lettering

**Files:**
- Create: `pipeline/topics.py`, `pipeline/ingest/collect.py`, `tests/pipeline/fakes.py`
- Test: `tests/pipeline/test_collect.py`

**Interfaces:**
- Consumes: `week_bounds`, `build_envelope`, `build_dlq_envelope` (Task 2); `operation_name`, `MELEE_VIDEOGAME_ID`, `StartGGError`, `InvalidTokenError`, `TOURNAMENTS_BY_GAME`, `EVENT_SETS` (Task 3).
- Produces:
  - `pipeline.topics`: `TOURNAMENTS_TOPIC = "startgg.tournaments.raw"`, `SETS_TOPIC = "startgg.sets.raw"`, `DLQ_TOPIC = "startgg.ingest.dlq"`, `RAW_TOPICS = (TOURNAMENTS_TOPIC, SETS_TOPIC)`, `ALL_TOPICS = (*RAW_TOPICS, DLQ_TOPIC)`
  - `WeekStats` dataclass: `tournaments: int`, `events: int`, `sets: int`, `dlq_messages: int`
  - `collect_week(client, publisher, partition_week: str, *, run_id: str, min_attendees: int = 50, now=...) -> WeekStats`, where `client` has `.paginate(query, variables, data_path)` and `publisher` has `.publish(topic: str, key: str, value: dict)`
  - `tests/pipeline/fakes.py`: `FakePaginator(tournaments: list[dict], sets_by_event: dict[int, list[dict] | Exception], discovery_error: Exception | None = None)` with `.calls: list[tuple[str, dict]]` and `.api_calls`; `FakePublisher` with `.messages: list[tuple[str, str, dict]]`, `.flushed: bool`, `.publish(...)`, `.flush()`

- [ ] **Step 1: Create topic constants**

`pipeline/topics.py`:

```python
"""Kafka topic names shared by producers, the Spark job and Compose."""

TOURNAMENTS_TOPIC = "startgg.tournaments.raw"
SETS_TOPIC = "startgg.sets.raw"
DLQ_TOPIC = "startgg.ingest.dlq"

RAW_TOPICS = (TOURNAMENTS_TOPIC, SETS_TOPIC)
ALL_TOPICS = (*RAW_TOPICS, DLQ_TOPIC)
```

- [ ] **Step 2: Create shared test fakes**

`tests/pipeline/fakes.py`:

```python
"""Test doubles for the start.gg client and Kafka publisher."""

from pipeline.ingest.startgg import operation_name


class FakePaginator:
    """Stands in for StartGGClient.paginate.

    sets_by_event maps an event ID to its set nodes, or to an exception that
    paginating that event should raise.
    """

    def __init__(self, tournaments, sets_by_event=None, discovery_error=None):
        self.tournaments = tournaments
        self.sets_by_event = sets_by_event or {}
        self.discovery_error = discovery_error
        self.calls = []
        self.api_calls = 0

    def paginate(self, query, variables, data_path):
        name = operation_name(query)
        self.calls.append((name, dict(variables)))
        self.api_calls += 1
        if name == "TournamentsByVideogame":
            if self.discovery_error:
                raise self.discovery_error
            return self.tournaments
        result = self.sets_by_event.get(variables["eventId"], [])
        if isinstance(result, Exception):
            raise result
        return result


class FakePublisher:
    def __init__(self):
        self.messages = []
        self.flushed = False

    def publish(self, topic, key, value):
        self.messages.append((topic, key, value))

    def flush(self, timeout=60.0):
        self.flushed = True

    def on(self, topic):
        return [m for m in self.messages if m[0] == topic]


def tournament(tid, *, attendees=100, online=False, events=None):
    return {"id": tid, "name": f"T{tid}", "numAttendees": attendees, "isOnline": online, "events": events}


def melee_event(eid):
    return {"id": eid, "name": f"E{eid}", "videogame": {"id": 1}}
```

- [ ] **Step 3: Write the failing tests**

`tests/pipeline/test_collect.py`:

```python
from datetime import UTC, datetime

import pytest

from pipeline.ingest.collect import collect_week
from pipeline.ingest.startgg import InvalidTokenError, PaginationCapError, RetriesExhaustedError
from pipeline.topics import DLQ_TOPIC, SETS_TOPIC, TOURNAMENTS_TOPIC
from tests.pipeline.fakes import FakePaginator, FakePublisher, melee_event, tournament

WEEK = "2025-01-06"
FIXED_NOW = datetime(2025, 1, 13, 6, 0, tzinfo=UTC)


def run(paginator, min_attendees=50):
    publisher = FakePublisher()
    stats = collect_week(
        paginator, publisher, WEEK, run_id="run-1", min_attendees=min_attendees, now=lambda: FIXED_NOW
    )
    return stats, publisher


def test_discovery_uses_week_bounds():
    paginator = FakePaginator([])
    run(paginator)
    name, variables = paginator.calls[0]
    assert name == "TournamentsByVideogame"
    assert variables["afterDate"] == 1736121600
    assert variables["beforeDate"] == 1736726399
    assert variables["videogameId"] == 1


def test_empty_week_publishes_nothing():
    stats, publisher = run(FakePaginator([]))
    assert publisher.messages == []
    assert (stats.tournaments, stats.events, stats.sets, stats.dlq_messages) == (0, 0, 0, 0)


def test_only_eligible_tournaments_are_published():
    tournaments = [
        tournament(1, attendees=60),
        tournament(2, attendees=60, online=True),
        tournament(3, attendees=10),
        {"id": 4, "numAttendees": None, "isOnline": False, "events": []},
    ]
    stats, publisher = run(FakePaginator(tournaments))
    assert [key for _, key, _ in publisher.on(TOURNAMENTS_TOPIC)] == ["1"]
    assert stats.tournaments == 1


def test_sets_published_for_melee_events_only():
    other_game = {"id": 20, "name": "Ultimate", "videogame": {"id": 1386}}
    paginator = FakePaginator(
        [tournament(1, events=[melee_event(10), other_game])],
        sets_by_event={10: [{"id": 100}, {"id": "preview_10_1"}], 20: [{"id": 200}]},
    )
    stats, publisher = run(paginator)
    assert [key for _, key, _ in publisher.on(SETS_TOPIC)] == ["100", "preview_10_1"]
    assert (stats.events, stats.sets) == (1, 2)


def test_null_events_and_videogame_are_skipped():
    no_videogame = {"id": 11, "name": "E11", "videogame": None}
    paginator = FakePaginator([tournament(1, events=None), tournament(2, events=[no_videogame])])
    stats, publisher = run(paginator)
    assert stats.tournaments == 2
    assert publisher.on(SETS_TOPIC) == []


def test_set_envelope_fields():
    paginator = FakePaginator([tournament(1, events=[melee_event(10)])], {10: [{"id": 100}]})
    _, publisher = run(paginator)
    _, _, envelope = publisher.on(SETS_TOPIC)[0]
    assert envelope["entity"] == "set"
    assert envelope["dagster_run_id"] == "run-1"
    assert envelope["partition_week"] == WEEK
    assert envelope["query_name"] == "EventSets"
    assert envelope["query_variables"] == {"eventId": 10}
    assert envelope["ingested_at"] == "2025-01-13T06:00:00+00:00"
    assert envelope["payload"] == {"id": 100}


@pytest.mark.parametrize("error", [RetriesExhaustedError("boom"), PaginationCapError("cap")])
def test_event_failure_goes_to_dlq_and_run_continues(error):
    paginator = FakePaginator(
        [tournament(1, events=[melee_event(10), melee_event(11)])],
        sets_by_event={10: error, 11: [{"id": 110}]},
    )
    stats, publisher = run(paginator)
    (_, key, dlq), = publisher.on(DLQ_TOPIC)
    assert key == "10"
    assert dlq["entity"] == "event"
    assert dlq["payload"] is None
    assert dlq["error_type"] == type(error).__name__
    assert [k for _, k, _ in publisher.on(SETS_TOPIC)] == ["110"]
    assert stats.dlq_messages == 1


def test_invalid_token_propagates():
    paginator = FakePaginator(
        [tournament(1, events=[melee_event(10)])], sets_by_event={10: InvalidTokenError("bad")}
    )
    with pytest.raises(InvalidTokenError):
        run(paginator)


def test_discovery_failure_fails_the_run():
    with pytest.raises(RetriesExhaustedError):
        run(FakePaginator([], discovery_error=RetriesExhaustedError("down")))
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_collect.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.ingest.collect'`.

- [ ] **Step 5: Implement**

`pipeline/ingest/collect.py`:

```python
"""Collect one week of start.gg data and publish it to Kafka."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from pipeline.ingest.envelope import build_dlq_envelope, build_envelope
from pipeline.ingest.queries import EVENT_SETS, TOURNAMENTS_BY_GAME
from pipeline.ingest.startgg import (
    MELEE_VIDEOGAME_ID,
    InvalidTokenError,
    StartGGError,
    operation_name,
)
from pipeline.ingest.windows import week_bounds
from pipeline.topics import DLQ_TOPIC, SETS_TOPIC, TOURNAMENTS_TOPIC

logger = logging.getLogger(__name__)

TOURNAMENTS_PER_PAGE = 50
SETS_PER_PAGE = 15


class Paginator(Protocol):
    def paginate(self, query: str, variables: dict, data_path: list[str]) -> list[dict]: ...


class Publisher(Protocol):
    def publish(self, topic: str, key: str, value: dict) -> None: ...


@dataclass
class WeekStats:
    tournaments: int = 0
    events: int = 0
    sets: int = 0
    dlq_messages: int = 0


def is_eligible_tournament(tournament: dict, min_attendees: int) -> bool:
    return not tournament.get("isOnline", False) and (
        tournament.get("numAttendees") or 0
    ) >= min_attendees


def melee_events(tournament: dict) -> list[dict]:
    return [
        event
        for event in tournament.get("events") or []
        if (event.get("videogame") or {}).get("id") == MELEE_VIDEOGAME_ID
    ]


def collect_week(
    client: Paginator,
    publisher: Publisher,
    partition_week: str,
    *,
    run_id: str,
    min_attendees: int = 50,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> WeekStats:
    """Publish a week's eligible tournaments and their Melee sets.

    An event whose sets cannot be fetched is dead-lettered and the week
    continues. An invalid token or a failed discovery query raises.
    """
    after_date, before_date = week_bounds(partition_week)
    discovery_vars = {
        "videogameId": MELEE_VIDEOGAME_ID,
        "afterDate": after_date,
        "beforeDate": before_date,
        "perPage": TOURNAMENTS_PER_PAGE,
    }
    common = {"run_id": run_id, "partition_week": partition_week}
    stats = WeekStats()

    tournaments = client.paginate(TOURNAMENTS_BY_GAME, discovery_vars, ["tournaments"])
    for tournament in tournaments:
        if not is_eligible_tournament(tournament, min_attendees):
            continue
        publisher.publish(
            TOURNAMENTS_TOPIC,
            str(tournament["id"]),
            build_envelope(
                entity="tournament",
                entity_id=tournament["id"],
                payload=tournament,
                query_name=operation_name(TOURNAMENTS_BY_GAME),
                query_variables=discovery_vars,
                ingested_at=now(),
                **common,
            ),
        )
        stats.tournaments += 1

        for event in melee_events(tournament):
            stats.events += 1
            set_vars = {"eventId": event["id"], "perPage": SETS_PER_PAGE}
            try:
                sets = client.paginate(EVENT_SETS, set_vars, ["event", "sets"])
            except InvalidTokenError:
                raise
            except StartGGError as exc:
                logger.error("Dead-lettering event %s: %s", event["id"], exc)
                publisher.publish(
                    DLQ_TOPIC,
                    str(event["id"]),
                    build_dlq_envelope(
                        entity="event",
                        entity_id=event["id"],
                        error=exc,
                        query_name=operation_name(EVENT_SETS),
                        query_variables=set_vars,
                        ingested_at=now(),
                        **common,
                    ),
                )
                stats.dlq_messages += 1
                continue

            for set_node in sets:
                if set_node.get("id") is None:
                    logger.warning("Skipping set without an id in event %s", event["id"])
                    continue
                publisher.publish(
                    SETS_TOPIC,
                    str(set_node["id"]),
                    build_envelope(
                        entity="set",
                        entity_id=set_node["id"],
                        payload=set_node,
                        query_name=operation_name(EVENT_SETS),
                        query_variables=set_vars,
                        ingested_at=now(),
                        **common,
                    ),
                )
                stats.sets += 1
    return stats
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_collect.py -v`
Expected: all pass (10 tests).

- [ ] **Step 7: Commit**

```bash
git add pipeline/topics.py pipeline/ingest/collect.py tests/pipeline/fakes.py tests/pipeline/test_collect.py
git commit -m "Add weekly collection that publishes envelopes and dead-letters failed events"
```

---

### Task 5: Recorded fixtures (record and replay)

**Files:**
- Create: `pipeline/ingest/fixtures.py`, `scripts/record_startgg_fixture.py`, `tests/fixtures/startgg/week_2025-01-06.json` (generated)
- Test: `tests/pipeline/test_fixtures.py`

**Interfaces:**
- Consumes: `collect_week`, `WeekStats`, `is_eligible_tournament`, `melee_events` (Task 4); `StartGGClient`, `operation_name` (Task 3); `strip_pagination` (Task 2).
- Produces:
  - `fixture_key(query: str, variables: dict) -> str`
  - `RecordingClient(inner)` with `.paginate(...)`, `.recorded: dict[str, list]`, `.save(path) -> None`, `.api_calls` (delegates to inner)
  - `FixtureClient(path)` with `.paginate(...)` and `.api_calls = 0`
  - Fixture file `tests/fixtures/startgg/week_2025-01-06.json`, path constant used by Tasks 10–11: `FIXTURE_WEEK = "2025-01-06"`

- [ ] **Step 1: Write the failing tests**

`tests/pipeline/test_fixtures.py`:

```python
import json
from pathlib import Path

import pytest

from pipeline.ingest.collect import collect_week, is_eligible_tournament, melee_events
from pipeline.ingest.fixtures import FixtureClient, RecordingClient, fixture_key
from pipeline.ingest.queries import EVENT_SETS
from pipeline.topics import DLQ_TOPIC, SETS_TOPIC, TOURNAMENTS_TOPIC
from tests.pipeline.fakes import FakePaginator, FakePublisher, melee_event, tournament

FIXTURE_WEEK = "2025-01-06"
FIXTURE = Path("tests/fixtures/startgg") / f"week_{FIXTURE_WEEK}.json"


def test_fixture_key_ignores_pagination():
    assert fixture_key(EVENT_SETS, {"eventId": 1, "page": 4, "perPage": 15}) == (
        'EventSets:{"eventId": 1}'
    )


def test_record_then_replay_round_trips(tmp_path):
    inner = FakePaginator([tournament(1, events=[melee_event(10)])], {10: [{"id": 100}]})
    recorder = RecordingClient(inner)
    collect_week(recorder, FakePublisher(), FIXTURE_WEEK, run_id="r")
    path = tmp_path / "week.json"
    recorder.save(path)

    replayed = FakePublisher()
    collect_week(FixtureClient(path), replayed, FIXTURE_WEEK, run_id="r")
    assert [k for _, k, _ in replayed.on(SETS_TOPIC)] == ["100"]


def test_fixture_client_reports_missing_key(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(KeyError, match="EventSets"):
        FixtureClient(path).paginate(EVENT_SETS, {"eventId": 1}, ["event", "sets"])


def test_recorded_week_replays_consistently():
    """The real recorded week: message counts must match the raw fixture."""
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    (discovery,) = [v for k, v in data.items() if k.startswith("TournamentsByVideogame:")]
    eligible = [t for t in discovery if is_eligible_tournament(t, 50)]
    expected_sets = sum(
        len(data[fixture_key(EVENT_SETS, {"eventId": e["id"]})])
        for t in eligible
        for e in melee_events(t)
    )

    publisher = FakePublisher()
    stats = collect_week(FixtureClient(FIXTURE), publisher, FIXTURE_WEEK, run_id="r")

    assert stats.tournaments == len(eligible) > 0
    assert stats.sets == expected_sets > 0
    assert len(publisher.on(TOURNAMENTS_TOPIC)) == stats.tournaments
    assert publisher.on(DLQ_TOPIC) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_fixtures.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.ingest.fixtures'`.

- [ ] **Step 3: Implement the fixture clients**

`pipeline/ingest/fixtures.py`:

```python
"""Record and replay start.gg pagination results, for tests and the smoke test."""

import json
from pathlib import Path

from pipeline.ingest.envelope import strip_pagination
from pipeline.ingest.startgg import operation_name


def fixture_key(query: str, variables: dict) -> str:
    """Identify a paginated query by operation name and non-paging variables."""
    return f"{operation_name(query)}:{json.dumps(strip_pagination(variables), sort_keys=True)}"


class RecordingClient:
    """Wraps a real client and records every paginate() result."""

    def __init__(self, inner):
        self._inner = inner
        self.recorded: dict[str, list] = {}

    @property
    def api_calls(self) -> int:
        return self._inner.api_calls

    def paginate(self, query: str, variables: dict, data_path: list[str]) -> list[dict]:
        nodes = self._inner.paginate(query, variables, data_path)
        self.recorded[fixture_key(query, variables)] = nodes
        return nodes

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.recorded, separators=(",", ":"), sort_keys=True), encoding="utf-8"
        )


class FixtureClient:
    """Replays a recorded week without calling the API."""

    api_calls = 0

    def __init__(self, path: str | Path):
        self._data = json.loads(Path(path).read_text(encoding="utf-8"))

    def paginate(self, query: str, variables: dict, data_path: list[str]) -> list[dict]:
        key = fixture_key(query, variables)
        if key not in self._data:
            raise KeyError(f"No recorded response for {key}")
        return self._data[key]
```

- [ ] **Step 4: Write the recording script**

`scripts/record_startgg_fixture.py`:

```python
"""Record one week of real start.gg responses as a test fixture.

Usage (from the repo root):
    uv run python -m scripts.record_startgg_fixture 2025-01-06
"""

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from pipeline.ingest.collect import collect_week
from pipeline.ingest.fixtures import RecordingClient
from pipeline.ingest.startgg import StartGGClient

FIXTURE_DIR = Path("tests/fixtures/startgg")


class NullPublisher:
    def publish(self, topic: str, key: str, value: dict) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("week", help="Monday partition key, e.g. 2025-01-06")
    parser.add_argument("--min-attendees", type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    load_dotenv(".env")
    client = RecordingClient(StartGGClient(os.environ.get("STARTGG_API_TOKEN", "")))
    stats = collect_week(
        client, NullPublisher(), args.week, run_id="fixture-recording", min_attendees=args.min_attendees
    )

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIXTURE_DIR / f"week_{args.week}.json"
    client.save(out)
    print(f"{stats} api_calls={client.api_calls} -> {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Record the fixture (calls the real API; needs `.env`)**

Run: `uv run python -m scripts.record_startgg_fixture 2025-01-06`

Expected: a line like `WeekStats(tournaments=N, events=M, sets=S, dlq_messages=0) api_calls=... -> tests/fixtures/startgg/week_2025-01-06.json (X MB)`.

Acceptance: `tournaments >= 1`, `sets >= 1`, `dlq_messages == 0`, file under 3 MB. If the week has zero sets, re-run with `2025-01-13` and then replace every occurrence of `2025-01-06` used as the fixture week in this plan (Tasks 5, 10, 11) with that week. If it is over 3 MB, re-run with `--min-attendees 100` and use `min_attendees=100` in `test_recorded_week_replays_consistently`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_fixtures.py -v`
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add pipeline/ingest/fixtures.py scripts/record_startgg_fixture.py tests/pipeline/test_fixtures.py tests/fixtures/startgg/week_2025-01-06.json
git commit -m "Add record/replay fixture clients and a recorded start.gg week"
```

---

### Task 6: Kafka publisher

**Files:**
- Create: `pipeline/ingest/kafka_publisher.py`
- Test: `tests/pipeline/test_kafka_publisher.py`

**Interfaces:**
- Produces:
  - `PRODUCER_CONFIG: dict` (includes `"acks": "all"`, `"enable.idempotence": True`)
  - `DeliveryError(RuntimeError)`
  - `KafkaPublisher(bootstrap_servers: str, producer_factory=confluent_kafka.Producer)` with `.publish(topic: str, key: str, value: dict) -> None`, `.flush(timeout: float = 60.0) -> None` (raises `DeliveryError`), `.published: int`

- [ ] **Step 1: Write the failing tests**

`tests/pipeline/test_kafka_publisher.py`:

```python
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


def make(**kwargs):
    holder = {}

    def factory(config):
        holder["producer"] = FakeProducer(config, **kwargs)
        return holder["producer"]

    return KafkaPublisher("kafka:9092", producer_factory=factory), holder


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_kafka_publisher.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.ingest.kafka_publisher'`.

- [ ] **Step 3: Implement**

`pipeline/ingest/kafka_publisher.py`:

```python
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
    def __init__(self, bootstrap_servers: str, producer_factory: Callable[[dict], Producer] = Producer):
        self._producer = producer_factory({"bootstrap.servers": bootstrap_servers, **PRODUCER_CONFIG})
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_kafka_publisher.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add pipeline/ingest/kafka_publisher.py tests/pipeline/test_kafka_publisher.py
git commit -m "Add idempotent Kafka JSON publisher"
```

---

### Task 7: Local stack (Silo, Lakekeeper, Postgres, Kafka, Trino)

**Files:**
- Create: `infra/pipeline.Dockerfile`, `infra/docker-compose.yml`, `infra/postgres/init-databases.sql`, `infra/lakekeeper/create-warehouse.json`, `infra/spark/spark-defaults.conf`, `infra/spark/warm_ivy_cache.py`, `infra/spark/check_lakehouse.py`, `infra/trino/catalog/lakekeeper.properties`
- Modify: `.env.example`, `.dockerignore`, `Makefile`
- Test: `tests/pipeline/test_infra_config.py`

**Interfaces:**
- Consumes: `pipeline.topics.ALL_TOPICS` (Task 4).
- Produces:
  - Compose project `melee` with services `postgres`, `lakekeeper-migrate`, `lakekeeper`, `lakekeeper-bootstrap`, `lakekeeper-warehouse`, `silo`, `create-bucket`, `kafka`, `kafka-init`, `kafka-ui`, `trino`, and `pipeline-tools` (profile `tools`).
  - Image `melee-pipeline:dev` with Python 3.12 venv at `/opt/venv`, Java 17, `SPARK_CONF_DIR=/opt/spark-conf`, `PYTHONPATH=/app`, `DAGSTER_HOME=/opt/dagster/home`, Ivy cache at `/opt/ivy`.
  - Spark catalog name `lakekeeper` (default catalog), warehouse `melee`. Trino catalog `lakekeeper`.
  - Named volume `checkpoints` mounted at `/checkpoints`.
  - Make targets: `up`, `down`, `nuke`, `ps`, `logs`, `check-lakehouse`.
  - Host ports: Silo 9000/9001, Lakekeeper 8181, Kafka 9094, Kafka UI 8085, Trino 8090.

- [ ] **Step 1: Write the failing config test**

`tests/pipeline/test_infra_config.py`:

```python
"""Keep infra config consistent with the Python code."""

import re
import tomllib
from pathlib import Path

from pipeline.topics import ALL_TOPICS

COMPOSE = Path("infra/docker-compose.yml").read_text(encoding="utf-8")
SPARK_DEFAULTS = Path("infra/spark/spark-defaults.conf").read_text(encoding="utf-8")


def test_compose_creates_every_topic():
    for topic in ALL_TOPICS:
        assert topic in COMPOSE, f"{topic} missing from kafka-init"


def test_kafka_connector_matches_pyspark_version():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    (pin,) = [d for d in pyproject["project"]["optional-dependencies"]["pipeline"] if d.startswith("pyspark")]
    pyspark_version = pin.split("==")[1]
    assert f"spark-sql-kafka-0-10_2.13:{pyspark_version}" in SPARK_DEFAULTS
    minor = ".".join(pyspark_version.split(".")[:2])
    assert f"iceberg-spark-runtime-{minor}_2.13" in SPARK_DEFAULTS


def test_compose_images_are_pinned():
    images = re.findall(r"^\s*image:\s*(\S+)", COMPOSE, flags=re.MULTILINE)
    assert images
    for image in images:
        if image == "melee-pipeline:dev":
            continue
        assert ":" in image and not image.endswith((":latest", ":latest-main")), image
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/pipeline/test_infra_config.py -v`
Expected: FAIL with `FileNotFoundError: ... infra/docker-compose.yml`.

- [ ] **Step 3: Postgres, Lakekeeper, Trino and Spark config files**

`infra/postgres/init-databases.sql`:

```sql
-- One Postgres instance serves both the Iceberg catalog and Dagster.
CREATE DATABASE lakekeeper;
CREATE DATABASE dagster;
```

`infra/lakekeeper/create-warehouse.json` (the `__ACCESS_KEY__`/`__SECRET_KEY__` placeholders are filled from the environment at startup):

```json
{
  "warehouse-name": "melee",
  "project-id": "00000000-0000-0000-0000-000000000000",
  "storage-profile": {
    "type": "s3",
    "bucket": "lakehouse",
    "key-prefix": "warehouse",
    "assume-role-arn": null,
    "endpoint": "http://silo:9000",
    "sts-endpoint": "http://silo:9000",
    "region": "local-01",
    "path-style-access": true,
    "flavor": "s3-compat",
    "sts-enabled": true
  },
  "storage-credential": {
    "type": "s3",
    "credential-type": "access-key",
    "access-key-id": "__ACCESS_KEY__",
    "secret-access-key": "__SECRET_KEY__"
  }
}
```

`infra/trino/catalog/lakekeeper.properties`:

```properties
connector.name=iceberg
iceberg.catalog.type=rest
iceberg.rest-catalog.uri=http://lakekeeper:8181/catalog
iceberg.rest-catalog.warehouse=melee
iceberg.rest-catalog.security=NONE
iceberg.rest-catalog.vended-credentials-enabled=true
fs.native-s3.enabled=true
s3.endpoint=http://silo:9000
s3.region=local-01
s3.path-style-access=true
```

`infra/spark/spark-defaults.conf`:

```properties
# Spark runs in local mode inside the pipeline container (see spec: memory budget).
spark.master                                   local[2]
spark.driver.memory                            1g
spark.sql.session.timeZone                     UTC
spark.ui.showConsoleProgress                   false

# Jars are resolved into /opt/ivy at image build time (warm_ivy_cache.py).
spark.jars.ivy                                 /opt/ivy
spark.jars.packages                            org.apache.iceberg:iceberg-spark-runtime-4.0_2.13:1.11.0,org.apache.iceberg:iceberg-aws-bundle:1.11.0,org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.4

# Iceberg tables live in the Lakekeeper REST catalog; Lakekeeper vends S3 credentials.
spark.sql.extensions                           org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions
spark.sql.defaultCatalog                       lakekeeper
spark.sql.catalog.lakekeeper                   org.apache.iceberg.spark.SparkCatalog
spark.sql.catalog.lakekeeper.type              rest
spark.sql.catalog.lakekeeper.uri               http://lakekeeper:8181/catalog
spark.sql.catalog.lakekeeper.warehouse         melee
spark.sql.catalog.lakekeeper.io-impl           org.apache.iceberg.aws.s3.S3FileIO
spark.sql.catalog.lakekeeper.header.X-Iceberg-Access-Delegation  vended-credentials
```

`infra/spark/warm_ivy_cache.py`:

```python
"""Resolve spark.jars.packages at image build so jobs start without downloading jars."""

from pyspark.sql import SparkSession

SparkSession.builder.master("local[1]").appName("warm-ivy-cache").getOrCreate().stop()
```

`infra/spark/check_lakehouse.py`:

```python
"""Check Spark -> Lakekeeper -> Silo end to end: create, write and read a table."""

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("check-lakehouse").getOrCreate()
spark.sql("CREATE NAMESPACE IF NOT EXISTS healthcheck")
spark.sql("CREATE OR REPLACE TABLE healthcheck.ping (id INT, note STRING) USING iceberg")
spark.sql("INSERT INTO healthcheck.ping VALUES (1, 'ok')")
rows = [tuple(r) for r in spark.sql("SELECT id, note FROM healthcheck.ping").collect()]
assert rows == [(1, "ok")], rows
print("spark lakehouse check ok:", rows)
spark.stop()
```

- [ ] **Step 4: Pipeline image**

`infra/pipeline.Dockerfile`:

```dockerfile
# One image for the Dagster code location, webserver, daemon and Spark jobs.
FROM python:3.12-slim-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends default-jre-headless curl procps \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH=/opt/venv/bin:$PATH \
    JAVA_HOME=/usr/lib/jvm/default-java \
    SPARK_CONF_DIR=/opt/spark-conf \
    PYSPARK_PYTHON=/opt/venv/bin/python \
    PYTHONPATH=/app \
    DAGSTER_HOME=/opt/dagster/home

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project --extra pipeline --extra dev

COPY infra/spark/spark-defaults.conf /opt/spark-conf/spark-defaults.conf
COPY infra/spark/warm_ivy_cache.py /tmp/warm_ivy_cache.py
RUN python /tmp/warm_ivy_cache.py && mkdir -p /opt/dagster/home

COPY pipeline ./pipeline
COPY tests ./tests
COPY infra/spark ./infra/spark
```

Append to `.dockerignore`:

```
data
docs
*.md
.git
```

(`.git` is already listed; leave the existing line and skip the duplicate.)

- [ ] **Step 5: Compose file**

`infra/docker-compose.yml`:

```yaml
name: melee

x-pipeline-image: &pipeline-image
  image: melee-pipeline:dev
  build:
    context: ..
    dockerfile: infra/pipeline.Dockerfile

x-db-env: &db-env
  POSTGRES_USER: ${POSTGRES_USER:-melee}
  POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-melee}

x-lakekeeper-env: &lakekeeper-env
  LAKEKEEPER__PG_ENCRYPTION_KEY: ${LAKEKEEPER_ENCRYPTION_KEY:-local-dev-only-not-a-secret}
  LAKEKEEPER__PG_DATABASE_URL_READ: postgresql://${POSTGRES_USER:-melee}:${POSTGRES_PASSWORD:-melee}@postgres:5432/lakekeeper
  LAKEKEEPER__PG_DATABASE_URL_WRITE: postgresql://${POSTGRES_USER:-melee}:${POSTGRES_PASSWORD:-melee}@postgres:5432/lakekeeper
  RUST_LOG: info

services:
  postgres:
    image: postgres:17.11
    environment:
      <<: *db-env
      POSTGRES_DB: postgres
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./postgres/init-databases.sql:/docker-entrypoint-initdb.d/01-init-databases.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $${POSTGRES_USER} -d postgres"]
      interval: 2s
      timeout: 5s
      retries: 30

  silo:
    # Maintained MinIO fork; serves S3 plus the STS endpoint Lakekeeper vends credentials from.
    image: pgsty/silo:RELEASE.2026-09-16T00-00-00Z
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: ${SILO_ROOT_USER:-melee-admin}
      MINIO_ROOT_PASSWORD: ${SILO_ROOT_PASSWORD:-melee-admin-password}
    volumes:
      - silo-data:/data
    ports:
      - "9000:9000"
      - "9001:9001"
    healthcheck:
      test: ["CMD", "mc", "ready", "local"]
      interval: 2s
      timeout: 10s
      retries: 30

  create-bucket:
    image: pgsty/silo:RELEASE.2026-09-16T00-00-00Z
    depends_on:
      silo:
        condition: service_healthy
    environment:
      SILO_ROOT_USER: ${SILO_ROOT_USER:-melee-admin}
      SILO_ROOT_PASSWORD: ${SILO_ROOT_PASSWORD:-melee-admin-password}
    entrypoint: /bin/sh
    command:
      - -c
      - mc alias set local http://silo:9000 "$$SILO_ROOT_USER" "$$SILO_ROOT_PASSWORD" && mc mb --ignore-existing local/lakehouse
    restart: "no"

  lakekeeper-migrate:
    image: quay.io/lakekeeper/catalog:v0.14.0
    command: ["migrate"]
    environment: *lakekeeper-env
    depends_on:
      postgres:
        condition: service_healthy
    restart: "no"

  lakekeeper:
    image: quay.io/lakekeeper/catalog:v0.14.0
    command: ["serve"]
    environment: *lakekeeper-env
    ports:
      - "8181:8181"
    depends_on:
      lakekeeper-migrate:
        condition: service_completed_successfully
      create-bucket:
        condition: service_completed_successfully
    healthcheck:
      test: ["CMD", "/home/nonroot/lakekeeper", "healthcheck"]
      interval: 2s
      timeout: 10s
      retries: 30

  lakekeeper-bootstrap:
    # Re-running on an already bootstrapped server returns an error status, which is fine.
    image: curlimages/curl:8.22.0
    depends_on:
      lakekeeper:
        condition: service_healthy
    command:
      - -sS
      - -X
      - POST
      - http://lakekeeper:8181/management/v1/bootstrap
      - -H
      - "Content-Type: application/json"
      - --data
      - '{"accept-terms-of-use": true}'
    restart: "no"

  lakekeeper-warehouse:
    image: curlimages/curl:8.22.0
    depends_on:
      lakekeeper-bootstrap:
        condition: service_completed_successfully
    environment:
      SILO_ROOT_USER: ${SILO_ROOT_USER:-melee-admin}
      SILO_ROOT_PASSWORD: ${SILO_ROOT_PASSWORD:-melee-admin-password}
    volumes:
      - ./lakekeeper/create-warehouse.json:/config/create-warehouse.json:ro
    entrypoint: /bin/sh
    command:
      - -c
      - |
        set -e
        if curl -sf http://lakekeeper:8181/management/v1/warehouse | grep -q '"melee"'; then
          echo "warehouse melee already exists"; exit 0
        fi
        sed -e "s|__ACCESS_KEY__|$$SILO_ROOT_USER|" -e "s|__SECRET_KEY__|$$SILO_ROOT_PASSWORD|" \
          /config/create-warehouse.json > /tmp/warehouse.json
        curl -sS --fail-with-body -X POST http://lakekeeper:8181/management/v1/warehouse \
          -H "Content-Type: application/json" --data @/tmp/warehouse.json
    restart: "no"

  kafka:
    image: apache/kafka:4.3.1
    environment:
      CLUSTER_ID: 4L6g3nShT-eMCtK--X86sw
      KAFKA_NODE_ID: 1
      KAFKA_PROCESS_ROLES: broker,controller
      KAFKA_LISTENERS: PLAINTEXT://:9092,CONTROLLER://:9093,EXTERNAL://:9094
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092,EXTERNAL://localhost:9094
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT,EXTERNAL:PLAINTEXT
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@kafka:9093
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "false"
      KAFKA_LOG_DIRS: /var/lib/kafka/data
      KAFKA_HEAP_OPTS: -Xms256m -Xmx512m
    volumes:
      - kafka-data:/var/lib/kafka/data
    ports:
      - "9094:9094"
    mem_limit: 1g
    healthcheck:
      test: ["CMD-SHELL", "/opt/kafka/bin/kafka-broker-api-versions.sh --bootstrap-server localhost:9092 > /dev/null"]
      interval: 5s
      timeout: 10s
      retries: 30

  kafka-init:
    image: apache/kafka:4.3.1
    depends_on:
      kafka:
        condition: service_healthy
    entrypoint: /bin/bash
    command:
      - -c
      - |
        set -e
        for topic in startgg.tournaments.raw startgg.sets.raw startgg.ingest.dlq; do
          /opt/kafka/bin/kafka-topics.sh --bootstrap-server kafka:9092 --create --if-not-exists \
            --topic "$$topic" --partitions 3 --replication-factor 1 \
            --config retention.ms=-1 --config cleanup.policy=delete
        done
    restart: "no"

  kafka-ui:
    image: kafbat/kafka-ui:v1.5.0
    environment:
      KAFKA_CLUSTERS_0_NAME: local
      KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS: kafka:9092
      JAVA_OPTS: -Xmx256m
    ports:
      - "8085:8080"
    mem_limit: 512m
    depends_on:
      kafka:
        condition: service_healthy

  trino:
    image: trinodb/trino:483
    volumes:
      - ./trino/catalog/lakekeeper.properties:/etc/trino/catalog/lakekeeper.properties:ro
    ports:
      - "8090:8080"
    mem_limit: 1536m
    depends_on:
      lakekeeper-warehouse:
        condition: service_completed_successfully
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:8080/v1/info | grep -q '\"starting\":false'"]
      interval: 5s
      timeout: 10s
      retries: 30

  pipeline-tools:
    # One-off commands: Spark tests, lakehouse checks, ad-hoc bronze runs.
    <<: *pipeline-image
    profiles: ["tools"]
    volumes:
      - ..:/app
      - checkpoints:/checkpoints
    depends_on:
      kafka-init:
        condition: service_completed_successfully
      lakekeeper-warehouse:
        condition: service_completed_successfully

volumes:
  postgres-data:
  silo-data:
  kafka-data:
  checkpoints:
```

- [ ] **Step 6: Run the config test**

Run: `uv run pytest tests/pipeline/test_infra_config.py -v`
Expected: 3 passed.

- [ ] **Step 7: Update `.env.example`**

Replace its contents with:

```
# Get your token at https://developer.start.gg/docs/authentication
STARTGG_API_TOKEN=your_token_here

# Local-only credentials for the Docker Compose stack. The defaults work as-is.
POSTGRES_USER=melee
POSTGRES_PASSWORD=melee
SILO_ROOT_USER=melee-admin
SILO_ROOT_PASSWORD=melee-admin-password
LAKEKEEPER_ENCRYPTION_KEY=local-dev-only-not-a-secret
```

- [ ] **Step 8: Add stack targets to the Makefile**

Insert after the `# ── Setup` block (the rest of the Makefile is converted to uv in Task 12):

```makefile
# ── Local stack ────────────────────────────────────────
COMPOSE := docker compose -f infra/docker-compose.yml --env-file .env

up:
	$(COMPOSE) up -d --build --wait

down:
	$(COMPOSE) down

nuke:  ## Stops the stack and DELETES all Kafka, Silo, Postgres and checkpoint data
	$(COMPOSE) down -v

ps:
	$(COMPOSE) ps -a

logs:
	$(COMPOSE) logs -f --tail=100

check-lakehouse:
	$(COMPOSE) run --rm pipeline-tools spark-submit infra/spark/check_lakehouse.py
	$(COMPOSE) exec -T trino trino --execute "SELECT * FROM lakekeeper.healthcheck.ping"
	$(COMPOSE) exec -T trino trino --execute "DROP TABLE lakekeeper.healthcheck.ping"
	$(COMPOSE) exec -T trino trino --execute "DROP SCHEMA lakekeeper.healthcheck"
```

Also add `SHELL := bash` as the first line of the Makefile, and add the new target names to the `.PHONY` line: `up down nuke ps logs check-lakehouse`.

- [ ] **Step 9: Bring the stack up and verify**

Run (open a new Git Bash so `make` is on PATH; copy `.env` from `.env.example` first if any of the new variables are missing — the defaults also work without them):

```bash
make up
make ps
```

Expected: `make up` exits 0. `make ps` shows `postgres`, `silo`, `lakekeeper`, `kafka`, `kafka-ui`, `trino` as `running (healthy)` or `running`, and `create-bucket`, `lakekeeper-migrate`, `lakekeeper-bootstrap`, `lakekeeper-warehouse`, `kafka-init` as `exited (0)`.

If `up --wait` fails only because one-shot services exited, change the `up` target to `$(COMPOSE) up -d --build` followed by `$(COMPOSE) up -d --wait trino kafka-ui` and re-run.

- [ ] **Step 10: Verify topics and the lakehouse round trip**

```bash
docker compose -f infra/docker-compose.yml exec -T kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server kafka:9092 --describe
make check-lakehouse
```

Expected: three topics, each `PartitionCount: 3` with `retention.ms=-1`. `check-lakehouse` prints `spark lakehouse check ok: [(1, 'ok')]`, then Trino prints `"1","ok"`, then `DROP TABLE` and `DROP SCHEMA`.

If Spark fails with an S3 credentials error, Lakekeeper is not vending credentials: confirm `lakekeeper-warehouse` logged a created warehouse with `"sts-enabled": true`, then `make nuke && make up` and retry.

- [ ] **Step 11: Verify a second `make up` is idempotent**

Run: `make down && make up`
Expected: exits 0; `lakekeeper-warehouse` logs `warehouse melee already exists`.

- [ ] **Step 12: Commit**

```bash
git add infra .env.example .dockerignore Makefile tests/pipeline/test_infra_config.py
git commit -m "Add local lakehouse stack: Silo, Lakekeeper, Postgres, Kafka, Trino"
```

---

### Task 8: Bronze transforms (pure Spark functions)

**Files:**
- Create: `pipeline/spark_jobs/bronze.py`, `tests/pipeline/conftest.py`
- Test: `tests/pipeline/test_bronze_transform.py`

**Interfaces:**
- Consumes: `TOURNAMENTS_TOPIC`, `SETS_TOPIC` (Task 4); `build_envelope` (Task 2, test only).
- Produces:
  - `ENVELOPE_SCHEMA`, `TOPIC_ENTITIES: dict[str, str]`
  - `parse_envelopes(kafka_df: DataFrame) -> DataFrame` (adds `env`, `ingested_ts`, `reject_reason`, keeps Kafka columns, `raw_key`, `raw_value`)
  - `bronze_rows(parsed: DataFrame, entity: str) -> DataFrame` with columns `BRONZE_COLUMNS`
  - `reject_rows(parsed: DataFrame) -> DataFrame` with columns `kafka_topic, kafka_partition, kafka_offset, kafka_timestamp, raw_key, raw_value, reject_reason, rejected_at`
  - `BRONZE_NAMESPACE = "bronze"`, `BRONZE_TABLES = {"tournament": "bronze.startgg_tournaments", "set": "bronze.startgg_sets"}`, `REJECTS_TABLE = "bronze.startgg_rejects"`
  - `bronze_table_ddl(table: str) -> str`, `REJECTS_DDL: str`
  - Reject reasons: `null_value`, `unparseable_envelope`, `missing_required_field`, `invalid_ingested_at`, `entity_topic_mismatch`

- [ ] **Step 1: Spark session fixture**

`tests/pipeline/conftest.py`:

```python
import pytest


@pytest.fixture(scope="session")
def spark():
    """Local SparkSession. Only Spark-marked tests use it; they need Java."""
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[1]")
        .appName("pipeline-tests")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    yield session
    session.stop()
```

- [ ] **Step 2: Write the failing tests**

`tests/pipeline/test_bronze_transform.py`:

```python
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
    assert reasons(spark, [(TOURNAMENTS_TOPIC, envelope(entity="set"))]) == ["entity_topic_mismatch"]


def test_rejects_keep_raw_bytes(spark):
    (row,) = reject_rows(parse_envelopes(kafka_df(spark, [(SETS_TOPIC, "garbage")]))).collect()
    assert bytes(row.raw_value) == b"garbage"
    assert bytes(row.raw_key) == b"k"
    assert row.rejected_at is not None


def test_rejected_rows_never_reach_bronze(spark):
    parsed = parse_envelopes(kafka_df(spark, [(SETS_TOPIC, "garbage"), (SETS_TOPIC, envelope())]))
    assert bronze_rows(parsed, "set").count() == 1
```

- [ ] **Step 3: Write the implementation**

`pipeline/spark_jobs/bronze.py`:

```python
"""Pure transforms from Kafka source rows to bronze rows and rejects, plus table DDL."""

from functools import reduce

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from pipeline.topics import SETS_TOPIC, TOURNAMENTS_TOPIC

ENVELOPE_FIELDS = (
    "entity",
    "entity_id",
    "ingested_at",
    "dagster_run_id",
    "partition_week",
    "query_name",
    "query_variables",
    "payload",
)
# Declaring nested objects as strings makes from_json keep their raw JSON text.
ENVELOPE_SCHEMA = StructType([StructField(name, StringType()) for name in ENVELOPE_FIELDS])
REQUIRED_FIELDS = ("entity", "entity_id", "ingested_at", "payload")

TOPIC_ENTITIES = {TOURNAMENTS_TOPIC: "tournament", SETS_TOPIC: "set"}

BRONZE_NAMESPACE = "bronze"
BRONZE_TABLES = {"tournament": "bronze.startgg_tournaments", "set": "bronze.startgg_sets"}
REJECTS_TABLE = "bronze.startgg_rejects"

KAFKA_COLUMNS = ("kafka_topic", "kafka_partition", "kafka_offset", "kafka_timestamp")
BRONZE_COLUMNS = (
    *KAFKA_COLUMNS,
    "kafka_key",
    "entity",
    "entity_id",
    "ingested_at",
    "dagster_run_id",
    "partition_week",
    "query_name",
    "query_variables",
    "payload",
)


def bronze_table_ddl(table: str) -> str:
    return f"""
        CREATE TABLE IF NOT EXISTS {table} (
            kafka_topic STRING, kafka_partition INT, kafka_offset BIGINT,
            kafka_timestamp TIMESTAMP, kafka_key STRING,
            entity STRING, entity_id STRING, ingested_at TIMESTAMP,
            dagster_run_id STRING, partition_week STRING,
            query_name STRING, query_variables STRING, payload STRING
        ) USING iceberg
        PARTITIONED BY (days(ingested_at))
    """


REJECTS_DDL = f"""
    CREATE TABLE IF NOT EXISTS {REJECTS_TABLE} (
        kafka_topic STRING, kafka_partition INT, kafka_offset BIGINT,
        kafka_timestamp TIMESTAMP, raw_key BINARY, raw_value BINARY,
        reject_reason STRING, rejected_at TIMESTAMP
    ) USING iceberg
    PARTITIONED BY (days(kafka_timestamp))
"""


def _any_null(columns: list[Column]) -> Column:
    return reduce(lambda a, b: a | b, [c.isNull() for c in columns])


def _all_null(columns: list[Column]) -> Column:
    return reduce(lambda a, b: a & b, [c.isNull() for c in columns])


def parse_envelopes(kafka_df: DataFrame) -> DataFrame:
    """Parse each Kafka value as an envelope and decide whether to reject it."""
    expected_entity = F.create_map(
        *[F.lit(x) for pair in TOPIC_ENTITIES.items() for x in pair]
    )[F.col("topic")]
    parsed = kafka_df.select(
        F.col("topic").alias("kafka_topic"),
        F.col("partition").alias("kafka_partition"),
        F.col("offset").alias("kafka_offset"),
        F.col("timestamp").alias("kafka_timestamp"),
        F.col("key").cast("string").alias("kafka_key"),
        F.col("key").alias("raw_key"),
        F.col("value").alias("raw_value"),
        F.from_json(F.col("value").cast("string"), ENVELOPE_SCHEMA).alias("env"),
        expected_entity.alias("expected_entity"),
    )
    env_fields = [F.col(f"env.{name}") for name in ENVELOPE_FIELDS]
    ingested_ts = F.try_to_timestamp(F.col("env.ingested_at"))
    reject_reason = (
        F.when(F.col("raw_value").isNull(), "null_value")
        .when(F.col("env").isNull() | _all_null(env_fields), "unparseable_envelope")
        .when(_any_null([F.col(f"env.{name}") for name in REQUIRED_FIELDS]), "missing_required_field")
        .when(ingested_ts.isNull(), "invalid_ingested_at")
        .when(F.col("env.entity") != F.col("expected_entity"), "entity_topic_mismatch")
    )
    return parsed.select("*", ingested_ts.alias("ingested_ts"), reject_reason.alias("reject_reason"))


def bronze_rows(parsed: DataFrame, entity: str) -> DataFrame:
    """Accepted rows for one entity, in bronze table column order."""
    return parsed.where(
        F.col("reject_reason").isNull() & (F.col("env.entity") == F.lit(entity))
    ).select(
        *KAFKA_COLUMNS,
        "kafka_key",
        F.col("env.entity").alias("entity"),
        F.col("env.entity_id").alias("entity_id"),
        F.col("ingested_ts").alias("ingested_at"),
        F.col("env.dagster_run_id").alias("dagster_run_id"),
        F.col("env.partition_week").alias("partition_week"),
        F.col("env.query_name").alias("query_name"),
        F.col("env.query_variables").alias("query_variables"),
        F.col("env.payload").alias("payload"),
    )


def reject_rows(parsed: DataFrame) -> DataFrame:
    """Rows that failed validation, with the raw bytes kept for inspection."""
    return parsed.where(F.col("reject_reason").isNotNull()).select(
        *KAFKA_COLUMNS,
        "raw_key",
        "raw_value",
        "reject_reason",
        F.current_timestamp().alias("rejected_at"),
    )
```

- [ ] **Step 4: Confirm the tests are skipped on the host by default**

Run: `uv run pytest tests/pipeline/test_bronze_transform.py -v`
Expected: all deselected (`-m 'not spark'`), exit code 5 ("no tests ran") is acceptable here.

- [ ] **Step 5: Add `test-spark` to the Makefile**

Under the `# ── Local stack` block:

```makefile
test-spark:
	$(COMPOSE) run --rm --no-deps pipeline-tools pytest -m spark -v tests/pipeline
```

Add `test-spark` to `.PHONY`.

- [ ] **Step 6: Run the Spark tests in the container**

Run: `make test-spark`
Expected: 11 passed. If `test_valid_set_becomes_bronze_row` fails because `payload` is `None`, `from_json` is not keeping nested objects as strings: change the `payload` and `query_variables` extraction to `F.get_json_object(F.col("value").cast("string"), "$.payload")` (and `"$.query_variables"`), keep `ENVELOPE_SCHEMA` for the other fields, and re-run.

- [ ] **Step 7: Commit**

```bash
git add pipeline/spark_jobs/bronze.py tests/pipeline/conftest.py tests/pipeline/test_bronze_transform.py Makefile
git commit -m "Add bronze transforms with envelope validation and rejects"
```

---

### Task 9: Bronze landing job

**Files:**
- Create: `pipeline/spark_jobs/bronze_job.py`
- Modify: `Makefile`

**Interfaces:**
- Consumes: everything from `pipeline.spark_jobs.bronze` (Task 8); `TOURNAMENTS_TOPIC`, `SETS_TOPIC`, `RAW_TOPICS` (Task 4).
- Produces: CLI `spark-submit pipeline/spark_jobs/bronze_job.py --bootstrap-servers HOST:PORT --checkpoint-root PATH`, whose **last stdout line** is JSON `{"tournaments": int, "sets": int, "rejects": int}` (rows added by this run). Checkpoint subdirectories: `tournaments`, `sets`, `rejects`.

- [ ] **Step 1: Write the job**

`pipeline/spark_jobs/bronze_job.py`:

```python
"""Land the raw Kafka topics into Iceberg bronze tables, then stop (availableNow).

Run with spark-submit. The last line of stdout is a JSON summary of rows added.
Each target table has its own streaming query and checkpoint, so every write
goes through Iceberg's exactly-once streaming sink.
"""

import argparse
import json
from datetime import UTC, datetime

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.streaming import StreamingQuery

from pipeline.spark_jobs.bronze import (
    BRONZE_NAMESPACE,
    BRONZE_TABLES,
    REJECTS_DDL,
    REJECTS_TABLE,
    bronze_rows,
    bronze_table_ddl,
    parse_envelopes,
    reject_rows,
)
from pipeline.topics import RAW_TOPICS, SETS_TOPIC, TOURNAMENTS_TOPIC


def read_kafka(spark: SparkSession, bootstrap_servers: str, topics: list[str]) -> DataFrame:
    return (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("subscribe", ",".join(topics))
        .option("startingOffsets", "earliest")
        .load()
    )


def write_once(df: DataFrame, table: str, checkpoint: str) -> StreamingQuery:
    return (
        df.writeStream.format("iceberg")
        .outputMode("append")
        .trigger(availableNow=True)
        .option("checkpointLocation", checkpoint)
        .option("fanout-enabled", "true")
        .toTable(table)
    )


def added_rows_since(spark: SparkSession, table: str, since: datetime) -> int:
    """Rows committed to a table since `since`, from Iceberg snapshot summaries."""
    row = spark.sql(
        f"""
        SELECT coalesce(sum(cast(summary['added-records'] AS BIGINT)), 0) AS n
        FROM {table}.snapshots
        WHERE committed_at >= TIMESTAMP '{since:%Y-%m-%d %H:%M:%S.%f}'
        """
    ).first()
    return int(row.n)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-servers", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    args = parser.parse_args(argv)

    spark = SparkSession.builder.appName("bronze-startgg").getOrCreate()
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {BRONZE_NAMESPACE}")
    for table in BRONZE_TABLES.values():
        spark.sql(bronze_table_ddl(table))
    spark.sql(REJECTS_DDL)

    started = datetime.now(UTC)
    servers, root = args.bootstrap_servers, args.checkpoint_root
    queries = [
        write_once(
            bronze_rows(parse_envelopes(read_kafka(spark, servers, [TOURNAMENTS_TOPIC])), "tournament"),
            BRONZE_TABLES["tournament"],
            f"{root}/tournaments",
        ),
        write_once(
            bronze_rows(parse_envelopes(read_kafka(spark, servers, [SETS_TOPIC])), "set"),
            BRONZE_TABLES["set"],
            f"{root}/sets",
        ),
        write_once(
            reject_rows(parse_envelopes(read_kafka(spark, servers, list(RAW_TOPICS)))),
            REJECTS_TABLE,
            f"{root}/rejects",
        ),
    ]
    for query in queries:
        query.awaitTermination()

    summary = {
        "tournaments": added_rows_since(spark, BRONZE_TABLES["tournament"], started),
        "sets": added_rows_since(spark, BRONZE_TABLES["set"], started),
        "rejects": added_rows_since(spark, REJECTS_TABLE, started),
    }
    spark.stop()
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add a Makefile target for ad-hoc runs**

Under `# ── Local stack`:

```makefile
bronze-once:
	$(COMPOSE) run --rm pipeline-tools spark-submit pipeline/spark_jobs/bronze_job.py \
		--bootstrap-servers kafka:9092 --checkpoint-root /checkpoints/bronze
```

Add `bronze-once` to `.PHONY`.

- [ ] **Step 3: Lint**

Run: `uv run ruff check pipeline/`
Expected: `All checks passed!`

- [ ] **Step 4: Run against empty topics**

With the stack up (`make up`):

Run: `make bronze-once`
Expected: last line `{"tournaments": 0, "sets": 0, "rejects": 0}`. Then:

```bash
docker compose -f infra/docker-compose.yml exec -T trino trino --execute "SHOW TABLES FROM lakekeeper.bronze"
```

Expected: `startgg_rejects`, `startgg_sets`, `startgg_tournaments`.

- [ ] **Step 5: Verify a reject and checkpoint resume**

```bash
echo 'k1:not-json' | docker compose -f infra/docker-compose.yml exec -T kafka \
  /opt/kafka/bin/kafka-console-producer.sh --bootstrap-server kafka:9092 \
  --topic startgg.sets.raw --property parse.key=true --property key.separator=:
make bronze-once
make bronze-once
```

Expected: the first `bronze-once` prints `{"tournaments": 0, "sets": 0, "rejects": 1}`; the second prints `{"tournaments": 0, "sets": 0, "rejects": 0}` (checkpoint resumed; nothing re-read).

- [ ] **Step 6: Reset the stack (the bad message is kept forever otherwise)**

Run: `make nuke && make up`
Expected: exits 0.

- [ ] **Step 7: Commit**

```bash
git add pipeline/spark_jobs/bronze_job.py Makefile
git commit -m "Add availableNow Spark job landing Kafka topics into Iceberg bronze"
```

---

### Task 10: Dagster definitions and services

**Files:**
- Create: `pipeline/dagster_defs/partitions.py`, `pipeline/dagster_defs/resources.py`, `pipeline/dagster_defs/assets.py`, `pipeline/dagster_defs/automation.py`, `infra/dagster/dagster.yaml`, `infra/dagster/workspace.yaml`
- Modify: `pipeline/dagster_defs/__init__.py`, `infra/docker-compose.yml`
- Test: `tests/pipeline/test_dagster_defs.py`

**Interfaces:**
- Consumes: `collect_week` (Task 4), `FixtureClient` (Task 5), `KafkaPublisher` (Task 6), `StartGGClient` (Task 3), bronze job CLI and summary format (Task 9), `FakePaginator`/`FakePublisher`/`tournament`/`melee_event` (Task 4 test fakes).
- Produces:
  - `WEEKLY_PARTITIONS` (Monday weeks from 2018-01-01, UTC, `end_offset=1` so the current week exists)
  - Resources: `StartGGResource(token: str, fixture_path: str | None = None, min_attendees: int = 50)` with `.get_client()`; `KafkaResource(bootstrap_servers: str = "kafka:9092")` with `.get_publisher()`; `SparkJobResource(job_path, bootstrap_servers, checkpoint_root)` with `.run_bronze_job() -> dict[str, int]`
  - Pure helpers: `build_bronze_command(job_path, bootstrap_servers, checkpoint_root) -> list[str]`, `parse_job_summary(stdout: str) -> dict[str, int]`
  - Assets: `startgg_raw` (pool `startgg_api`, check `no_dlq_messages`), `bronze_startgg` (pool `spark_bronze`, check `no_rejects`, retry 2× exponential)
  - Jobs: `ingest_job` (`ingest_startgg`), `bronze_job` (`land_bronze`); sensor `land_bronze_after_ingest`; schedule `refresh_recent_weeks`
  - `pipeline.dagster_defs.defs: dg.Definitions`
  - Compose services `dagster-code` (gRPC on 4000, hostname `dagster-code`), `dagster-webserver` (port 3000), `dagster-daemon`

- [ ] **Step 1: Write the failing tests**

`tests/pipeline/test_dagster_defs.py`:

```python
from datetime import datetime
from typing import ClassVar
from zoneinfo import ZoneInfo

import dagster as dg
import pytest

from pipeline.dagster_defs import defs
from pipeline.dagster_defs.assets import bronze_startgg, startgg_raw
from pipeline.dagster_defs.automation import land_bronze_after_ingest, refresh_recent_weeks
from pipeline.dagster_defs.partitions import WEEKLY_PARTITIONS
from pipeline.dagster_defs.resources import (
    KafkaResource,
    SparkJobResource,
    StartGGResource,
    build_bronze_command,
    parse_job_summary,
)
from pipeline.ingest.startgg import RetriesExhaustedError
from pipeline.topics import SETS_TOPIC
from tests.pipeline.fakes import FakePaginator, FakePublisher, melee_event, tournament


class FakeStartGG(StartGGResource):
    scenario: str = "ok"

    def get_client(self):
        sets = {10: [{"id": 100}, {"id": 101}]}
        if self.scenario == "dlq":
            sets = {10: RetriesExhaustedError("down")}
        return FakePaginator([tournament(1, events=[melee_event(10)])], sets)


class FakeKafka(KafkaResource):
    publishers: ClassVar[list[FakePublisher]] = []

    def get_publisher(self):
        publisher = FakePublisher()
        FakeKafka.publishers.append(publisher)
        return publisher


class FakeSpark(SparkJobResource):
    rejects: int = 0

    def run_bronze_job(self):
        return {"tournaments": 1, "sets": 2, "rejects": self.rejects}


def materialize_raw(scenario="ok", instance=None):
    return dg.materialize(
        [startgg_raw],
        partition_key="2025-01-06",
        resources={"startgg": FakeStartGG(token="unused", scenario=scenario), "kafka": FakeKafka()},
        instance=instance,
    )


def checks(result):
    return {e.check_name: e.passed for e in result.get_asset_check_evaluations()}


def test_definitions_load():
    dg.Definitions.validate_loadable(defs)


def test_partitions_are_mondays_from_2018():
    keys = WEEKLY_PARTITIONS.get_partition_keys(current_time=datetime(2018, 1, 20, tzinfo=ZoneInfo("UTC")))
    assert keys[0] == "2018-01-01"
    assert all(datetime.strptime(k, "%Y-%m-%d").weekday() == 0 for k in keys)


def test_startgg_raw_publishes_and_flushes():
    FakeKafka.publishers.clear()
    result = materialize_raw()
    assert result.success
    (publisher,) = FakeKafka.publishers
    assert publisher.flushed
    assert len(publisher.on(SETS_TOPIC)) == 2
    metadata = result.asset_materializations_for_node("startgg_raw")[0].metadata
    assert metadata["sets"].value == 2
    assert metadata["dlq_messages"].value == 0
    assert checks(result) == {"no_dlq_messages": True}


def test_dlq_fails_the_check_but_not_the_run():
    result = materialize_raw(scenario="dlq")
    assert result.success
    assert checks(result) == {"no_dlq_messages": False}


@pytest.mark.parametrize(("rejects", "passed"), [(0, True), (3, False)])
def test_bronze_check_reflects_rejects(rejects, passed):
    result = dg.materialize([bronze_startgg], resources={"spark_job": FakeSpark(rejects=rejects)})
    assert result.success
    assert checks(result) == {"no_rejects": passed}


def test_parse_job_summary_reads_last_line():
    stdout = 'some log line\n{"tournaments": 2, "sets": 30, "rejects": 0}\n'
    assert parse_job_summary(stdout) == {"tournaments": 2, "sets": 30, "rejects": 0}


@pytest.mark.parametrize("stdout", ["", "\n", '{"tournaments": 1}'])
def test_parse_job_summary_rejects_bad_output(stdout):
    with pytest.raises(ValueError):
        parse_job_summary(stdout)


def test_build_bronze_command():
    assert build_bronze_command("/job.py", "kafka:9092", "/ck") == [
        "spark-submit",
        "/job.py",
        "--bootstrap-servers",
        "kafka:9092",
        "--checkpoint-root",
        "/ck",
    ]


def test_schedule_requests_current_and_previous_week():
    context = dg.build_schedule_context(
        scheduled_execution_time=datetime(2025, 1, 8, 6, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
    )
    requests = refresh_recent_weeks(context)
    assert [r.partition_key for r in requests] == ["2024-12-30", "2025-01-06"]


def sensor_context(result, instance):
    return dg.build_run_status_sensor_context(
        sensor_name="land_bronze_after_ingest",
        dagster_event=result.get_run_success_event(),
        dagster_instance=instance,
        dagster_run=result.dagster_run,
    )


def test_sensor_requests_bronze_after_ingest():
    with dg.instance_for_test() as instance:
        result = materialize_raw(instance=instance)
        request = land_bronze_after_ingest(sensor_context(result, instance))
        assert isinstance(request, dg.RunRequest)
        assert request.run_key == result.run_id


def test_sensor_ignores_other_runs():
    with dg.instance_for_test() as instance:
        result = dg.materialize(
            [bronze_startgg], resources={"spark_job": FakeSpark()}, instance=instance
        )
        assert isinstance(land_bronze_after_ingest(sensor_context(result, instance)), dg.SkipReason)


def test_sensor_skips_when_bronze_run_already_queued():
    with dg.instance_for_test() as instance:
        result = materialize_raw(instance=instance)
        instance.add_run(
            dg.DagsterRun(job_name="land_bronze", run_id="queued-1", status=dg.DagsterRunStatus.QUEUED)
        )
        assert isinstance(land_bronze_after_ingest(sensor_context(result, instance)), dg.SkipReason)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_dagster_defs.py -v`
Expected: FAIL with `ImportError: cannot import name 'defs' from 'pipeline.dagster_defs'`.

- [ ] **Step 3: Partitions and resources**

`pipeline/dagster_defs/partitions.py`:

```python
"""Partition definitions shared by assets and automation."""

import dagster as dg

# Monday-to-Sunday weeks in UTC. end_offset=1 makes the in-progress week a
# partition too, so the daily schedule can refresh it.
WEEKLY_PARTITIONS = dg.WeeklyPartitionsDefinition(
    start_date="2018-01-01", day_offset=1, end_offset=1, timezone="UTC"
)
```

`pipeline/dagster_defs/resources.py`:

```python
"""Dagster resources wrapping the start.gg client, Kafka and the Spark job."""

import json
import subprocess

import dagster as dg

from pipeline.ingest.fixtures import FixtureClient
from pipeline.ingest.kafka_publisher import KafkaPublisher
from pipeline.ingest.startgg import StartGGClient

SUMMARY_KEYS = ("tournaments", "sets", "rejects")


class StartGGResource(dg.ConfigurableResource):
    """start.gg API access. With fixture_path set, replays a recorded week instead."""

    token: str
    fixture_path: str | None = None
    min_attendees: int = 50

    def get_client(self):
        if self.fixture_path:
            return FixtureClient(self.fixture_path)
        return StartGGClient(self.token)


class KafkaResource(dg.ConfigurableResource):
    bootstrap_servers: str = "kafka:9092"

    def get_publisher(self) -> KafkaPublisher:
        return KafkaPublisher(self.bootstrap_servers)


def build_bronze_command(job_path: str, bootstrap_servers: str, checkpoint_root: str) -> list[str]:
    return [
        "spark-submit",
        job_path,
        "--bootstrap-servers",
        bootstrap_servers,
        "--checkpoint-root",
        checkpoint_root,
    ]


def parse_job_summary(stdout: str) -> dict[str, int]:
    """Read the bronze job's JSON summary from the last non-empty stdout line."""
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Bronze job printed no summary")
    summary = json.loads(lines[-1])
    missing = set(SUMMARY_KEYS) - summary.keys()
    if missing:
        raise ValueError(f"Bronze job summary is missing {sorted(missing)}: {lines[-1]}")
    return {key: int(summary[key]) for key in SUMMARY_KEYS}


class SparkJobResource(dg.ConfigurableResource):
    job_path: str = "/app/pipeline/spark_jobs/bronze_job.py"
    bootstrap_servers: str = "kafka:9092"
    checkpoint_root: str = "/checkpoints/bronze"

    def run_bronze_job(self) -> dict[str, int]:
        command = build_bronze_command(self.job_path, self.bootstrap_servers, self.checkpoint_root)
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise dg.Failure(
                description=f"Bronze Spark job exited with code {result.returncode}",
                metadata={"stderr_tail": dg.MetadataValue.text(result.stderr[-5000:])},
            )
        return parse_job_summary(result.stdout)
```

- [ ] **Step 4: Assets**

`pipeline/dagster_defs/assets.py`:

```python
"""Ingestion and bronze landing assets."""

import dagster as dg

from pipeline.dagster_defs.partitions import WEEKLY_PARTITIONS
from pipeline.dagster_defs.resources import KafkaResource, SparkJobResource, StartGGResource
from pipeline.ingest.collect import collect_week


@dg.asset(
    partitions_def=WEEKLY_PARTITIONS,
    pool="startgg_api",
    group_name="ingest",
    kinds={"kafka", "python"},
    check_specs=[dg.AssetCheckSpec("no_dlq_messages", asset="startgg_raw")],
)
def startgg_raw(
    context: dg.AssetExecutionContext, startgg: StartGGResource, kafka: KafkaResource
) -> dg.MaterializeResult:
    """One week of start.gg tournaments and sets, published to the raw Kafka topics."""
    client = startgg.get_client()
    publisher = kafka.get_publisher()
    stats = collect_week(
        client,
        publisher,
        context.partition_key,
        run_id=context.run_id,
        min_attendees=startgg.min_attendees,
    )
    publisher.flush()
    return dg.MaterializeResult(
        metadata={
            "tournaments": stats.tournaments,
            "events": stats.events,
            "sets": stats.sets,
            "dlq_messages": stats.dlq_messages,
            "api_calls": client.api_calls,
        },
        check_results=[
            dg.AssetCheckResult(
                check_name="no_dlq_messages",
                passed=stats.dlq_messages == 0,
                metadata={"dlq_messages": stats.dlq_messages},
            )
        ],
    )


@dg.asset(
    deps=[startgg_raw],
    pool="spark_bronze",
    group_name="bronze",
    kinds={"spark", "iceberg"},
    retry_policy=dg.RetryPolicy(max_retries=2, delay=30, backoff=dg.Backoff.EXPONENTIAL),
    check_specs=[dg.AssetCheckSpec("no_rejects", asset="bronze_startgg")],
)
def bronze_startgg(spark_job: SparkJobResource) -> dg.MaterializeResult:
    """Everything new in the raw topics, landed into the Iceberg bronze tables."""
    summary = spark_job.run_bronze_job()
    return dg.MaterializeResult(
        metadata={
            "tournament_rows": summary["tournaments"],
            "set_rows": summary["sets"],
            "reject_rows": summary["rejects"],
        },
        check_results=[
            dg.AssetCheckResult(
                check_name="no_rejects",
                passed=summary["rejects"] == 0,
                metadata={"reject_rows": summary["rejects"]},
            )
        ],
    )
```

- [ ] **Step 5: Jobs, sensor and schedule**

`pipeline/dagster_defs/automation.py`:

```python
"""Jobs, the bronze sensor and the daily refresh schedule."""

import dagster as dg

from pipeline.dagster_defs.assets import bronze_startgg, startgg_raw
from pipeline.dagster_defs.partitions import WEEKLY_PARTITIONS

ingest_job = dg.define_asset_job(
    "ingest_startgg", selection=[startgg_raw], partitions_def=WEEKLY_PARTITIONS
)
bronze_job = dg.define_asset_job("land_bronze", selection=[bronze_startgg])

PENDING = [dg.DagsterRunStatus.QUEUED, dg.DagsterRunStatus.NOT_STARTED]


def _ingested_startgg(run: dg.DagsterRun) -> bool:
    # Backfills launched from the asset graph run as an implicit job, so check
    # the asset selection as well as the job name.
    if run.job_name == ingest_job.name:
        return True
    return bool(run.asset_selection) and startgg_raw.key in run.asset_selection


@dg.run_status_sensor(
    run_status=dg.DagsterRunStatus.SUCCESS,
    request_job=bronze_job,
    default_status=dg.DefaultSensorStatus.RUNNING,
)
def land_bronze_after_ingest(context: dg.RunStatusSensorContext):
    """Land bronze after each successful ingestion run.

    Bronze runs share one streaming checkpoint, so a run that is already
    queued will pick up these messages too; don't queue another.
    """
    if not _ingested_startgg(context.dagster_run):
        return dg.SkipReason("Not an ingestion run")
    pending = context.instance.get_runs(
        filters=dg.RunsFilter(job_name=bronze_job.name, statuses=PENDING), limit=1
    )
    if pending:
        return dg.SkipReason("A bronze run is already queued")
    return dg.RunRequest(run_key=context.dagster_run.run_id)


@dg.schedule(
    job=ingest_job,
    cron_schedule="0 6 * * *",
    execution_timezone="America/Los_Angeles",
    default_status=dg.DefaultScheduleStatus.RUNNING,
)
def refresh_recent_weeks(context: dg.ScheduleEvaluationContext) -> list[dg.RunRequest]:
    """Re-ingest the current and previous week to pick up newly finished tournaments."""
    keys = WEEKLY_PARTITIONS.get_partition_keys(current_time=context.scheduled_execution_time)[-2:]
    day = context.scheduled_execution_time.strftime("%Y-%m-%d")
    return [dg.RunRequest(run_key=f"{key}@{day}", partition_key=key) for key in keys]
```

- [ ] **Step 6: Definitions**

Replace `pipeline/dagster_defs/__init__.py`:

```python
"""Dagster definitions for the pipeline."""

import os

import dagster as dg

from pipeline.dagster_defs.assets import bronze_startgg, startgg_raw
from pipeline.dagster_defs.automation import (
    bronze_job,
    ingest_job,
    land_bronze_after_ingest,
    refresh_recent_weeks,
)
from pipeline.dagster_defs.resources import KafkaResource, SparkJobResource, StartGGResource

defs = dg.Definitions(
    assets=[startgg_raw, bronze_startgg],
    jobs=[ingest_job, bronze_job],
    sensors=[land_bronze_after_ingest],
    schedules=[refresh_recent_weeks],
    resources={
        # EnvVar keeps the token out of the config Dagster shows in the UI.
        "startgg": StartGGResource(
            token=dg.EnvVar("STARTGG_API_TOKEN"),
            fixture_path=os.getenv("STARTGG_FIXTURE_PATH") or None,
        ),
        "kafka": KafkaResource(),
        "spark_job": SparkJobResource(),
    },
)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_dagster_defs.py -v`
Expected: 15 passed. If `test_sensor_requests_bronze_after_ingest` fails because `result.dagster_run.asset_selection` is empty for `materialize()` runs, assert on `result.dagster_run.job_name` in the test output and add that job name check to `_ingested_startgg` only if it is specific to `startgg_raw`; do not loosen the filter to accept every run.

- [ ] **Step 8: Validate with the Dagster CLI**

Run: `uv run dagster definitions validate -m pipeline.dagster_defs`
Expected: a success message, exit 0.

- [ ] **Step 9: Dagster instance and workspace config**

`infra/dagster/dagster.yaml`:

```yaml
storage:
  postgres:
    postgres_db:
      hostname: postgres
      port: 5432
      db_name: dagster
      username:
        env: POSTGRES_USER
      password:
        env: POSTGRES_PASSWORD

concurrency:
  pools:
    # startgg_api: the API rate limit is shared, so one ingestion run at a time.
    # spark_bronze: bronze runs share streaming checkpoints, so one at a time.
    default_limit: 1
    granularity: run

run_monitoring:
  enabled: true
  free_slots_after_run_end_seconds: 300

telemetry:
  enabled: false
```

`infra/dagster/workspace.yaml`:

```yaml
load_from:
  - grpc_server:
      host: dagster-code
      port: 4000
      location_name: melee
```

- [ ] **Step 10: Add Dagster services to Compose**

In `infra/docker-compose.yml`, add this anchor after `x-lakekeeper-env`:

```yaml
x-dagster: &dagster
  <<: *pipeline-image
  environment:
    <<: *db-env
    DAGSTER_HOME: /opt/dagster/home
  volumes:
    - ./dagster/dagster.yaml:/opt/dagster/home/dagster.yaml:ro
    - ./dagster/workspace.yaml:/opt/dagster/workspace.yaml:ro
```

and these services before `pipeline-tools`:

```yaml
  dagster-code:
    <<: *dagster
    hostname: dagster-code
    command: ["dagster", "code-server", "start", "-h", "0.0.0.0", "-p", "4000", "-m", "pipeline.dagster_defs"]
    env_file:
      - path: ../.env
        required: false
    environment:
      <<: *db-env
      DAGSTER_HOME: /opt/dagster/home
      STARTGG_FIXTURE_PATH: ${STARTGG_FIXTURE_PATH:-}
    volumes:
      - ./dagster/dagster.yaml:/opt/dagster/home/dagster.yaml:ro
      - checkpoints:/checkpoints
    mem_limit: 3g
    depends_on:
      postgres:
        condition: service_healthy
      kafka-init:
        condition: service_completed_successfully
      lakekeeper-warehouse:
        condition: service_completed_successfully
    healthcheck:
      test: ["CMD", "dagster", "api", "grpc-health-check", "-p", "4000"]
      interval: 5s
      timeout: 10s
      retries: 30

  dagster-webserver:
    <<: *dagster
    command: ["dagster-webserver", "-h", "0.0.0.0", "-p", "3000", "-w", "/opt/dagster/workspace.yaml"]
    ports:
      - "3000:3000"
    mem_limit: 768m
    depends_on:
      dagster-code:
        condition: service_healthy

  dagster-daemon:
    <<: *dagster
    command: ["dagster-daemon", "run", "-w", "/opt/dagster/workspace.yaml"]
    mem_limit: 768m
    depends_on:
      dagster-code:
        condition: service_healthy
```

Runs launched by the daemon execute inside `dagster-code` (the default run launcher runs them on the gRPC code server), which is where Spark, the checkpoint volume and the token are.

- [ ] **Step 11: Verify in the running stack with the real API**

```bash
make up
docker compose -f infra/docker-compose.yml exec -T dagster-code dagster instance concurrency get --all
```

Open http://localhost:3000. Expected: assets `startgg_raw` and `bronze_startgg`, sensor `land_bronze_after_ingest` and schedule `refresh_recent_weeks` both running.

From the UI, materialize `startgg_raw` for partition `2025-01-06`. Expected within a few minutes:
- the run succeeds and its `no_dlq_messages` check passes;
- Kafka UI (http://localhost:8085) shows messages on `startgg.tournaments.raw` and `startgg.sets.raw`;
- the sensor launches a `land_bronze` run that succeeds with `no_rejects` passing;
- `docker compose -f infra/docker-compose.yml exec -T trino trino --execute "SELECT count(*) FROM lakekeeper.bronze.startgg_sets"` returns the run's `sets` metadata value.

If pool limits show as unset, run `docker compose -f infra/docker-compose.yml exec -T dagster-code dagster instance concurrency set startgg_api 1` and the same for `spark_bronze`, then add those two commands to a one-shot `dagster-pools` service so they persist across `make nuke`.

- [ ] **Step 12: Commit**

```bash
git add pipeline/dagster_defs infra/dagster infra/docker-compose.yml tests/pipeline/test_dagster_defs.py
git commit -m "Add Dagster ingestion and bronze assets, checks, sensor and schedule"
```

---

### Task 11: End-to-end smoke test

**Files:**
- Create: `scripts/smoke.sh`, `tests/smoke/__init__.py`, `tests/smoke/check_bronze.py`, `infra/docker-compose.smoke.yml`
- Modify: `Makefile`

**Interfaces:**
- Consumes: fixture `tests/fixtures/startgg/week_2025-01-06.json` (Task 5), `FixtureClient`, `collect_week` (Tasks 4–5), Dagster asset names (Task 10), Trino catalog `lakekeeper` (Task 7).
- Produces: `make smoke`, used by CI in Task 12.

- [ ] **Step 1: Compose override without host ports**

`infra/docker-compose.smoke.yml` (lets the smoke stack run while the dev stack is up):

```yaml
services:
  silo:
    ports: !reset []
  lakekeeper:
    ports: !reset []
  kafka:
    ports: !reset []
  kafka-ui:
    ports: !reset []
  trino:
    ports: !reset []
  dagster-webserver:
    ports: !reset []
```

- [ ] **Step 2: Bronze assertions**

`tests/smoke/__init__.py`: empty file.

`tests/smoke/check_bronze.py`:

```python
"""Assert bronze contents after the smoke test's ingestion runs.

Usage: python -m tests.smoke.check_bronze --fixture PATH --week KEY --copies N
"""

import argparse

import trino

from pipeline.ingest.collect import collect_week
from pipeline.ingest.fixtures import FixtureClient


class CountingPublisher:
    def publish(self, topic: str, key: str, value: dict) -> None:
        pass


def scalar_row(cursor, sql: str) -> tuple:
    cursor.execute(sql)
    return tuple(cursor.fetchone())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--week", required=True)
    parser.add_argument("--copies", type=int, required=True)
    args = parser.parse_args()

    expected = collect_week(FixtureClient(args.fixture), CountingPublisher(), args.week, run_id="check")
    cursor = trino.dbapi.connect(
        host="trino", port=8080, user="smoke", catalog="lakekeeper", schema="bronze"
    ).cursor()

    for table, unique in (("startgg_tournaments", expected.tournaments), ("startgg_sets", expected.sets)):
        rows, distinct = scalar_row(cursor, f"SELECT count(*), count(DISTINCT entity_id) FROM {table}")
        assert rows == unique * args.copies, f"{table}: {rows} rows, expected {unique * args.copies}"
        assert distinct == unique, f"{table}: {distinct} distinct ids, expected {unique}"
        print(f"{table}: {rows} rows, {distinct} distinct ids")

    (rejects,) = scalar_row(cursor, "SELECT count(*) FROM startgg_rejects")
    assert rejects == 0, f"{rejects} rejected messages"
    print("startgg_rejects: 0 rows")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke script**

`scripts/smoke.sh`:

```bash
#!/usr/bin/env bash
# End-to-end smoke test: recorded week -> Kafka -> bronze -> Trino, twice.
# Uses a separate Compose project with no host ports and no daemon, so it can
# run next to the dev stack and nothing else launches runs mid-test.
set -euo pipefail

WEEK=2025-01-06
FIXTURE=/app/tests/fixtures/startgg/week_${WEEK}.json
COMPOSE=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.smoke.yml -p melee-smoke)

cleanup() { "${COMPOSE[@]}" down -v --remove-orphans >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

"${COMPOSE[@]}" up -d --build --wait dagster-code trino

in_code() {
  "${COMPOSE[@]}" exec -T \
    -e STARTGG_FIXTURE_PATH="$FIXTURE" -e STARTGG_API_TOKEN=unused-in-fixture-mode \
    dagster-code "$@"
}

for copies in 1 2; do
  echo "--- ingestion pass $copies"
  in_code dagster asset materialize -m pipeline.dagster_defs --select startgg_raw --partition "$WEEK"
  in_code dagster asset materialize -m pipeline.dagster_defs --select bronze_startgg
  in_code python -m tests.smoke.check_bronze --fixture "$FIXTURE" --week "$WEEK" --copies "$copies"
done

echo "smoke test passed"
```

- [ ] **Step 4: Makefile target**

Under `# ── Local stack`:

```makefile
smoke:
	bash scripts/smoke.sh
```

Add `smoke` to `.PHONY`.

- [ ] **Step 5: Run it**

Run: `make smoke`
Expected: pass 1 prints `startgg_tournaments: T rows, T distinct ids`, `startgg_sets: S rows, S distinct ids`, `startgg_rejects: 0 rows`; pass 2 prints `2T` and `2S` rows with the same distinct counts; ends with `smoke test passed`. The `melee-smoke` project is removed afterwards (`docker compose -p melee-smoke ps -a` shows nothing).

- [ ] **Step 6: Commit**

```bash
git add scripts/smoke.sh tests/smoke infra/docker-compose.smoke.yml Makefile
git commit -m "Add end-to-end smoke test from recorded week to Trino"
```

---

### Task 12: CI, Makefile on uv, pre-commit and README

**Files:**
- Modify: `.github/workflows/ci.yml`, `Makefile`, `.pre-commit-config.yaml`, `README.md`

**Interfaces:**
- Consumes: `make smoke` (Task 11), Spark tests (Task 8), `dagster definitions validate` (Task 10).

- [ ] **Step 1: Replace the CI workflow**

`.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [master, main]
  pull_request:

jobs:
  lint-and-unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true
      - name: Install dependencies
        run: uv sync --frozen --all-extras
      - name: Lint
        run: uv run ruff check src/ tests/ app.py pipeline/ scripts/
      - name: Unit and Dagster tests
        run: uv run pytest -v
      - name: Validate Dagster definitions
        run: uv run dagster definitions validate -m pipeline.dagster_defs

  spark-and-smoke:
    if: github.event_name == 'pull_request' || github.ref == 'refs/heads/master'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-java@v4
        with:
          distribution: temurin
          java-version: "17"
      - uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true
      - name: Install dependencies
        run: uv sync --frozen --all-extras
      - name: Spark tests
        run: uv run pytest -m spark -v tests/pipeline
      - name: Smoke test
        run: |
          cp .env.example .env
          make smoke
```

- [ ] **Step 2: Convert the remaining Makefile targets to uv**

Replace these targets' recipes (leave `docker-build`, `docker-run`, `clean` as they are):

```makefile
install:
	uv sync --all-extras

collect:
	uv run python -m src.collect --start-date 2018-01-01 --min-attendees 50

features:
	uv run python -m src.features

validate:
	uv run python -m src.validation

train:
	uv run python -m src.model

tune:
	uv run python -m src.tuning --n-trials 50

export:
	uv run python -m src.export_app_data

app:
	uv run streamlit run app.py

test:
	uv run pytest -v

lint:
	uv run ruff check src/ tests/ app.py pipeline/ scripts/

format:
	uv run ruff format src/ tests/ app.py pipeline/ scripts/
```

Make sure `.PHONY` lists every target, including `format`, `up`, `down`, `nuke`, `ps`, `logs`, `check-lakehouse`, `test-spark`, `bronze-once`, `smoke`.

- [ ] **Step 3: Pre-commit adjustments**

In `.pre-commit-config.yaml`, change the `check-yaml` hook to allow Compose's `!reset` tag, and exempt recorded fixtures from the size limit:

```yaml
      - id: check-yaml
        args: ['--unsafe']
      - id: check-added-large-files
        args: ['--maxkb=500']
        exclude: ^tests/fixtures/
```

- [ ] **Step 4: README section**

Insert after the "Key Findings" list in `README.md`:

````markdown
## Data Platform (in progress)

The project is being rebuilt as a data engineering platform. Every record
enters through Kafka and lands in an Iceberg lakehouse:

```
start.gg API --(Dagster, weekly partitions)--> Kafka --(Spark, availableNow)--> Iceberg bronze --> Trino
```

| Component | Tool | Local URL |
|---|---|---|
| Orchestration | Dagster | http://localhost:3000 |
| Event backbone | Kafka (KRaft) + Kafka UI | http://localhost:8085 |
| Table catalog | Lakekeeper (Iceberg REST) | http://localhost:8181 |
| Object storage | Silo (S3-compatible) | http://localhost:9001 |
| Processing | Spark 4.0 (local mode) | — |
| SQL | Trino | `localhost:8090` |

**Run it** (needs Docker with ~6 GB of memory and GNU make):

```bash
cp .env.example .env      # then set STARTGG_API_TOKEN
make up                   # build and start the stack
make check-lakehouse      # Spark -> Iceberg -> Trino round trip
```

Then open Dagster and backfill `startgg_raw` from the asset page. The
`land_bronze_after_ingest` sensor lands each finished week into
`bronze.startgg_tournaments` and `bronze.startgg_sets`.

**Tests:** `make test` (unit and Dagster), `make test-spark` (Spark
transforms in the pipeline container), `make smoke` (end to end against a
recorded week; no API token needed).

**Failure handling:** events that still fail after retries go to the
`startgg.ingest.dlq` topic and fail the `no_dlq_messages` asset check;
malformed messages go to `bronze.startgg_rejects` and fail `no_rejects`.

Design: [`docs/superpowers/specs/2026-10-07-ingestion-foundation-design.md`](docs/superpowers/specs/2026-10-07-ingestion-foundation-design.md)
````

- [ ] **Step 5: Run the full local check**

```bash
make lint
make test
make test-spark
make smoke
```

Expected: ruff passes; `make test` shows the 20 original tests plus all pipeline tests passing; Spark tests pass; smoke ends with `smoke test passed`.

- [ ] **Step 6: Commit and push the branch**

```bash
git add .github/workflows/ci.yml Makefile .pre-commit-config.yaml README.md
git commit -m "Run CI on uv with Spark and smoke jobs; document the data platform"
git push -u origin de/ingestion-foundation
```

Pushing goes through Git Credential Manager as `justinko157` (folder-scoped config). Then open a pull request and confirm both CI jobs pass:

```bash
GH_TOKEN=$(gh auth token --user justinko157) gh pr create --fill --base master
GH_TOKEN=$(gh auth token --user justinko157) gh pr checks --watch
```

Expected: `lint-and-unit` and `spark-and-smoke` both pass. Do not add a Claude co-author line or a "Generated with Claude Code" line to the PR body.

---

### Task 13: 2025 backfill (done criterion)

**Files:** none (operational).

- [ ] **Step 1: Start the stack with the real token**

Run: `make up`
Expected: all services healthy; `.env` has the working token.

- [ ] **Step 2: Launch the backfill**

In Dagster (http://localhost:3000) open `startgg_raw` → **Materialize** → select partitions from `2025-01-06` to the current week → **Launch backfill**.

Expected: runs queue and execute one at a time (`startgg_api` pool); each successful run triggers at most one queued `land_bronze` run.

- [ ] **Step 3: Monitor**

Expected while running: Kafka UI message counts grow; any partition with `no_dlq_messages` failing is listed in the asset's checks tab. At ~80 requests/minute, a full 2025-to-now backfill takes several hours.

- [ ] **Step 4: Verify in Trino when the backfill finishes**

```bash
docker compose -f infra/docker-compose.yml exec -T trino trino --execute "
SELECT partition_week, count(DISTINCT entity_id) AS sets
FROM lakekeeper.bronze.startgg_sets
GROUP BY partition_week ORDER BY partition_week"
```

Expected: one row per week from `2025-01-06` onward with non-zero set counts for weeks that had eligible tournaments. Re-run failed DLQ partitions from the UI; they are safe to repeat.

- [ ] **Step 5: Record the result**

Note the totals (tournaments, sets, DLQ partitions, wall-clock time) in the PR description.
