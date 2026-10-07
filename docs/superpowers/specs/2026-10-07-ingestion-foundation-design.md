# Ingestion Foundation — Design

**Date:** 2026-10-07
**Status:** Approved in brainstorming, pending written-spec review
**Sub-project:** 1 of 4 in the data engineering pivot

## Context and intent

melee-matchup is pivoting from a data science project to a data engineering
project. Its job is to give hands-on experience with tools that the
companion project, `healthcare-data-platform` (Airflow, dbt, DuckDB/Snowflake,
Terraform for Snowflake, Grafana), does not cover:

- Kafka streaming
- a Spark + Iceberg lakehouse
- Dagster
- AWS (near-free: S3, Glue, Athena)

The existing ML model and Streamlit app stay and become consumers of the
pipeline's gold tables. The original SQLite dataset (`data/raw/melee.db`) is
not available, so all data must be re-collected from the start.gg API.

### Overall architecture (approach A)

Every record enters the platform through Kafka, whether it comes from a batch
backfill, the live tournament poller or a historical replay. Spark lands Kafka
into Iceberg bronze, and Spark batch jobs build silver and gold. Replaying
history means re-reading a topic.

### Sub-project sequence

1. **Ingestion foundation** (this spec): local stack, Dagster-partitioned API
   ingestion into Kafka, Spark landing Kafka into Iceberg bronze.
2. **Silver and gold, plus the ML switch:** Spark transforms (deduplication,
   cleaning, SCD Type 2 for player tags), Elo and features in gold, asset
   checks for data quality. The model and app read gold instead of SQLite.
3. **Live streaming:** live tournament poller and replay producer, with a
   continuous Spark stream keeping Elo live in the app.
4. **AWS:** Terraform for S3, Glue catalog and Athena, with the same code
   pointed at AWS.

Each sub-project gets its own spec, plan and build cycle.

## Goals

- `make up` starts the full local stack.
- A Dagster backfill from 2025-01-01 onward lands real start.gg data in
  Iceberg bronze tables.
- Trino can query the bronze tables.
- Unit, Spark and end-to-end smoke tests pass locally and in CI.

## Non-goals

- Silver and gold tables, deduplication, Elo or features (sub-project 2).
- Changing `src/`, the model or the Streamlit app (sub-project 2).
- In-progress (live) tournaments. The existing tournament query filters on
  `past: true`, so only finished tournaments are ingested (sub-project 3).
- A schema registry or Avro/Protobuf. Messages are JSON.
- Any AWS resources (sub-project 4).

## Section 1: Services

All services run in Docker Compose, defined under `infra/`.

| Service | Role |
|---|---|
| MinIO | S3-compatible object storage for Iceberg data files and Spark checkpoints |
| Lakekeeper | Iceberg REST catalog, backed by Postgres |
| Postgres | One instance, two databases: Lakekeeper's catalog and Dagster's run and event storage |
| Kafka | Single broker in KRaft mode, official `apache/kafka` image |
| Kafka UI | Web UI for browsing topics and messages |
| Spark | Standalone cluster: one master, one worker. Jobs are launched with `spark-submit` against the master. |
| Trino | SQL over the Iceberg tables through the Lakekeeper REST catalog |
| Dagster | Webserver, daemon and one user code location container |

Spark uses the 4.0.x line with the matching `iceberg-spark-runtime` and the
Spark-Kafka connector. Exact image tags and versions are pinned in the
implementation plan.

The stack needs about 8–10 GB of RAM for Docker Desktop. The fallback, if
memory is tight, is to run Spark in local mode inside the Dagster code
container. That fallback is not built unless needed.

### Code layout

```
pipeline/
  ingest/        start.gg client, queries, envelope builder, Kafka producer
  spark_jobs/    Kafka -> bronze job and its pure transform function
  dagster_defs/  assets, resources, sensors, schedules, asset checks
infra/           docker-compose.yml, service config (Trino catalog, Spark conf, Lakekeeper bootstrap)
tests/           unit, Spark and Dagster tests; fixtures/ holds recorded API responses
```

The existing `src/` package is left untouched in this sub-project.
`pipeline/ingest/` reuses the rate limiting and pagination logic from
`src/api_client.py`, with the changes described in Section 3.

The project pins Python 3.12 with `.python-version`, because the PySpark and
Dagster releases used must support it. The pin replaces the 3.14 virtual
environment that `uv sync` created on first setup.

## Section 2: Data flow

```
start.gg API --(Dagster asset, weekly partition)--> Kafka topics (raw JSON)
Kafka --(Spark job, availableNow trigger, launched by Dagster)--> Iceberg bronze on MinIO
Iceberg <-- Trino (ad-hoc SQL)        Kafka <-- Kafka UI (inspection)
```

### Ingestion asset: `startgg_raw`

- Partitioned with a Dagster weekly partitions definition starting
  2018-01-01. Backfills begin with partitions from 2025-01-01 onward and widen
  later.
- Each partition queries tournaments whose start falls in that week, using the
  existing `TOURNAMENTS_BY_GAME` query with `afterDate` and `beforeDate` set to
  the week's bounds. The 90-day chunking in `src/collector.py` is not needed,
  because one week stays far below the 10,000-result cap.
- The existing filters apply: offline tournaments only, minimum attendees
  configurable with a default of 50, Melee events only (videogame ID 1).
- For each Melee event, the asset pages through sets with the existing
  `EVENT_SETS` query.
- All API calls share one rate limiter. A Dagster concurrency pool limits
  ingestion to one partition run at a time, so backfills queue.
- A daily schedule materializes the current and previous week's partitions,
  picking up tournaments that finished since the last run.
- Each run records materialization metadata: messages published per topic,
  API calls made and DLQ messages published.

### Kafka topics

| Topic | Key | One message per |
|---|---|---|
| `startgg.tournaments.raw` | tournament ID | tournament, including its events list |
| `startgg.sets.raw` | set ID | set |
| `startgg.ingest.dlq` | entity ID | failed tournament or event (Section 3) |

- Each topic has 3 partitions. Keying by ID keeps every version of a record in
  one partition, in order.
- Retention is unlimited (`retention.ms=-1`, `cleanup.policy=delete`). The
  full history is a few GB, and approach A depends on replaying from Kafka.
- The producer uses `acks=all` and `enable.idempotence=true`.
- Topics are created by an init step in Compose, not by auto-creation.

### Message envelope

Every message value is JSON with these fields:

| Field | Description |
|---|---|
| `entity` | `tournament` or `set` on raw topics; `tournament` or `event` on the DLQ |
| `entity_id` | start.gg ID as a string |
| `ingested_at` | UTC ISO-8601 timestamp of when the record was fetched |
| `dagster_run_id` | run that produced the message |
| `partition_week` | partition key, e.g. `2025-03-03` |
| `query_name` | GraphQL operation name |
| `query_variables` | variables used, without pagination fields |
| `payload` | the unchanged API node for this entity |

DLQ messages use the same envelope with `entity` set to the failed entity's
type (`tournament` or `event`), `payload` set to `null`, and two extra fields: `error_type` and
`error_message`.

### Bronze landing job

- A Spark Structured Streaming job reads both raw topics with the
  `availableNow` trigger: it processes everything new, commits and stops.
- The checkpoint lives in MinIO, so each run continues from the last committed
  offsets.
- Tables, in the Iceberg namespace `bronze`:
  - `bronze.startgg_tournaments`
  - `bronze.startgg_sets`
  - `bronze.startgg_rejects`
- Columns for the first two: `kafka_topic`, `kafka_partition`,
  `kafka_offset`, `kafka_timestamp`, `kafka_key`, the envelope fields, and
  `payload` stored as a JSON string. Tables are partitioned by
  `days(ingested_at)`.
- The DLQ topic is not landed in bronze in this sub-project; it is inspected
  through Kafka UI and counted by the asset check.
- A Dagster sensor launches the job after each successful `startgg_raw` run.
  The job is a Dagster asset, `bronze_startgg`, so its runs appear in the
  same lineage graph.

### Duplicates

Bronze is append-only and keeps duplicates. Re-running a week publishes the
same records again, and both copies land in bronze as an audit trail.
Deduplication (latest version per entity ID) happens in silver in
sub-project 2. The checkpoint and Iceberg's atomic commits mean the Spark job
never writes the same Kafka offset twice, including after a crash mid-run.

## Section 3: Failure handling

### start.gg API

- A 400 or 401 response saying the token is invalid fails the run
  immediately with a message that says to replace `STARTGG_API_TOKEN`. It is
  not retried.
- 429, 5xx, timeouts and connection errors are retried with exponential
  backoff. (The current client does not retry 5xx.)
- GraphQL complexity errors still halve the page size and retry, as today.
- Hitting the 10,000-result pagination cap is treated as a failure for that
  event and sent to the DLQ, instead of silently returning partial results.

### Dead-letter queue

- When a tournament or event still fails after retries, a DLQ message is
  published and the run continues with the next one. One broken event does
  not block a whole week.
- The asset check `startgg_raw_no_dlq_messages` fails for a partition run
  that published any DLQ messages, and reports the count.
- Retrying means re-running the partition, which is safe because bronze
  tolerates duplicates.

### Kafka producer

The run flushes the producer at the end and fails if any message was not
delivered. Re-running the partition is the fix.

### Spark bronze job

- Dagster retries the asset twice with backoff. The checkpoint means a retry
  continues where the failed run stopped.
- Messages whose envelope cannot be parsed go to `bronze.startgg_rejects`
  with the raw key and value bytes, Kafka coordinates and a reason, instead of
  failing the job.
- The asset check `bronze_startgg_no_rejects` fails when the run wrote any
  reject rows.

### Secrets

- The start.gg token reaches the Dagster code container from `.env`. It is
  never logged or written into envelopes or query variables.
- MinIO, Postgres and Kafka credentials are local-only defaults documented in
  `.env.example`.

## Section 4: Testing

### Unit tests

Run with `uv run pytest`, with no Docker and no network. They cover:

- envelope building
- weekly window calculation
- tournament and event filtering
- the rate limiter
- error classification (retry, fail fast, or dead-letter)
- DLQ routing

They use recorded start.gg responses for one small real week, captured once
and saved under `tests/fixtures/startgg/`. Tournament data is public, so the
fixtures need no scrubbing. Tests never call the live API.

### Spark tests

- The Kafka-to-bronze logic is a pure function that takes a DataFrame with
  Kafka's source schema and returns bronze rows and reject rows. It is tested
  with a local SparkSession and constructed rows, with no Kafka.
- These tests are marked `spark`, are skipped by default locally, and run
  inside the Spark container with `make test-spark` and in CI on Ubuntu.

### Dagster tests

- `dagster definitions validate` checks that all definitions load.
- Assets are materialized in-process with fake API and Kafka resources. Tests
  cover partition window logic, DLQ routing and the asset check failing when
  DLQ messages were published.

### End-to-end smoke test

`make smoke`:

1. Starts the Compose stack.
2. Ingests one week from the recorded fixtures, using a fixture-backed client
   instead of the live API, so it needs no token.
3. Runs the bronze job.
4. Queries Trino and checks that bronze row counts match the fixture and
   `bronze.startgg_rejects` is empty.
5. Re-runs the same week and bronze job, then checks that bronze row counts
   doubled while the count of distinct set IDs stayed the same.

### CI

GitHub Actions moves from pip to `uv` and uses the pinned Python version. Two
jobs:

- **lint-and-unit** on every push and pull request: ruff, unit tests and
  Dagster tests.
- **spark-and-smoke** on pull requests and pushes to `master`: Spark tests and
  `make smoke`.

The existing 20 tests in `tests/` keep passing.

## Done criteria

1. `make up` starts all services and they report healthy.
2. A Dagster backfill of all weekly partitions from 2025-01-01 to the current
   week completes, with any DLQ messages visible through the asset check.
3. Trino returns rows from `bronze.startgg_tournaments` and
   `bronze.startgg_sets`.
4. Unit, Dagster, Spark and smoke tests pass locally and in CI.
5. The README gains a section describing the new stack and how to run it.
