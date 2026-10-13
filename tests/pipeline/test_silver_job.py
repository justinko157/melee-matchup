import json
from datetime import UTC, datetime

import pytest

pytestmark = pytest.mark.spark
pytest.importorskip("pyspark")

from pipeline.spark_jobs.bronze import BRONZE_TABLES, bronze_table_ddl  # noqa: E402
from pipeline.spark_jobs.silver import MERGE_ORDER, WATERMARKS_TABLE, table_name  # noqa: E402
from pipeline.spark_jobs.silver_job import run_silver  # noqa: E402

BRONZE_SETS, BRONZE_TOURNAMENTS = BRONZE_TABLES["set"], BRONZE_TABLES["tournament"]
COLUMNS = (
    "kafka_topic", "kafka_partition", "kafka_offset", "kafka_timestamp", "kafka_key", "entity",
    "entity_id", "ingested_at", "dagster_run_id", "partition_week", "query_name",
    "query_variables", "payload",
)  # fmt: skip
OFFSETS = iter(range(1, 10**9))


@pytest.fixture
def lakehouse(spark):
    """Fresh bronze and silver tables for each test."""
    for entity in MERGE_ORDER:
        spark.sql(f"DROP TABLE IF EXISTS {table_name(entity)}")
    spark.sql(f"DROP TABLE IF EXISTS {WATERMARKS_TABLE}")
    for table in (BRONZE_SETS, BRONZE_TOURNAMENTS):
        spark.sql(f"DROP TABLE IF EXISTS {table}")
    spark.sql("CREATE NAMESPACE IF NOT EXISTS bronze")
    return spark


def create_bronze(spark):
    for table in (BRONZE_SETS, BRONZE_TOURNAMENTS):
        spark.sql(bronze_table_ddl(table))


def append(spark, table, entity, payload, query_variables, ingested_at):
    """One bronze append = one Iceberg snapshot."""
    row = (
        "topic", 0, next(OFFSETS), ingested_at, str(payload["id"]), entity, str(payload["id"]),
        ingested_at, "run", "2025-01-06", "Q", json.dumps(query_variables), json.dumps(payload),
    )  # fmt: skip
    schema = spark.table(table).schema
    spark.createDataFrame([row], schema).select(*COLUMNS).writeTo(table).append()


def tournament(tid=7, event_ids=(70,)):
    events = [
        {"id": e, "name": "Singles", "slug": "s", "numEntrants": 64, "videogame": {"id": 1}}
        for e in event_ids
    ]
    return {"id": tid, "name": "T", "slug": "t", "numAttendees": 100, "startAt": 1736150400,
            "endAt": 1736200000, "city": "X", "addrState": "CA", "countryCode": "US",
            "isOnline": False, "events": events}  # fmt: skip


def a_set(set_id, score=2.0):
    def slot(eid, pid, tag, value):
        entrant = {"id": eid, "name": tag, "initialSeedNum": 1,
                   "participants": [{"player": {"id": pid, "gamerTag": tag}}]}  # fmt: skip
        standing = {"placement": 1, "stats": {"score": {"value": value}}}
        return {"seed": {"seedNum": 1}, "standing": standing, "entrant": entrant}

    return {"id": set_id, "displayScore": "", "winnerId": 11, "round": 1, "fullRoundText": "R1",
            "completedAt": 1736200000, "totalGames": 3, "state": 3, "games": [],
            "slots": [slot(11, 1, "A", score), slot(12, 2, "B", 0.0)],
            "phaseGroup": {"phase": {"name": "Pools"}}}  # fmt: skip


T0 = datetime(2025, 1, 6, tzinfo=UTC)
T1 = datetime(2025, 1, 7, tzinfo=UTC)


def count(spark, entity):
    return spark.table(table_name(entity)).count()


def test_missing_bronze_tables_is_a_no_op(lakehouse):
    summary = run_silver(lakehouse)
    assert summary["bronze_rows_read"] == 0
    assert summary["tables"]["sets"] == {"inserted": 0, "updated": 0}
    assert summary["orphan_sets"] == 0
    assert all(count(lakehouse, e) == 0 for e in MERGE_ORDER)


def test_first_run_reads_everything(lakehouse):
    create_bronze(lakehouse)
    append(lakehouse, BRONZE_TOURNAMENTS, "tournament", tournament(), {}, T0)
    append(lakehouse, BRONZE_SETS, "set", a_set(100), {"eventId": 70}, T0)
    append(lakehouse, BRONZE_SETS, "set", a_set(101), {"eventId": 70}, T0)
    summary = run_silver(lakehouse)
    assert summary["bronze_rows_read"] == 2
    assert summary["tables"]["sets"] == {"inserted": 2, "updated": 0}
    assert summary["tables"]["players"] == {"inserted": 2, "updated": 0}
    assert (count(lakehouse, "tournaments"), count(lakehouse, "events")) == (1, 1)
    watermarks = {
        r.bronze_table: r.snapshot_id for r in lakehouse.table(WATERMARKS_TABLE).collect()
    }
    assert set(watermarks) == {BRONZE_SETS, BRONZE_TOURNAMENTS}


def test_second_run_reads_only_new_snapshots(lakehouse):
    create_bronze(lakehouse)
    append(lakehouse, BRONZE_TOURNAMENTS, "tournament", tournament(), {}, T0)
    append(lakehouse, BRONZE_SETS, "set", a_set(100), {"eventId": 70}, T0)
    run_silver(lakehouse)
    append(lakehouse, BRONZE_SETS, "set", a_set(101), {"eventId": 70}, T1)
    summary = run_silver(lakehouse)
    assert summary["bronze_rows_read"] == 1
    assert summary["bronze_tournament_rows_read"] == 0
    assert summary["tables"]["sets"] == {"inserted": 1, "updated": 0}
    assert count(lakehouse, "sets") == 2


def test_unchanged_bronze_is_a_no_op(lakehouse):
    create_bronze(lakehouse)
    append(lakehouse, BRONZE_TOURNAMENTS, "tournament", tournament(), {}, T0)
    append(lakehouse, BRONZE_SETS, "set", a_set(100), {"eventId": 70}, T0)
    run_silver(lakehouse)
    summary = run_silver(lakehouse)
    assert summary["bronze_rows_read"] == 0
    assert all(v == {"inserted": 0, "updated": 0} for v in summary["tables"].values())


def test_older_version_never_overwrites_newer(lakehouse):
    create_bronze(lakehouse)
    append(lakehouse, BRONZE_TOURNAMENTS, "tournament", tournament(), {}, T0)
    append(lakehouse, BRONZE_SETS, "set", a_set(100, score=3.0), {"eventId": 70}, T1)
    run_silver(lakehouse)
    append(
        lakehouse, BRONZE_SETS, "set", a_set(100, score=1.0), {"eventId": 70}, T0
    )  # replayed old copy
    summary = run_silver(lakehouse)
    assert summary["tables"]["sets"] == {"inserted": 0, "updated": 0}
    assert lakehouse.table(table_name("sets")).first().player1_score == 3.0


def test_expired_watermark_falls_back_to_full_scan(lakehouse):
    create_bronze(lakehouse)
    append(lakehouse, BRONZE_TOURNAMENTS, "tournament", tournament(), {}, T0)
    append(lakehouse, BRONZE_SETS, "set", a_set(100), {"eventId": 70}, T0)
    append(lakehouse, BRONZE_SETS, "set", a_set(101), {"eventId": 70}, T0)
    run_silver(lakehouse)
    lakehouse.sql(
        f"UPDATE {WATERMARKS_TABLE} SET snapshot_id = -1 WHERE bronze_table = '{BRONZE_SETS}'"
    )
    append(lakehouse, BRONZE_SETS, "set", a_set(102), {"eventId": 70}, T1)
    summary = run_silver(lakehouse)
    assert summary["bronze_rows_read"] == 3  # full scan
    assert summary["tables"]["sets"] == {
        "inserted": 1,
        "updated": 2,
    }  # re-reads are same-value updates
    assert count(lakehouse, "sets") == 3


def test_compaction_snapshot_does_not_force_a_full_scan(lakehouse):
    create_bronze(lakehouse)
    append(lakehouse, BRONZE_TOURNAMENTS, "tournament", tournament(), {}, T0)
    append(lakehouse, BRONZE_SETS, "set", a_set(100), {"eventId": 70}, T0)
    run_silver(lakehouse)
    append(lakehouse, BRONZE_SETS, "set", a_set(101), {"eventId": 70}, T1)
    lakehouse.sql(
        f"CALL local.system.rewrite_data_files(table => '{BRONZE_SETS}', "
        f"options => map('min-input-files', '1'))"
    ).collect()
    append(lakehouse, BRONZE_SETS, "set", a_set(102), {"eventId": 70}, T1)
    summary = run_silver(lakehouse)
    assert summary["bronze_rows_read"] == 2  # the two appends after the watermark
    assert count(lakehouse, "sets") == 3


def test_orphan_sets_are_counted(lakehouse):
    create_bronze(lakehouse)
    append(lakehouse, BRONZE_TOURNAMENTS, "tournament", tournament(event_ids=(70,)), {}, T0)
    append(lakehouse, BRONZE_SETS, "set", a_set(100), {"eventId": 999}, T0)
    summary = run_silver(lakehouse)
    assert summary["orphan_sets"] == 1
    assert summary["winner_not_player"] == 0
