from datetime import UTC, datetime

import pytest

pytest.importorskip("pyspark")

from pipeline.spark_jobs.bronze_job import (  # noqa: E402
    added_rows_sql,
    lock_checkpoints,
    parse_since,
)


def test_parse_since_converts_to_utc():
    assert parse_since("2025-01-06T08:00:00-08:00") == datetime(2025, 1, 6, 16, tzinfo=UTC)


def test_parse_since_requires_a_time_zone():
    with pytest.raises(ValueError):
        parse_since("2025-01-06T08:00:00")


def test_added_rows_sql_uses_epoch_millis_and_appends_only():
    sql = added_rows_sql("bronze.startgg_sets", datetime(2025, 1, 6, 16, 0, 0, 123000, tzinfo=UTC))
    assert "FROM bronze.startgg_sets.snapshots" in sql
    assert "committed_at >= timestamp_millis(1736179200123)" in sql
    assert "operation = 'append'" in sql


def test_checkpoint_lock_refuses_a_second_job(tmp_path):
    pytest.importorskip("fcntl")  # the job only runs in Linux containers

    root = tmp_path / "bronze"
    held = lock_checkpoints(str(root))
    assert (root / ".lock").exists()
    with pytest.raises(SystemExit, match="another bronze job is running") as info:
        lock_checkpoints(str(root))
    assert info.value.code != 0
    held.close()
    lock_checkpoints(str(root)).close()
