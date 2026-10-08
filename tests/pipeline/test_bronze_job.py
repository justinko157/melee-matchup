from datetime import UTC, datetime

import pytest

pytest.importorskip("pyspark")

from pipeline.spark_jobs.bronze_job import added_rows_sql, parse_since  # noqa: E402


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
