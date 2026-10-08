from datetime import UTC, datetime

import pytest

pytest.importorskip("pyspark")

from pipeline.spark_jobs.bronze_job import (  # noqa: E402
    added_rows_sql,
    await_all,
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


class FakeQuery:
    def __init__(self, name, error=None):
        self.name = name
        self.isActive = True
        self.stopped = False
        self._error = error

    def awaitTermination(self):
        if self._error:
            self.isActive = False
            raise self._error
        self.isActive = False

    def stop(self):
        self.stopped = True
        self.isActive = False


class HangingQuery(FakeQuery):
    """Never finishes on its own; only stop() ends it."""

    def awaitTermination(self):
        raise AssertionError("should not be awaited after an earlier failure")


def test_await_all_waits_for_every_query():
    queries = [FakeQuery("a"), FakeQuery("b")]
    await_all(queries)
    assert not any(q.isActive for q in queries)
    assert not any(q.stopped for q in queries)


def test_await_all_stops_the_others_when_one_fails():
    failing = FakeQuery("tournaments", error=RuntimeError("stream failed"))
    sets, rejects = HangingQuery("sets"), HangingQuery("rejects")
    with pytest.raises(RuntimeError, match="stream failed"):
        await_all([failing, sets, rejects])
    assert sets.stopped and rejects.stopped
    assert not failing.stopped


def test_await_all_keeps_the_original_error_if_stop_fails():
    class StopFails(HangingQuery):
        def stop(self):
            raise RuntimeError("stop failed")

    other = HangingQuery("rejects")
    with pytest.raises(RuntimeError, match="stream failed"):
        await_all([FakeQuery("a", error=RuntimeError("stream failed")), StopFails("b"), other])
    assert other.stopped
