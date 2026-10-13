from datetime import UTC, datetime

import pytest

pytest.importorskip("pyspark")

from pipeline.spark_jobs.maintenance_job import fatal_errors, maintenance_sql  # noqa: E402


def test_maintenance_sql_uses_the_catalog_and_utc_literals():
    now = datetime(2026, 10, 11, 10, 0, tzinfo=UTC)
    sql = maintenance_sql("lakekeeper", "bronze.startgg_sets", now)
    assert sql["rewrite_data_files"].startswith("CALL lakekeeper.system.rewrite_data_files(")
    assert "'target-file-size-bytes', '134217728'" in sql["rewrite_data_files"]
    assert "older_than => TIMESTAMP '2026-10-04 10:00:00'" in sql["expire_snapshots"]
    assert "retain_last => 5" in sql["expire_snapshots"]
    assert "older_than => TIMESTAMP '2026-10-08 10:00:00'" in sql["remove_orphan_files"]


def test_only_compaction_and_expiry_failures_are_fatal():
    summary = {
        "tables": {
            "a": {"files_before": 3, "files_after": 1, "errors": {"remove_orphan_files": "x"}},
            "b": {"files_before": 3, "files_after": 3, "errors": {"expire_snapshots": "y"}},
        }
    }
    assert fatal_errors(summary) == ["b: expire_snapshots"]
