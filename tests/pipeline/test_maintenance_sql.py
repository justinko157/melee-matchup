import json
from datetime import UTC, datetime

import pytest

pytest.importorskip("pyspark")

from pipeline.spark_jobs.maintenance_job import (  # noqa: E402
    MAINTAINED_TABLES,
    fatal_errors,
    maintenance_sql,
    report,
)


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


def test_orphan_cleanup_lists_through_the_table_file_io():
    # The Hadoop FileSystem has no s3 scheme in the Spark image; prefix listing uses S3FileIO.
    now = datetime(2026, 10, 11, 10, 0, tzinfo=UTC)
    assert "prefix_listing => true" in maintenance_sql("c", "t", now)["remove_orphan_files"]


def test_watermarks_table_is_maintained():
    assert "silver._watermarks" in MAINTAINED_TABLES


def test_fatal_failure_prints_the_full_summary_to_stderr(capsys):
    summary = {
        "tables": {
            "b": {"files_before": 3, "files_after": 3, "errors": {"expire_snapshots": "boom"}}
        }
    }
    assert report(summary) == 1
    out, err = capsys.readouterr()
    assert json.loads(out.strip().splitlines()[-1]) == summary
    assert "maintenance failed: ['b: expire_snapshots']" in err
    assert json.loads(err.strip().splitlines()[-1]) == summary


def test_success_exits_zero_and_keeps_stderr_quiet(capsys):
    summary = {"tables": {"a": {"files_before": 3, "files_after": 1, "errors": {}}}}
    assert report(summary) == 0
    out, err = capsys.readouterr()
    assert json.loads(out.strip().splitlines()[-1]) == summary
    assert err == ""
