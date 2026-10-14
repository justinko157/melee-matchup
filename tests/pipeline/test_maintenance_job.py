import pytest

pytestmark = pytest.mark.spark
pytest.importorskip("pyspark")

from pipeline.spark_jobs.maintenance_job import run_maintenance  # noqa: E402


def test_maintenance_compacts_small_files(spark):
    spark.sql("CREATE NAMESPACE IF NOT EXISTS maint")
    spark.sql("DROP TABLE IF EXISTS maint.events")
    spark.sql("CREATE TABLE maint.events (id BIGINT) USING iceberg")
    for i in range(6):
        spark.createDataFrame([(i,)], "id BIGINT").writeTo("maint.events").append()
    summary = run_maintenance(spark, tables=("maint.events",))
    result = summary["tables"]["maint.events"]
    assert result["errors"] == {}
    assert result["files_before"] == 6
    assert result["files_after"] < result["files_before"]
    assert spark.table("maint.events").count() == 6


def test_missing_tables_are_skipped(spark):
    assert run_maintenance(spark, tables=("maint.does_not_exist",)) == {"tables": {}}
