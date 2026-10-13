"""Weekly Iceberg maintenance: compact small files, expire old snapshots, remove orphans.

Run with spark-submit. The last line of stdout is a JSON summary. The exit code is 1
if compaction or snapshot expiry failed on any table; orphan-file cleanup failures
are reported but not fatal. On a fatal failure the JSON summary is also printed
to stderr, because the Dagster resource keeps only stderr when the job fails.
"""

import json
import sys
from datetime import UTC, datetime, timedelta

from pyspark.sql import SparkSession

from pipeline.gold.tables import GOLD_TABLES
from pipeline.spark_jobs.bronze import BRONZE_TABLES, REJECTS_TABLE
from pipeline.spark_jobs.silver import MERGE_ORDER, WATERMARKS_TABLE, table_name

MAINTAINED_TABLES = (
    *BRONZE_TABLES.values(),
    REJECTS_TABLE,
    *(table_name(entity) for entity in MERGE_ORDER),
    WATERMARKS_TABLE,
    *GOLD_TABLES,
)
TARGET_FILE_SIZE_BYTES = 128 * 1024 * 1024
SNAPSHOT_RETENTION = timedelta(days=7)
RETAIN_LAST = 5
ORPHAN_MIN_AGE = timedelta(days=3)
FATAL_PROCEDURES = frozenset({"rewrite_data_files", "expire_snapshots"})


def _literal(moment: datetime) -> str:
    # Spark reads TIMESTAMP literals in the session time zone, which the job pins to UTC.
    return moment.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


def maintenance_sql(catalog: str, table: str, now: datetime) -> dict[str, str]:
    expire_before = _literal(now - SNAPSHOT_RETENTION)
    orphans_before = _literal(now - ORPHAN_MIN_AGE)
    return {
        "rewrite_data_files": (
            f"CALL {catalog}.system.rewrite_data_files(table => '{table}', "
            f"options => map('target-file-size-bytes', '{TARGET_FILE_SIZE_BYTES}'))"
        ),
        "expire_snapshots": (
            f"CALL {catalog}.system.expire_snapshots(table => '{table}', "
            f"older_than => TIMESTAMP '{expire_before}', retain_last => {RETAIN_LAST})"
        ),
        # Prefix listing goes through the table's FileIO (S3FileIO with vended credentials);
        # the default lists through Hadoop FileSystem, which has no s3 scheme in our image.
        "remove_orphan_files": (
            f"CALL {catalog}.system.remove_orphan_files(table => '{table}', "
            f"older_than => TIMESTAMP '{orphans_before}', prefix_listing => true)"
        ),
    }


def file_count(spark: SparkSession, table: str) -> int:
    return int(spark.sql(f"SELECT count(*) AS n FROM {table}.files").first().n)


def maintain_table(spark: SparkSession, catalog: str, table: str, now: datetime) -> dict:
    before = file_count(spark, table)
    errors = {}
    for procedure, sql in maintenance_sql(catalog, table, now).items():
        try:
            spark.sql(sql).collect()
        except Exception as exc:  # report every table, then decide what is fatal
            errors[procedure] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return {"files_before": before, "files_after": file_count(spark, table), "errors": errors}


def run_maintenance(
    spark: SparkSession, tables: tuple[str, ...] = MAINTAINED_TABLES, now: datetime | None = None
) -> dict:
    now = now or datetime.now(UTC)
    catalog = spark.conf.get("spark.sql.defaultCatalog")
    return {
        "tables": {
            table: maintain_table(spark, catalog, table, now)
            for table in tables
            if spark.catalog.tableExists(table)
        }
    }


def fatal_errors(summary: dict) -> list[str]:
    return [
        f"{table}: {procedure}"
        for table, result in summary["tables"].items()
        for procedure in result["errors"]
        if procedure in FATAL_PROCEDURES
    ]


def report(summary: dict) -> int:
    """Print the summary (last stdout line) and return the exit code."""
    line = json.dumps(summary)
    print(line)
    fatal = fatal_errors(summary)
    if not fatal:
        return 0
    print(f"maintenance failed: {fatal}", file=sys.stderr)
    print(line, file=sys.stderr)
    return 1


def main() -> None:
    spark = SparkSession.builder.appName("lakehouse-maintenance").getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    summary = run_maintenance(spark)
    spark.stop()
    sys.exit(report(summary))


if __name__ == "__main__":
    main()
