"""Weekly Iceberg maintenance job (compaction, snapshot expiry, orphan cleanup)."""

import dagster as dg

from pipeline.dagster_defs.pools import SPARK_LAKEHOUSE
from pipeline.dagster_defs.resources import SparkJobResource
from pipeline.dagster_defs.run_limits import MAINTENANCE, max_runtime


def maintenance_totals(summary: dict) -> tuple[int, int]:
    """(data files before, data files after) across every maintained table."""
    tables = summary["tables"].values()
    return sum(t["files_before"] for t in tables), sum(t["files_after"] for t in tables)


@dg.op(pool=SPARK_LAKEHOUSE)
def run_lakehouse_maintenance(
    context: dg.OpExecutionContext, spark_job: SparkJobResource
) -> dg.Output[dict]:
    summary = spark_job.run_maintenance_job()
    for table, result in summary["tables"].items():
        if result["errors"]:
            context.log.warning(f"{table}: {result['errors']}")
    before, after = maintenance_totals(summary)
    return dg.Output(
        summary,
        metadata={
            "files_before": before,
            "files_after": after,
            "tables": dg.MetadataValue.json(summary["tables"]),
        },
    )


@dg.job(tags=max_runtime(MAINTENANCE))
def lakehouse_maintenance():
    run_lakehouse_maintenance()


weekly_maintenance = dg.ScheduleDefinition(
    name="weekly_maintenance",
    job=lakehouse_maintenance,
    cron_schedule="0 3 * * 0",
    execution_timezone="America/Los_Angeles",
    default_status=dg.DefaultScheduleStatus.RUNNING,
)
