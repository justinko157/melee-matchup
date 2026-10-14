"""Jobs, the bronze sensor, the daily refresh schedule and lakehouse automation."""

import dagster as dg

from pipeline.dagster_defs.assets import bronze_startgg, startgg_raw
from pipeline.dagster_defs.lakehouse_assets import gold_startgg, silver_startgg
from pipeline.dagster_defs.partitions import WEEKLY_PARTITIONS
from pipeline.dagster_defs.run_limits import BRONZE, INGEST, LAKEHOUSE, max_runtime

ingest_job = dg.define_asset_job(
    "ingest_startgg", selection=[startgg_raw], tags=max_runtime(INGEST)
)
bronze_job = dg.define_asset_job(
    "land_bronze", selection=[bronze_startgg], tags=max_runtime(BRONZE)
)
# Silver then gold by hand, for when declarative automation doesn't fire.
refresh_lakehouse_job = dg.define_asset_job(
    "refresh_lakehouse", selection=[silver_startgg, gold_startgg], tags=max_runtime(LAKEHOUSE)
)

PENDING = [dg.DagsterRunStatus.QUEUED, dg.DagsterRunStatus.NOT_STARTED]


def _ingested_startgg(instance: dg.DagsterInstance, run_id: str) -> bool:
    # Runs from the asset graph or from materialize() use an implicit job name
    # with no recorded asset selection, so look for a startgg_raw
    # materialization in the run's event log instead.
    records = instance.get_records_for_run(
        run_id, of_type=dg.DagsterEventType.ASSET_MATERIALIZATION
    ).records
    return any(record.asset_key == startgg_raw.key for record in records)


@dg.run_status_sensor(
    run_status=dg.DagsterRunStatus.SUCCESS,
    request_job=bronze_job,
    default_status=dg.DefaultSensorStatus.RUNNING,
)
def land_bronze_after_ingest(context: dg.RunStatusSensorContext):
    """Land bronze after each successful ingestion run.

    Bronze runs share one streaming checkpoint, so a run that is already
    queued will pick up these messages too; don't queue another.
    """
    if not _ingested_startgg(context.instance, context.dagster_run.run_id):
        return dg.SkipReason("Not an ingestion run")
    pending = context.instance.get_runs(
        filters=dg.RunsFilter(job_name=bronze_job.name, statuses=PENDING), limit=1
    )
    if pending:
        return dg.SkipReason("A bronze run is already queued")
    return dg.RunRequest(run_key=context.dagster_run.run_id)


@dg.schedule(
    job=ingest_job,
    cron_schedule="0 6 * * *",
    execution_timezone="America/Los_Angeles",
    default_status=dg.DefaultScheduleStatus.RUNNING,
)
def refresh_recent_weeks(context: dg.ScheduleEvaluationContext) -> list[dg.RunRequest]:
    """Re-ingest the current and previous week to pick up newly finished tournaments."""
    keys = WEEKLY_PARTITIONS.get_partition_keys(current_time=context.scheduled_execution_time)[-2:]
    day = context.scheduled_execution_time.strftime("%Y-%m-%d")
    return [dg.RunRequest(run_key=f"{key}@{day}", partition_key=key) for key in keys]


# Declarative automation sensors start STOPPED unless told otherwise.
lakehouse_automation = dg.AutomationConditionSensorDefinition(
    "lakehouse_automation",
    target=dg.AssetSelection.assets(silver_startgg, gold_startgg),
    run_tags=max_runtime(LAKEHOUSE),
    default_status=dg.DefaultSensorStatus.RUNNING,
)
