import dagster as dg

from pipeline.dagster_defs.maintenance import (
    lakehouse_maintenance,
    maintenance_totals,
    run_lakehouse_maintenance,
    weekly_maintenance,
)
from pipeline.dagster_defs.resources import SparkJobResource

SUMMARY = {
    "tables": {
        "bronze.startgg_sets": {"files_before": 40, "files_after": 2, "errors": {}},
        "gold.set_features": {"files_before": 3, "files_after": 3, "errors": {}},
    }
}


class FakeMaintenanceSpark(SparkJobResource):
    def run_maintenance_job(self):
        return SUMMARY


def test_maintenance_job_reports_file_counts():
    result = lakehouse_maintenance.execute_in_process(
        resources={"spark_job": FakeMaintenanceSpark()}
    )
    assert result.success
    assert result.output_for_node("run_lakehouse_maintenance") == SUMMARY
    assert maintenance_totals(SUMMARY) == (43, 5)


def test_weekly_maintenance_runs_sunday_3am_pacific():
    assert weekly_maintenance.cron_schedule == "0 3 * * 0"
    assert weekly_maintenance.execution_timezone == "America/Los_Angeles"
    assert weekly_maintenance.default_status == dg.DefaultScheduleStatus.RUNNING


def test_maintenance_uses_the_spark_pool():
    assert run_lakehouse_maintenance.pool == "spark_lakehouse"
