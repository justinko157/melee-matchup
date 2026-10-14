"""Every job carries a dagster/max_runtime tag so a run whose worker died frees its pool slot."""

from pathlib import Path

import dagster as dg
import pytest
import yaml

from pipeline.dagster_defs import defs
from pipeline.dagster_defs.automation import lakehouse_automation

MAX_RUNTIME = "dagster/max_runtime"


@pytest.mark.parametrize(
    ("job", "seconds"),
    [
        ("ingest_startgg", 1800),
        ("land_bronze", 16200),
        ("train_matchup_model", 3600),
        ("lakehouse_maintenance", 9000),
    ],
)
def test_jobs_carry_a_max_runtime(job, seconds):
    job_def = defs.resolve_job_def(job)
    # The webserver's launchRun merges definition tags; schedules and sensors merge run tags.
    assert job_def.tags[MAX_RUNTIME] == str(seconds)
    assert job_def.run_tags[MAX_RUNTIME] == str(seconds)


def test_automation_runs_cover_silver_and_gold():
    # Silver (4-hour Spark timeout plus slack) and gold usually run together in one run.
    assert lakehouse_automation.run_tags[MAX_RUNTIME] == "19800"


def test_run_monitoring_enforces_max_runtime():
    config = yaml.safe_load(Path("infra/dagster/dagster.yaml").read_text(encoding="utf-8"))
    monitoring = config["run_monitoring"]
    assert monitoring["enabled"] is True
    # Runs with no tag (asset-page launches, backfills) still get the longest limit.
    assert monitoring["max_runtime_seconds"] == 19800


def test_refresh_lakehouse_job_rebuilds_silver_and_gold():
    # A fallback for `dagster_ops.py launch refresh_lakehouse` when automation doesn't fire.
    job = defs.resolve_job_def("refresh_lakehouse")
    assert job.asset_layer.executable_asset_keys == {
        dg.AssetKey("silver_startgg"),
        dg.AssetKey("gold_startgg"),
    }
    assert job.tags[MAX_RUNTIME] == "19800"
