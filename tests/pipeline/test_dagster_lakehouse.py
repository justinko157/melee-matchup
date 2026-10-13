import json

import dagster as dg
import pytest

from pipeline.dagster_defs import defs
from pipeline.dagster_defs.assets import bronze_startgg, startgg_raw
from pipeline.dagster_defs.automation import lakehouse_automation
from pipeline.dagster_defs.lakehouse_assets import drop_rate, gold_startgg, silver_startgg
from pipeline.dagster_defs.resources import (
    LakehouseResource,
    SparkJobResource,
    parse_silver_summary,
)
from tests.pipeline.lakehouse_data import silver_set, write_silver

ENTITIES = ("tournaments", "events", "players", "sets", "games")


def silver_summary(orphans=0, dropped=0, read=100, bad_winners=0):
    return {
        "tables": {e: {"inserted": 1, "updated": 0} for e in ENTITIES},
        "dropped": {
            "too_few_slots": dropped,
            "missing_player": 0,
            "not_completed": 0,
            "unresolved_winner": 0,
        },  # fmt: skip
        "bronze_rows_read": read,
        "bronze_tournament_rows_read": 0,
        "orphan_sets": orphans,
        "winner_not_player": bad_winners,
    }


class FakeSilverSpark(SparkJobResource):
    orphans: int = 0
    dropped: int = 0
    read: int = 100

    def run_silver_job(self):
        return silver_summary(self.orphans, self.dropped, self.read)


def check_results(**fake):
    result = dg.materialize([silver_startgg], resources={"spark_job": FakeSilverSpark(**fake)})
    return {e.check_name: e.passed for e in result.get_asset_check_evaluations()}


def test_clean_silver_summary_passes_every_check():
    assert check_results() == {
        "no_orphan_sets": True,
        "drop_rate_below_10pct": True,
        "winner_is_a_player": True,
    }


def test_drop_rate_above_10pct_fails():
    assert check_results(dropped=11, read=100)["drop_rate_below_10pct"] is False
    assert drop_rate(silver_summary(dropped=11, read=100)) == pytest.approx(0.11)


def test_nothing_read_passes_the_drop_rate_check():
    assert check_results(dropped=0, read=0)["drop_rate_below_10pct"] is True


def test_orphan_sets_fail_the_check():
    assert check_results(orphans=2)["no_orphan_sets"] is False


def test_parse_silver_summary_requires_every_key():
    with pytest.raises(ValueError, match="missing"):
        parse_silver_summary('{"tables": {}}\n')
    assert parse_silver_summary("log\n" + json.dumps(silver_summary()))["orphan_sets"] == 0


def test_silver_job_command_and_environment(monkeypatch):
    calls = []

    class Completed:
        returncode = 0
        stdout = json.dumps(silver_summary()) + "\n"
        stderr = ""

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return Completed()

    monkeypatch.setenv("STARTGG_API_TOKEN", "secret-token-value")
    monkeypatch.setattr("pipeline.dagster_defs.resources.subprocess.run", fake_run)
    SparkJobResource().run_silver_job()
    ((command, kwargs),) = calls
    assert command == [
        "spark-submit",
        "/app/pipeline/spark_jobs/silver_job.py",
        "--lock-dir",
        "/checkpoints/silver",
    ]
    assert "STARTGG_API_TOKEN" not in kwargs["env"]


def test_gold_asset_builds_tables_and_reports_checks(sql_settings, sql_catalog):
    write_silver(
        sql_catalog,
        sets=[silver_set(1, 100, 200, 100, 1_000_000)],
        events=[{"event_id": 10, "tournament_id": 1}],
        tournaments=[{"tournament_id": 1, "name": "T", "num_attendees": 64}],
        players=[{"player_id": 100, "gamer_tag": "A"}, {"player_id": 200, "gamer_tag": "B"}],
    )
    lakehouse = LakehouseResource(
        catalog_type="sql", uri=sql_settings.uri, warehouse=sql_settings.warehouse
    )
    result = dg.materialize([gold_startgg], resources={"lakehouse": lakehouse})
    assert all(e.passed for e in result.get_asset_check_evaluations())
    (materialization,) = result.asset_materializations_for_node("gold_startgg")
    assert materialization.metadata["set_features_rows"].value == 1


def automation(asset):
    return next(iter(asset.specs)).automation_condition


def test_silver_and_gold_are_eager_and_ingest_is_not():
    assert automation(silver_startgg) is not None
    assert automation(gold_startgg) is not None
    assert automation(startgg_raw) is None
    assert automation(bronze_startgg) is None


def test_automation_sensor_runs_by_default_and_targets_silver_and_gold():
    assert lakehouse_automation.default_status == dg.DefaultSensorStatus.RUNNING
    selected = lakehouse_automation.asset_selection.resolve(defs.resolve_asset_graph())
    assert selected == {silver_startgg.key, gold_startgg.key}


def test_spark_jobs_share_one_pool():
    assert bronze_startgg.op.pool == "spark_lakehouse"
    assert silver_startgg.op.pool == "spark_lakehouse"
    assert gold_startgg.op.pool == "gold"
