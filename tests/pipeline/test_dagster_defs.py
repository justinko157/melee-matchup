import subprocess
from datetime import UTC, datetime
from typing import ClassVar
from zoneinfo import ZoneInfo

import dagster as dg
import pytest
from dagster._core.remote_origin import (
    RegisteredCodeLocationOrigin,
    RemoteJobOrigin,
    RemoteRepositoryOrigin,
)

from pipeline.dagster_defs import defs
from pipeline.dagster_defs.assets import bronze_startgg, startgg_raw
from pipeline.dagster_defs.automation import land_bronze_after_ingest, refresh_recent_weeks
from pipeline.dagster_defs.partitions import WEEKLY_PARTITIONS
from pipeline.dagster_defs.resources import (
    KafkaResource,
    SparkJobResource,
    StartGGResource,
    build_bronze_command,
    parse_job_summary,
)
from pipeline.ingest.startgg import RetriesExhaustedError
from pipeline.topics import SETS_TOPIC
from tests.pipeline.fakes import FakePaginator, FakePublisher, melee_event, tournament


class FakeStartGG(StartGGResource):
    scenario: str = "ok"

    def get_client(self):
        sets = {10: [{"id": 100}, {"id": 101}]}
        if self.scenario == "dlq":
            sets = {10: RetriesExhaustedError("down")}
        return FakePaginator([tournament(1, events=[melee_event(10)])], sets)


class FakeKafka(KafkaResource):
    publishers: ClassVar[list[FakePublisher]] = []

    def get_publisher(self):
        publisher = FakePublisher()
        FakeKafka.publishers.append(publisher)
        return publisher


class FakeSpark(SparkJobResource):
    rejects: int = 0
    calls: ClassVar[list[datetime | None]] = []

    def run_bronze_job(self, since=None):
        FakeSpark.calls.append(since)
        return {"tournaments": 1, "sets": 2, "rejects": self.rejects}


def materialize_raw(scenario="ok", instance=None):
    return dg.materialize(
        [startgg_raw],
        partition_key="2025-01-06",
        resources={"startgg": FakeStartGG(token="unused", scenario=scenario), "kafka": FakeKafka()},
        instance=instance,
    )


def checks(result):
    return {e.check_name: e.passed for e in result.get_asset_check_evaluations()}


def test_definitions_load():
    dg.Definitions.validate_loadable(defs)


def test_partitions_are_mondays_from_2018():
    keys = WEEKLY_PARTITIONS.get_partition_keys(
        current_time=datetime(2018, 1, 20, tzinfo=ZoneInfo("UTC"))
    )
    assert keys[0] == "2018-01-01"
    assert all(datetime.strptime(k, "%Y-%m-%d").weekday() == 0 for k in keys)


def test_startgg_raw_publishes_and_flushes():
    FakeKafka.publishers.clear()
    result = materialize_raw()
    assert result.success
    (publisher,) = FakeKafka.publishers
    assert publisher.flushed
    assert len(publisher.on(SETS_TOPIC)) == 2
    metadata = result.asset_materializations_for_node("startgg_raw")[0].metadata
    assert metadata["sets"].value == 2
    assert metadata["dlq_messages"].value == 0
    assert checks(result) == {"no_dlq_messages": True}


def test_dlq_fails_the_check_but_not_the_run():
    result = materialize_raw(scenario="dlq")
    assert result.success
    assert checks(result) == {"no_dlq_messages": False}


@pytest.mark.parametrize(("rejects", "passed"), [(0, True), (3, False)])
def test_bronze_check_reflects_rejects(rejects, passed):
    result = dg.materialize([bronze_startgg], resources={"spark_job": FakeSpark(rejects=rejects)})
    assert result.success
    assert checks(result) == {"no_rejects": passed}


def test_parse_job_summary_reads_last_line():
    stdout = 'some log line\n{"tournaments": 2, "sets": 30, "rejects": 0}\n'
    assert parse_job_summary(stdout) == {"tournaments": 2, "sets": 30, "rejects": 0}


@pytest.mark.parametrize("stdout", ["", "\n", '{"tournaments": 1}'])
def test_parse_job_summary_rejects_bad_output(stdout):
    with pytest.raises(ValueError):
        parse_job_summary(stdout)


def test_build_bronze_command_passes_since():
    since = datetime(2025, 1, 6, 16, 0, tzinfo=UTC)
    assert build_bronze_command("/job.py", "kafka:9092", "/ck", since)[-2:] == [
        "--since",
        "2025-01-06T16:00:00+00:00",
    ]


def test_bronze_counts_from_the_run_start():
    # Every retry attempt of a run must count rows from the same moment.
    FakeSpark.calls.clear()
    with dg.instance_for_test() as instance:
        result = dg.materialize(
            [bronze_startgg], resources={"spark_job": FakeSpark()}, instance=instance
        )
        start = instance.get_run_record_by_id(result.run_id).start_time
    assert FakeSpark.calls == [datetime.fromtimestamp(start, UTC)]


def test_build_bronze_command():
    assert build_bronze_command("/job.py", "kafka:9092", "/ck") == [
        "spark-submit",
        "/job.py",
        "--bootstrap-servers",
        "kafka:9092",
        "--checkpoint-root",
        "/ck",
    ]


def test_schedule_requests_current_and_previous_week():
    context = dg.build_schedule_context(
        scheduled_execution_time=datetime(2025, 1, 8, 6, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
    )
    requests = refresh_recent_weeks(context)
    assert [r.partition_key for r in requests] == ["2024-12-30", "2025-01-06"]


def sensor_context(result, instance):
    return dg.build_run_status_sensor_context(
        sensor_name="land_bronze_after_ingest",
        dagster_event=result.get_run_success_event(),
        dagster_instance=instance,
        dagster_run=result.dagster_run,
    )


def test_sensor_requests_bronze_after_ingest():
    with dg.instance_for_test() as instance:
        result = materialize_raw(instance=instance)
        request = land_bronze_after_ingest(sensor_context(result, instance))
        assert isinstance(request, dg.RunRequest)
        assert request.run_key == result.run_id


def test_sensor_ignores_other_runs():
    with dg.instance_for_test() as instance:
        result = dg.materialize(
            [bronze_startgg], resources={"spark_job": FakeSpark()}, instance=instance
        )
        assert isinstance(land_bronze_after_ingest(sensor_context(result, instance)), dg.SkipReason)


def test_sensor_skips_when_bronze_run_already_queued():
    with dg.instance_for_test() as instance:
        result = materialize_raw(instance=instance)
        # Dagster requires queued runs to record where their job came from.
        origin = RemoteJobOrigin(
            RemoteRepositoryOrigin(RegisteredCodeLocationOrigin("melee"), "__repository__"),
            "land_bronze",
        )
        instance.add_run(
            dg.DagsterRun(
                job_name="land_bronze",
                run_id="queued-1",
                status=dg.DagsterRunStatus.QUEUED,
                remote_job_origin=origin,
            )
        )
        assert isinstance(land_bronze_after_ingest(sensor_context(result, instance)), dg.SkipReason)


def test_dlq_check_is_attributed_to_the_week():
    (evaluation,) = materialize_raw(scenario="dlq").get_asset_check_evaluations()
    assert evaluation.partition == "2025-01-06"
    assert evaluation.metadata["partition_week"].value == "2025-01-06"


class FakeCompleted:
    returncode = 0
    stdout = '{"tournaments": 1, "sets": 2, "rejects": 0}\n'
    stderr = ""


def test_bronze_job_runs_with_a_timeout_and_without_the_token(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(kwargs)
        return FakeCompleted()

    monkeypatch.setenv("STARTGG_API_TOKEN", "secret-token-value")
    monkeypatch.setenv("KEEP_ME", "1")
    monkeypatch.setattr("pipeline.dagster_defs.resources.subprocess.run", fake_run)
    summary = SparkJobResource(timeout_seconds=120).run_bronze_job()
    assert summary == {"tournaments": 1, "sets": 2, "rejects": 0}
    (kwargs,) = calls
    assert kwargs["timeout"] == 120
    assert "STARTGG_API_TOKEN" not in kwargs["env"]
    assert kwargs["env"]["KEEP_ME"] == "1"


def test_bronze_job_timeout_default_is_an_hour():
    assert SparkJobResource().timeout_seconds == 3600


def test_bronze_job_timeout_becomes_a_failure(monkeypatch):
    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr("pipeline.dagster_defs.resources.subprocess.run", fake_run)
    with pytest.raises(dg.Failure, match="timed out after 3600s"):
        SparkJobResource().run_bronze_job()
