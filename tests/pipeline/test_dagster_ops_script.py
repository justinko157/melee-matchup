import importlib.util
import json
from datetime import date
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("dagster_ops", Path("scripts/dagster_ops.py"))
dagster_ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dagster_ops)


def test_backfill_covers_every_monday_inclusive():
    keys = dagster_ops.weekly_partition_keys(date(2018, 1, 1), date(2024, 12, 30))
    assert len(keys) == 366
    assert (keys[0], keys[1], keys[-1]) == ("2018-01-01", "2018-01-08", "2024-12-30")


def test_backfill_bounds_must_be_mondays():
    with pytest.raises(ValueError, match="Monday"):
        dagster_ops.weekly_partition_keys(date(2018, 1, 2), date(2024, 12, 30))


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def serve(monkeypatch, payload):
    monkeypatch.setattr(
        dagster_ops.urllib.request, "urlopen", lambda request, timeout: FakeResponse(payload)
    )


def test_launch_success_prints_the_run_and_exits_zero(monkeypatch, capsys):
    serve(
        monkeypatch,
        {"data": {"launchRun": {"__typename": "LaunchRunSuccess", "run": {"runId": "abc"}}}},
    )
    dagster_ops.main(["launch", "train_matchup_model"])
    assert '"runId": "abc"' in capsys.readouterr().out


def test_launch_error_union_exits_non_zero(monkeypatch, capsys):
    serve(monkeypatch, {"data": {"launchRun": {"__typename": "PythonError", "message": "boom"}}})
    with pytest.raises(SystemExit) as exit_info:
        dagster_ops.main(["launch", "train_matchup_model"])
    assert exit_info.value.code == 1
    assert "PythonError" in capsys.readouterr().err


def test_run_status_for_an_unknown_run_exits_non_zero(monkeypatch, capsys):
    serve(monkeypatch, {"data": {"runOrError": {"__typename": "RunNotFoundError"}}})
    with pytest.raises(SystemExit) as exit_info:
        dagster_ops.main(["run-status", "nope"])
    assert exit_info.value.code == 1
    assert "RunNotFoundError" in capsys.readouterr().err


def test_unreachable_webserver_exits_with_one_line(monkeypatch, capsys):
    def refuse(request, timeout):
        raise dagster_ops.urllib.error.URLError("connection refused")

    monkeypatch.setattr(dagster_ops.urllib.request, "urlopen", refuse)
    monkeypatch.setenv("DAGSTER_GRAPHQL_URL", "http://localhost:9/graphql")
    with pytest.raises(SystemExit) as exit_info:
        dagster_ops.main(["run-status", "abc"])
    assert exit_info.value.code == 1
    err = capsys.readouterr().err
    assert "http://localhost:9/graphql" in err
    assert "Traceback" not in err
