import json
from pathlib import Path

import pytest

from pipeline.ingest.collect import collect_week, is_eligible_tournament, melee_events
from pipeline.ingest.fixtures import FixtureClient, RecordingClient, fixture_key
from pipeline.ingest.queries import EVENT_SETS
from pipeline.topics import DLQ_TOPIC, SETS_TOPIC, TOURNAMENTS_TOPIC
from tests.pipeline.fakes import FakePaginator, FakePublisher, melee_event, tournament

FIXTURE_WEEK = "2025-01-06"
FIXTURE = Path("tests/fixtures/startgg") / f"week_{FIXTURE_WEEK}.json"


def test_fixture_key_ignores_pagination():
    assert fixture_key(EVENT_SETS, {"eventId": 1, "page": 4, "perPage": 15}) == (
        'EventSets:{"eventId": 1}'
    )


def test_record_then_replay_round_trips(tmp_path):
    inner = FakePaginator([tournament(1, events=[melee_event(10)])], {10: [{"id": 100}]})
    recorder = RecordingClient(inner)
    collect_week(recorder, FakePublisher(), FIXTURE_WEEK, run_id="r")
    path = tmp_path / "week.json"
    recorder.save(path)

    replayed = FakePublisher()
    collect_week(FixtureClient(path), replayed, FIXTURE_WEEK, run_id="r")
    assert [k for _, k, _ in replayed.on(SETS_TOPIC)] == ["100"]


def test_fixture_client_reports_missing_key(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(KeyError, match="EventSets"):
        FixtureClient(path).paginate(EVENT_SETS, {"eventId": 1}, ["event", "sets"])


def test_recorded_week_replays_consistently():
    """The real recorded week: message counts must match the raw fixture."""
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    (discovery,) = [v for k, v in data.items() if k.startswith("TournamentsByVideogame:")]
    eligible = [t for t in discovery if is_eligible_tournament(t, 50)]
    expected_sets = sum(
        len(data[fixture_key(EVENT_SETS, {"eventId": e["id"]})])
        for t in eligible
        for e in melee_events(t)
    )

    publisher = FakePublisher()
    stats = collect_week(FixtureClient(FIXTURE), publisher, FIXTURE_WEEK, run_id="r")

    assert stats.tournaments == len(eligible) > 0
    assert stats.sets == expected_sets > 0
    assert len(publisher.on(TOURNAMENTS_TOPIC)) == stats.tournaments
    assert publisher.on(DLQ_TOPIC) == []
