from datetime import UTC, datetime

import pytest

from pipeline.ingest.collect import collect_week
from pipeline.ingest.startgg import InvalidTokenError, PaginationCapError, RetriesExhaustedError
from pipeline.topics import DLQ_TOPIC, SETS_TOPIC, TOURNAMENTS_TOPIC
from tests.pipeline.fakes import FakePaginator, FakePublisher, melee_event, tournament

WEEK = "2025-01-06"
FIXED_NOW = datetime(2025, 1, 13, 6, 0, tzinfo=UTC)


def run(paginator, min_attendees=50):
    publisher = FakePublisher()
    stats = collect_week(
        paginator,
        publisher,
        WEEK,
        run_id="run-1",
        min_attendees=min_attendees,
        now=lambda: FIXED_NOW,
    )
    return stats, publisher


def test_discovery_uses_week_bounds():
    paginator = FakePaginator([])
    run(paginator)
    name, variables = paginator.calls[0]
    assert name == "TournamentsByVideogame"
    assert variables["afterDate"] == 1736121600
    assert variables["beforeDate"] == 1736726399
    assert variables["videogameId"] == 1


def test_empty_week_publishes_nothing():
    stats, publisher = run(FakePaginator([]))
    assert publisher.messages == []
    assert (stats.tournaments, stats.events, stats.sets, stats.dlq_messages) == (0, 0, 0, 0)


def test_only_eligible_tournaments_are_published():
    tournaments = [
        tournament(1, attendees=60),
        tournament(2, attendees=60, online=True),
        tournament(3, attendees=10),
        {"id": 4, "numAttendees": None, "isOnline": False, "events": []},
    ]
    stats, publisher = run(FakePaginator(tournaments))
    assert [key for _, key, _ in publisher.on(TOURNAMENTS_TOPIC)] == ["1"]
    assert stats.tournaments == 1


def test_sets_published_for_melee_events_only():
    other_game = {"id": 20, "name": "Ultimate", "videogame": {"id": 1386}}
    paginator = FakePaginator(
        [tournament(1, events=[melee_event(10), other_game])],
        sets_by_event={10: [{"id": 100}, {"id": "preview_10_1"}], 20: [{"id": 200}]},
    )
    stats, publisher = run(paginator)
    assert [key for _, key, _ in publisher.on(SETS_TOPIC)] == ["100", "preview_10_1"]
    assert (stats.events, stats.sets) == (1, 2)


def test_null_events_and_videogame_are_skipped():
    no_videogame = {"id": 11, "name": "E11", "videogame": None}
    paginator = FakePaginator([tournament(1, events=None), tournament(2, events=[no_videogame])])
    stats, publisher = run(paginator)
    assert stats.tournaments == 2
    assert publisher.on(SETS_TOPIC) == []


def test_set_envelope_fields():
    paginator = FakePaginator([tournament(1, events=[melee_event(10)])], {10: [{"id": 100}]})
    _, publisher = run(paginator)
    _, _, envelope = publisher.on(SETS_TOPIC)[0]
    assert envelope["entity"] == "set"
    assert envelope["dagster_run_id"] == "run-1"
    assert envelope["partition_week"] == WEEK
    assert envelope["query_name"] == "EventSets"
    assert envelope["query_variables"] == {"eventId": 10}
    assert envelope["ingested_at"] == "2025-01-13T06:00:00+00:00"
    assert envelope["payload"] == {"id": 100}


@pytest.mark.parametrize("error", [RetriesExhaustedError("boom"), PaginationCapError("cap")])
def test_event_failure_goes_to_dlq_and_run_continues(error):
    paginator = FakePaginator(
        [tournament(1, events=[melee_event(10), melee_event(11)])],
        sets_by_event={10: error, 11: [{"id": 110}]},
    )
    stats, publisher = run(paginator)
    ((_, key, dlq),) = publisher.on(DLQ_TOPIC)
    assert key == "10"
    assert dlq["entity"] == "event"
    assert dlq["payload"] is None
    assert dlq["error_type"] == type(error).__name__
    assert [k for _, k, _ in publisher.on(SETS_TOPIC)] == ["110"]
    assert stats.dlq_messages == 1


def test_invalid_token_propagates():
    paginator = FakePaginator(
        [tournament(1, events=[melee_event(10)])], sets_by_event={10: InvalidTokenError("bad")}
    )
    with pytest.raises(InvalidTokenError):
        run(paginator)


def test_discovery_failure_fails_the_run():
    with pytest.raises(RetriesExhaustedError):
        run(FakePaginator([], discovery_error=RetriesExhaustedError("down")))


def test_tournaments_and_events_without_id_are_skipped():
    no_id_event = {"id": None, "name": "E?", "videogame": {"id": 1}}
    paginator = FakePaginator(
        [
            {"id": None, "numAttendees": 100, "isOnline": False, "events": [melee_event(9)]},
            tournament(1, events=[no_id_event, melee_event(10)]),
        ],
        sets_by_event={9: [{"id": 90}], 10: [{"id": 100}]},
    )
    stats, publisher = run(paginator)
    assert [key for _, key, _ in publisher.on(TOURNAMENTS_TOPIC)] == ["1"]
    assert [key for _, key, _ in publisher.on(SETS_TOPIC)] == ["100"]
    assert all(envelope["entity_id"] != "None" for _, _, envelope in publisher.messages)
    assert (stats.tournaments, stats.events, stats.sets) == (1, 1, 1)
