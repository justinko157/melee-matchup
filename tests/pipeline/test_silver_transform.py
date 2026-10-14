import json
from datetime import UTC, datetime

import pytest

pytestmark = pytest.mark.spark
pytest.importorskip("pyspark")

from pipeline.spark_jobs.silver import (  # noqa: E402
    EXCLUDED_TOURNAMENT_IDS,
    drop_counts,
    event_rows,
    excluded_event_ids,
    game_rows,
    newest_per_key,
    parse_sets,
    parse_tournaments,
    player_rows,
    set_rows,
    tournament_rows,
)

BRONZE_SCHEMA = (
    "kafka_partition INT, kafka_offset BIGINT, ingested_at TIMESTAMP, "
    "query_variables STRING, payload STRING"
)
T0 = datetime(2025, 1, 6, 12, tzinfo=UTC)
T1 = datetime(2025, 1, 7, 12, tzinfo=UTC)


def bronze(spark, rows):
    """rows: (payload dict, query_variables dict, ingested_at, offset)."""
    return spark.createDataFrame(
        [(0, offset, ts, json.dumps(qv), json.dumps(p)) for p, qv, ts, offset in rows],
        BRONZE_SCHEMA,
    )


def slot(entrant_id, player_id, tag, seed=None, initial_seed=None, score=None):
    participants = [{"player": {"id": player_id, "gamerTag": tag}}] if player_id else []
    return {
        "seed": {"seedNum": seed} if seed is not None else None,
        "standing": {"placement": 1, "stats": {"score": {"value": score}}},
        "entrant": {
            "id": entrant_id,
            "name": tag,
            "initialSeedNum": initial_seed,
            "participants": participants,
        },
    }


def set_node(set_id=100, state=3, winner=11, slots=None, games=None, completed=1736200000):
    return {
        "id": set_id,
        "displayScore": "A 2 - B 1",
        "winnerId": winner,
        "round": 1,
        "fullRoundText": "Winners Round 1",
        "completedAt": completed,
        "totalGames": 3,
        "state": state,
        "games": games,
        "slots": slots
        if slots is not None
        else [
            slot(11, 1, "Alice", seed=1, score=2.0),
            slot(12, 2, "Bob", initial_seed=4, score=1.0),
        ],
        "phaseGroup": {"phase": {"name": "Pools"}},
    }


def tournament_node(tid=7, events=None):
    return {
        "id": tid,
        "name": "Genesis",
        "slug": "tournament/genesis",
        "numAttendees": 500,
        "startAt": 1736150400,
        "endAt": 1736323200,
        "city": "San Jose",
        "addrState": "CA",
        "countryCode": "US",
        "isOnline": False,
        "events": events
        if events is not None
        else [
            {
                "id": 70,
                "name": "Melee Singles",
                "slug": "s",
                "numEntrants": 400,
                "videogame": {"id": 1},
            },
            {
                "id": 71,
                "name": "Ultimate",
                "slug": "u",
                "numEntrants": 900,
                "videogame": {"id": 1386},
            },
        ],
    }


def test_tournament_rows_are_typed(spark):
    parsed = parse_tournaments(bronze(spark, [(tournament_node(), {}, T0, 1)]))
    (row,) = tournament_rows(parsed).collect()
    assert row.tournament_id == 7
    assert row.state == "CA"
    assert row.num_attendees == 500
    assert row.start_at == datetime(2025, 1, 6, 8, 0)  # UTC session time zone, naive
    assert row.is_online is False
    assert row.source_ingested_at == datetime(2025, 1, 6, 12, 0)


def test_events_keep_only_melee(spark):
    parsed = parse_tournaments(bronze(spark, [(tournament_node(), {}, T0, 1)]))
    rows = event_rows(parsed).collect()
    assert [(r.event_id, r.tournament_id, r.num_entrants) for r in rows] == [(70, 7, 400)]


def test_completed_set_becomes_a_row(spark):
    parsed = parse_sets(bronze(spark, [(set_node(), {"eventId": 70}, T0, 1)]))
    (row,) = set_rows(parsed).collect()
    assert row.set_id == 100
    assert row.event_id == 70
    assert (row.player1_id, row.player2_id, row.winner_player_id) == (1, 2, 1)
    assert (row.player1_seed, row.player2_seed) == (1, 4)  # seedNum, then initialSeedNum
    assert (row.player1_score, row.player2_score) == (2.0, 1.0)
    assert row.phase_name == "Pools"
    assert row.completed_at == datetime(2025, 1, 6, 21, 46, 40)


def test_unplayable_sets_are_dropped_with_a_reason(spark):
    rows = [
        (set_node(set_id=1, slots=[slot(11, 1, "Alice")]), {"eventId": 70}, T0, 1),
        (
            set_node(set_id=2, slots=[slot(11, 1, "A"), slot(12, None, "bye")]),
            {"eventId": 70},
            T0,
            2,
        ),
        (set_node(set_id=3, state=2), {"eventId": 70}, T0, 3),
        (set_node(set_id="preview_9_1"), {"eventId": 70}, T0, 4),
        (set_node(set_id=5, winner=999), {"eventId": 70}, T0, 5),
        (set_node(set_id=6), {"eventId": 70}, T0, 6),
    ]
    parsed = parse_sets(bronze(spark, rows))
    total, dropped, excluded = drop_counts(parsed)
    assert (total, excluded) == (6, 0)
    assert dropped == {
        "too_few_slots": 1,
        "missing_player": 1,
        "not_completed": 2,
        "unresolved_winner": 1,
    }
    assert [r.set_id for r in set_rows(parsed).collect()] == [6]


def test_games_match_characters_by_entrant(spark):
    games = [
        {
            "id": 900,
            "orderNum": 1,
            "winnerId": 12,
            "stage": {"id": 3, "name": "Battlefield"},
            # Listed with player 2's entrant first, to prove matching is by entrant id.
            "selections": [
                {"entrant": {"id": 12}, "character": {"id": 2, "name": "Fox"}},
                {"entrant": {"id": 11}, "character": {"id": 1, "name": "Marth"}},
            ],
        },
        {"id": None, "orderNum": 2, "winnerId": 11, "stage": None, "selections": None},
    ]
    parsed = parse_sets(bronze(spark, [(set_node(games=games), {"eventId": 70}, T0, 1)]))
    (row,) = game_rows(parsed).collect()
    assert (row.game_id, row.set_id, row.game_number) == (900, 100, 1)
    assert (row.player1_character_name, row.player2_character_name) == ("Marth", "Fox")
    assert row.stage_name == "Battlefield"
    assert row.completed_at == datetime(2025, 1, 6, 21, 46, 40)


def test_dropped_sets_contribute_no_games_or_players(spark):
    games = [{"id": 901, "orderNum": 1, "winnerId": 11, "stage": None, "selections": []}]
    parsed = parse_sets(bronze(spark, [(set_node(state=2, games=games), {"eventId": 70}, T0, 1)]))
    assert game_rows(parsed).count() == 0
    assert player_rows(parsed).count() == 0


def test_players_take_the_newest_tag(spark):
    old = set_node(set_id=1)
    new = set_node(set_id=2, slots=[slot(11, 1, "Alice2"), slot(12, 2, "Bob")])
    parsed = parse_sets(
        bronze(spark, [(old, {"eventId": 70}, T0, 1), (new, {"eventId": 70}, T1, 2)])
    )
    players = {
        r.player_id: r.gamer_tag for r in newest_per_key(player_rows(parsed), "player_id").collect()
    }
    assert players == {1: "Alice2", 2: "Bob"}


def test_newest_version_wins_then_offset_breaks_ties(spark):
    v1 = set_node(
        set_id=1, slots=[slot(11, 1, "A", score=1.0), slot(12, 2, "B", score=2.0)], winner=12
    )
    v2 = set_node(set_id=1, slots=[slot(11, 1, "A", score=2.0), slot(12, 2, "B", score=0.0)])
    v3 = set_node(set_id=1, slots=[slot(11, 1, "A", score=3.0), slot(12, 2, "B", score=0.0)])
    rows = [
        (v2, {"eventId": 70}, T1, 5),
        (v1, {"eventId": 70}, T0, 9),  # older ingest loses despite the higher offset
        (v3, {"eventId": 70}, T1, 6),  # same ingest time as v2: higher offset wins
    ]
    (row,) = newest_per_key(set_rows(parse_sets(bronze(spark, rows))), "set_id").collect()
    assert row.player1_score == 3.0


def test_event_id_comes_from_query_variables(spark):
    parsed = parse_sets(bronze(spark, [(set_node(), {"eventId": 12345}, T0, 1)]))
    assert set_rows(parsed).first().event_id == 12345


def test_excluded_tournaments_and_their_events_are_skipped(spark):
    excluded = EXCLUDED_TOURNAMENT_IDS[0]
    rows = [(tournament_node(), {}, T0, 1), (tournament_node(tid=excluded), {}, T0, 2)]
    parsed = parse_tournaments(bronze(spark, rows))
    assert [r.tournament_id for r in tournament_rows(parsed).collect()] == [7]
    assert {r.tournament_id for r in event_rows(parsed).collect()} == {7}


def test_excluded_event_ids_come_from_excluded_tournaments_only(spark):
    excluded = tournament_node(tid=EXCLUDED_TOURNAMENT_IDS[0], events=[
        {"id": 80, "name": "Melee", "slug": "m", "numEntrants": 9, "videogame": {"id": 1}},
        {"id": 81, "name": "Melee 2", "slug": "m2", "numEntrants": 9, "videogame": {"id": 1}},
    ])  # fmt: skip
    rows = [(tournament_node(), {}, T0, 1), (excluded, {}, T0, 2), (excluded, {}, T1, 3)]
    ids = sorted(r.event_id for r in excluded_event_ids(bronze(spark, rows)).collect())
    assert ids == [80, 81]


def test_sets_of_excluded_events_are_counted_apart_from_drops(spark):
    rows = [
        (set_node(set_id=1), {"eventId": 80}, T0, 1),
        (set_node(set_id=2, state=2), {"eventId": 80}, T0, 2),  # excluded beats not_completed
        (set_node(set_id=3), {"eventId": 70}, T0, 3),
    ]
    parsed = parse_sets(bronze(spark, rows), excluded_events=(80,))
    total, dropped, excluded = drop_counts(parsed)
    assert (total, excluded, sum(dropped.values())) == (3, 2, 0)
    assert [r.set_id for r in set_rows(parsed).collect()] == [3]
    assert {r.player_id for r in player_rows(parsed).collect()} == {1, 2}
