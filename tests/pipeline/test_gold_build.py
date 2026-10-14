import pandas as pd
from pyiceberg.transforms import YearTransform

from pipeline.gold.build import build_gold, gold_checks
from pipeline.gold.tables import (
    HEAD_TO_HEAD,
    PLAYER_RATINGS,
    SET_FEATURES,
    SILVER_SNAPSHOT_PROPERTY,
)
from pipeline.lakehouse import current_snapshot_id
from tests.pipeline.lakehouse_data import silver_set, write_silver

EVENTS = [{"event_id": 10, "tournament_id": 1}]
TOURNAMENTS = [{"tournament_id": 1, "name": "Genesis", "num_attendees": 300}]
PLAYERS = [{"player_id": 100, "gamer_tag": "Alice"}, {"player_id": 200, "gamer_tag": "Bob"}]
SETS = [
    silver_set(1, 100, 200, 100, 1_000_000),
    silver_set(2, 200, 100, 100, 2_000_000),
    silver_set(3, 100, 300, 300, 3_000_000),  # player 300 has no silver.players row
]


def rows(catalog, identifier):
    return catalog.load_table(identifier).scan().to_pandas()


def test_build_gold_writes_three_tables(sql_catalog):
    write_silver(sql_catalog, sets=SETS, events=EVENTS, tournaments=TOURNAMENTS, players=PLAYERS)
    build = build_gold(sql_catalog)
    assert len(rows(sql_catalog, SET_FEATURES)) == 3
    assert len(rows(sql_catalog, PLAYER_RATINGS)) == 3
    assert len(rows(sql_catalog, HEAD_TO_HEAD)) == 2
    silver_snapshot = current_snapshot_id(sql_catalog, "silver.sets")
    assert build.silver_sets_snapshot == silver_snapshot
    properties = sql_catalog.load_table(SET_FEATURES).properties
    assert properties[SILVER_SNAPSHOT_PROPERTY] == str(silver_snapshot)
    assert all(passed for passed, _ in gold_checks(build).values())


def test_rebuild_replaces_instead_of_appending(sql_catalog):
    write_silver(sql_catalog, sets=SETS, events=EVENTS, tournaments=TOURNAMENTS, players=PLAYERS)
    build_gold(sql_catalog)
    build_gold(sql_catalog)
    assert len(rows(sql_catalog, SET_FEATURES)) == 3
    assert len(rows(sql_catalog, HEAD_TO_HEAD)) == 2


def test_player_ratings_and_head_to_head(sql_catalog):
    write_silver(sql_catalog, sets=SETS, events=EVENTS, tournaments=TOURNAMENTS, players=PLAYERS)
    build_gold(sql_catalog)
    ratings = rows(sql_catalog, PLAYER_RATINGS).set_index("player_id")
    assert ratings.loc[100, "gamer_tag"] == "Alice"
    assert ratings.loc[300, "gamer_tag"] == "300"  # no tag in silver: fall back to the id
    assert (ratings.loc[100, "wins"], ratings.loc[100, "losses"]) == (2, 1)
    assert ratings.loc[100, "sets_played"] == 3
    assert ratings.loc[100, "last_played_at"] == pd.Timestamp(3_000_000, unit="s", tz="UTC")
    h2h = rows(sql_catalog, HEAD_TO_HEAD).set_index(["player_a_id", "player_b_id"])
    assert tuple(h2h.loc[(100, 200), ["a_wins", "b_wins", "total"]]) == (2, 0, 2)


def test_orphan_set_breaks_features_match_silver(sql_catalog):
    orphan = silver_set(4, 100, 200, 200, 4_000_000, event_id=99)
    write_silver(
        sql_catalog, sets=[*SETS, orphan], events=EVENTS, tournaments=TOURNAMENTS, players=PLAYERS
    )
    passed, metadata = gold_checks(build_gold(sql_catalog))["features_match_silver"]
    assert not passed
    assert metadata == {"gold_rows": 3, "eligible_silver_sets": 4}


def test_empty_silver_writes_empty_gold(sql_catalog):
    write_silver(sql_catalog)
    build = build_gold(sql_catalog)
    for identifier in (SET_FEATURES, PLAYER_RATINGS, HEAD_TO_HEAD):
        assert len(rows(sql_catalog, identifier)) == 0
    assert all(passed for passed, _ in gold_checks(build).values())


def test_set_features_is_partitioned_by_year(sql_catalog):
    write_silver(sql_catalog, sets=SETS, events=EVENTS, tournaments=TOURNAMENTS, players=PLAYERS)
    build_gold(sql_catalog)
    (field,) = sql_catalog.load_table(SET_FEATURES).spec().fields
    assert isinstance(field.transform, YearTransform)


def test_unchanged_silver_skips_the_rebuild(sql_catalog):
    write_silver(sql_catalog, sets=SETS, events=EVENTS, tournaments=TOURNAMENTS, players=PLAYERS)
    first = build_gold(sql_catalog)
    second = build_gold(sql_catalog)
    assert (first.skipped, second.skipped) == (False, True)
    assert second.snapshots == first.snapshots  # nothing was written
    assert second.row_counts == first.row_counts
    assert second.checks == first.checks
    assert second.eligible_silver_sets == first.eligible_silver_sets == 3


def test_a_change_to_any_silver_input_rebuilds(sql_catalog):
    write_silver(sql_catalog, sets=SETS, events=EVENTS, tournaments=TOURNAMENTS, players=PLAYERS)
    first = build_gold(sql_catalog)
    write_silver(sql_catalog, players=[{"player_id": 300, "gamer_tag": "Carol"}])
    second = build_gold(sql_catalog)
    assert not second.skipped
    assert second.snapshots[PLAYER_RATINGS] != first.snapshots[PLAYER_RATINGS]
    ratings = rows(sql_catalog, PLAYER_RATINGS).set_index("player_id")
    assert ratings.loc[300, "gamer_tag"] == "Carol"


def test_force_rebuilds_unchanged_silver(sql_catalog):
    write_silver(sql_catalog, sets=SETS, events=EVENTS, tournaments=TOURNAMENTS, players=PLAYERS)
    first = build_gold(sql_catalog)
    forced = build_gold(sql_catalog, force=True)
    assert not forced.skipped
    assert forced.snapshots[SET_FEATURES] != first.snapshots[SET_FEATURES]


def test_a_skipped_build_reports_the_recorded_failing_checks(sql_catalog):
    orphan = silver_set(4, 100, 200, 200, 4_000_000, event_id=99)
    write_silver(
        sql_catalog, sets=[*SETS, orphan], events=EVENTS, tournaments=TOURNAMENTS, players=PLAYERS
    )
    build_gold(sql_catalog)
    skipped = build_gold(sql_catalog)
    assert skipped.skipped
    passed, metadata = gold_checks(skipped)["features_match_silver"]
    assert not passed
    assert metadata == {"gold_rows": 3, "eligible_silver_sets": 4}
