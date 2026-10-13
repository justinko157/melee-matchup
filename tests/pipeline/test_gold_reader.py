import pandas as pd

from pipeline.gold.tables import PLAYER_RATINGS, PLAYER_RATINGS_SCHEMA
from pipeline.gold.writer import replace_table, to_arrow
from pipeline.serving.gold_reader import load_player_ratings, lookup_h2h

TS = pd.Timestamp("2025-01-06", tz="UTC")


def test_player_ratings_add_win_rate_and_filter_min_sets(sql_catalog):
    ratings = pd.DataFrame(
        [
            {"player_id": 1, "gamer_tag": "A", "elo": 1600.0, "sets_played": 10, "recent_wr": 0.6,
             "wins": 6, "losses": 4, "last_played_at": TS},
            {"player_id": 2, "gamer_tag": "B", "elo": 1700.0, "sets_played": 8, "recent_wr": 0.75,
             "wins": 6, "losses": 2, "last_played_at": TS},
            {"player_id": 3, "gamer_tag": "C", "elo": 1800.0, "sets_played": 2, "recent_wr": 1.0,
             "wins": 2, "losses": 0, "last_played_at": TS},
        ]
    )  # fmt: skip
    replace_table(sql_catalog, PLAYER_RATINGS, to_arrow(ratings, PLAYER_RATINGS_SCHEMA), {})
    players = load_player_ratings(sql_catalog)
    assert players["gamer_tag"].tolist() == ["B", "A"]  # C has too few sets; sorted by Elo
    assert players["total_sets"].tolist() == [8, 10]
    assert players["win_rate"].tolist() == [0.75, 0.6]


H2H = pd.DataFrame([{"player_a_id": 1, "player_b_id": 2, "a_wins": 3, "b_wins": 1, "total": 4}])


def test_lookup_h2h_in_both_orders():
    assert lookup_h2h(H2H, 1, 2) == (3, 1)
    assert lookup_h2h(H2H, 2, 1) == (1, 3)


def test_lookup_h2h_for_players_who_never_met():
    assert lookup_h2h(H2H, 1, 99) == (0, 0)
