import random

import numpy as np
import pandas as pd

from pipeline.gold.features import FEATURE_COLUMNS, SET_INPUT_COLUMNS, build_features
from pipeline.gold.schema import features_schema
from pipeline.model_features import CORE_FEATURES

BASE = {
    "event_id": 10,
    "tournament_id": 1,
    "tournament_name": "T1",
    "num_attendees": 200,
    "player1_score": 2.0,
    "player2_score": 1.0,
    "player1_seed": 1,
    "player2_seed": 4,
}


def sets_frame(rows):
    frame = pd.DataFrame([{**BASE, **row} for row in rows])
    frame["completed_at"] = pd.to_datetime(frame["completed_at"], unit="s", utc=True)
    return frame


THREE_SETS = [
    # Alice (100) beats Bob (200) at T1
    {
        "set_id": 1,
        "completed_at": 1000010,
        "player1_id": 100,
        "player2_id": 200,
        "winner_player_id": 100,
    },
    # Bob beats Charlie (300) at T2
    {
        "set_id": 2,
        "completed_at": 2000010,
        "player1_id": 200,
        "player2_id": 300,
        "winner_player_id": 200,
    },
    # Alice beats Bob again at T2
    {
        "set_id": 3,
        "completed_at": 2000020,
        "player1_id": 100,
        "player2_id": 200,
        "winner_player_id": 100,
    },
]


def test_features_no_leakage():
    df, _ = build_features(sets_frame(THREE_SETS))
    first, third = df.iloc[0], df.iloc[2]
    assert first["p1_elo"] == 1500.0
    assert first["p2_elo"] == 1500.0
    assert first["p1_sets_played"] == 0
    assert first["h2h_total"] == 0
    assert third["h2h_total"] == 1
    assert third["p1_elo"] != 1500.0


def test_features_target_correct():
    df, _ = build_features(sets_frame(THREE_SETS))
    assert df["p1_won"].tolist() == [1, 1, 1]


def test_features_chronological_order():
    df, _ = build_features(sets_frame(list(reversed(THREE_SETS))))
    assert df["completed_at"].is_monotonic_increasing
    assert df["set_id"].tolist() == [1, 2, 3]
    assert tuple(df.columns) == FEATURE_COLUMNS


def test_equal_completed_at_breaks_ties_by_set_id():
    rows = [
        {
            "set_id": 7,
            "completed_at": 5000,
            "player1_id": 1,
            "player2_id": 2,
            "winner_player_id": 1,
        },
        {
            "set_id": 5,
            "completed_at": 5000,
            "player1_id": 1,
            "player2_id": 3,
            "winner_player_id": 3,
        },
    ]
    df, _ = build_features(sets_frame(rows))
    assert df["set_id"].tolist() == [5, 7]
    assert df.iloc[1]["p1_sets_played"] == 1  # set 5 happened first


def test_each_row_only_counts_earlier_sets():
    rng = random.Random(3)
    rows = []
    for i in range(40):
        p1, p2 = rng.sample([1, 2, 3, 4], 2)
        rows.append({"set_id": i, "completed_at": 1000 + i, "player1_id": p1, "player2_id": p2,
                     "winner_player_id": rng.choice([p1, p2])})  # fmt: skip
    df, _ = build_features(sets_frame(rows))
    played: dict[int, int] = {}
    for row in df.itertuples():
        assert row.p1_sets_played == played.get(row.player1_id, 0)
        assert row.p2_sets_played == played.get(row.player2_id, 0)
        played[row.player1_id] = played.get(row.player1_id, 0) + 1
        played[row.player2_id] = played.get(row.player2_id, 0) + 1


def test_null_seeds_and_attendees_never_produce_nan():
    row = {**THREE_SETS[0], "player1_seed": None, "num_attendees": None}
    df, _ = build_features(sets_frame([row]))
    assert df.iloc[0]["seed_diff"] == -4  # missing seed counts as 0
    assert df.iloc[0]["num_attendees"] == 0
    assert not df[CORE_FEATURES].isna().any().any()


def test_ineligible_sets_are_excluded():
    rows = [
        THREE_SETS[0],
        {**THREE_SETS[1], "winner_player_id": None},
        {**THREE_SETS[2], "completed_at": np.nan},
    ]
    df, _ = build_features(sets_frame(rows))
    assert df["set_id"].tolist() == [1]


def test_state_tracks_records_for_gold_tables():
    _, state = build_features(sets_frame(THREE_SETS))
    assert (state.wins[100], state.losses[100]) == (2, 0)
    assert (state.wins[200], state.losses[200]) == (1, 2)
    assert state.h2h[(100, 200)] == [2, 0]
    assert state.pair_last_played[(100, 200)] == pd.Timestamp(2000020, unit="s", tz="UTC")
    assert state.last_played[300] == pd.Timestamp(2000010, unit="s", tz="UTC")


def test_all_default_ratings_still_satisfy_the_schema():
    rows = [
        {
            "set_id": 1,
            "completed_at": 1000,
            "player1_id": 1,
            "player2_id": 2,
            "winner_player_id": 1,
        },
        {
            "set_id": 2,
            "completed_at": 1001,
            "player1_id": 3,
            "player2_id": 4,
            "winner_player_id": 4,
        },
    ]
    features, _ = build_features(sets_frame(rows))
    features_schema.validate(features)


def test_empty_input_yields_a_schema_valid_empty_frame():
    empty = pd.DataFrame({column: [] for column in SET_INPUT_COLUMNS})
    features, _ = build_features(empty)
    assert len(features) == 0
    assert tuple(features.columns) == FEATURE_COLUMNS
    features_schema.validate(features)


def test_recent_window_keeps_only_the_last_30_results():
    # Alice loses 10 sets, then wins 30: only the 30 wins are in her window.
    rows = [
        {
            "set_id": i,
            "completed_at": 1000000 + i,
            "player1_id": 100,
            "player2_id": 200,
            "winner_player_id": 200 if i < 10 else 100,
        }
        for i in range(40)
    ]
    features, state = build_features(sets_frame(rows))
    assert len(state.recent_wins[100]) == 30
    assert list(state.recent_wins[100]) == [True] * 30
    # Before the last set her window is sets 9-38: one loss, then 29 wins.
    assert features["p1_recent_wr"].iloc[-1] == 29 / 30
