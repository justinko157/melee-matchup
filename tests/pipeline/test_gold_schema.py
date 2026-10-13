import pandas as pd
import pandera.pandas as pandera
import pytest

from pipeline.gold.features import FEATURE_COLUMNS
from pipeline.gold.schema import features_schema


def _make_valid_row(**overrides):
    row = {
        "set_id": 1,
        "event_id": 100,
        "tournament_id": 10,
        "tournament_name": "Genesis 9",
        "completed_at": pd.Timestamp("2023-11-14 22:13:20", tz="UTC"),
        "player1_id": 1,
        "player2_id": 2,
        "p1_elo": 1800.0,
        "p2_elo": 1600.0,
        "elo_diff": 200.0,
        "p1_expected": 0.76,
        "p1_sets_played": 50,
        "p2_sets_played": 30,
        "p1_recent_wr": 0.7,
        "p2_recent_wr": 0.5,
        "recent_wr_diff": 0.2,
        "p1_h2h_wins": 3,
        "p2_h2h_wins": 1,
        "p1_h2h_wr": 0.75,
        "h2h_total": 4,
        "p1_seed": 1,
        "p2_seed": 5,
        "seed_diff": -4,
        "num_attendees": 500,
        "p1_score": 3.0,
        "p2_score": 1.0,
        "p1_won": 1,
    }
    row.update(overrides)
    return row


def test_valid_row_passes():
    features_schema.validate(pd.DataFrame([_make_valid_row()]))


def test_negative_elo_fails():
    with pytest.raises(pandera.errors.SchemaError):
        features_schema.validate(pd.DataFrame([_make_valid_row(p1_elo=-100.0)]))


def test_invalid_target_fails():
    with pytest.raises(pandera.errors.SchemaError):
        features_schema.validate(pd.DataFrame([_make_valid_row(p1_won=2)]))


def test_same_players_fails():
    with pytest.raises(pandera.errors.SchemaError):
        features_schema.validate(pd.DataFrame([_make_valid_row(player1_id=1, player2_id=1)]))


def test_h2h_inconsistency_fails():
    with pytest.raises(pandera.errors.SchemaError):
        features_schema.validate(pd.DataFrame([_make_valid_row(h2h_total=10)]))


def test_missing_seed_counts_as_zero_in_seed_diff():
    features_schema.validate(pd.DataFrame([_make_valid_row(p1_seed=None, seed_diff=-5)]))


def test_schema_columns_match_feature_columns():
    assert tuple(features_schema.columns) == FEATURE_COLUMNS
