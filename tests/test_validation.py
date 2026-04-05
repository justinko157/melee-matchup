"""Tests for the pandera data quality schema."""

import pandas as pd
import pandera.pandas as pa
import pytest

from src.validation import features_schema


def _make_valid_row(**overrides):
    """Build a single valid feature row, with optional overrides."""
    row = {
        "set_id": 1,
        "event_id": 100,
        "tournament_id": 10,
        "tournament_name": "Genesis 9",
        "completed_at": 1700000000,
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
    df = pd.DataFrame([_make_valid_row()])
    features_schema.validate(df)


def test_negative_elo_fails():
    df = pd.DataFrame([_make_valid_row(p1_elo=-100.0)])
    with pytest.raises(pa.errors.SchemaError):
        features_schema.validate(df)


def test_invalid_target_fails():
    df = pd.DataFrame([_make_valid_row(p1_won=2)])
    with pytest.raises(pa.errors.SchemaError):
        features_schema.validate(df)


def test_same_players_fails():
    df = pd.DataFrame([_make_valid_row(player1_id=1, player2_id=1)])
    with pytest.raises(pa.errors.SchemaError):
        features_schema.validate(df)


def test_h2h_inconsistency_fails():
    df = pd.DataFrame([_make_valid_row(h2h_total=10)])  # 3+1 != 10
    with pytest.raises(pa.errors.SchemaError):
        features_schema.validate(df)
