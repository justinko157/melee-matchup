"""Tests for the feature engineering pipeline."""

import tempfile
from pathlib import Path

from src.database import MeleeDB
from src.features import FeatureBuilder


def _create_test_db() -> Path:
    """Create a small test database with known data."""
    db_path = Path(tempfile.mktemp(suffix=".db"))
    db = MeleeDB(db_path)

    # Two tournaments, one early, one late
    db.upsert_tournament({"id": 1, "name": "T1", "slug": "t1", "numAttendees": 200,
                          "startAt": 1000000, "endAt": 1000100})
    db.upsert_tournament({"id": 2, "name": "T2", "slug": "t2", "numAttendees": 300,
                          "startAt": 2000000, "endAt": 2000100})

    db.upsert_event({"id": 10, "name": "E1", "slug": "e1", "numEntrants": 50}, 1)
    db.upsert_event({"id": 20, "name": "E2", "slug": "e2", "numEntrants": 60}, 2)

    db.upsert_player(100, "Alice")
    db.upsert_player(200, "Bob")
    db.upsert_player(300, "Charlie")

    # Alice beats Bob at T1
    db.upsert_set({
        "id": 1, "event_id": 10, "round": 1, "full_round_text": "R1",
        "phase_name": "Pools", "completed_at": 1000010, "total_games": 3,
        "display_score": "Alice 2 - Bob 1",
        "entrant1_id": 101, "entrant2_id": 201,
        "player1_id": 100, "player2_id": 200, "winner_player_id": 100,
        "player1_score": 2, "player2_score": 1,
        "player1_seed": 1, "player2_seed": 4, "state": 3,
    })

    # Bob beats Charlie at T2
    db.upsert_set({
        "id": 2, "event_id": 20, "round": 1, "full_round_text": "R1",
        "phase_name": "Pools", "completed_at": 2000010, "total_games": 3,
        "display_score": "Bob 2 - Charlie 0",
        "entrant1_id": 202, "entrant2_id": 302,
        "player1_id": 200, "player2_id": 300, "winner_player_id": 200,
        "player1_score": 2, "player2_score": 0,
        "player1_seed": 2, "player2_seed": 3, "state": 3,
    })

    # Alice beats Bob again at T2
    db.upsert_set({
        "id": 3, "event_id": 20, "round": 2, "full_round_text": "R2",
        "phase_name": "Bracket", "completed_at": 2000020, "total_games": 3,
        "display_score": "Alice 2 - Bob 1",
        "entrant1_id": 103, "entrant2_id": 203,
        "player1_id": 100, "player2_id": 200, "winner_player_id": 100,
        "player1_score": 2, "player2_score": 1,
        "player1_seed": 1, "player2_seed": 2, "state": 3,
    })

    db.commit()
    db.close()
    return db_path


def test_features_no_leakage():
    """Features for a set should only use data from BEFORE that set."""
    db_path = _create_test_db()
    try:
        builder = FeatureBuilder(db_path)
        df = builder.build_features()
        builder.close()

        # First set: Alice vs Bob — both should have default Elo (1500) since no prior data
        first = df.iloc[0]
        assert first["p1_elo"] == 1500.0, f"Expected 1500, got {first['p1_elo']}"
        assert first["p2_elo"] == 1500.0
        assert first["p1_sets_played"] == 0  # No games played yet
        assert first["h2h_total"] == 0  # No prior H2H

        # Third set: Alice vs Bob again — should have H2H history from first set
        third = df.iloc[2]
        assert third["h2h_total"] == 1  # Only 1 prior set between them
        assert third["p1_elo"] != 1500.0  # Elo should have updated
    finally:
        db_path.unlink(missing_ok=True)


def test_features_target_correct():
    """The p1_won target should correctly reflect the winner."""
    db_path = _create_test_db()
    try:
        builder = FeatureBuilder(db_path)
        df = builder.build_features()
        builder.close()

        # Set 1: Alice (p1) beat Bob (p2) → p1_won = 1
        assert df.iloc[0]["p1_won"] == 1

        # Set 2: Bob (p1) beat Charlie (p2) → p1_won = 1
        assert df.iloc[1]["p1_won"] == 1

        # Set 3: Alice (p1) beat Bob (p2) → p1_won = 1
        assert df.iloc[2]["p1_won"] == 1
    finally:
        db_path.unlink(missing_ok=True)


def test_features_chronological_order():
    """Features should be in chronological order."""
    db_path = _create_test_db()
    try:
        builder = FeatureBuilder(db_path)
        df = builder.build_features()
        builder.close()

        assert df["completed_at"].is_monotonic_increasing
    finally:
        db_path.unlink(missing_ok=True)
