"""Tests for the database module."""

import tempfile
from pathlib import Path

from src.database import MeleeDB


def _temp_db() -> tuple[MeleeDB, Path]:
    path = Path(tempfile.mktemp(suffix=".db"))
    return MeleeDB(path), path


def test_schema_creates_tables():
    db, path = _temp_db()
    try:
        stats = db.get_stats()
        assert set(stats.keys()) == {"tournaments", "events", "players", "sets", "games"}
        assert all(v == 0 for v in stats.values())
    finally:
        db.close()
        path.unlink(missing_ok=True)


def test_upsert_tournament_idempotent():
    db, path = _temp_db()
    try:
        t = {"id": 1, "name": "Test", "slug": "test", "numAttendees": 100,
             "startAt": 1000, "endAt": 2000}
        db.upsert_tournament(t)
        db.upsert_tournament(t)  # Insert again — should not duplicate
        db.commit()
        assert db.get_stats()["tournaments"] == 1

        # Update a field
        t["numAttendees"] = 200
        db.upsert_tournament(t)
        db.commit()
        row = db.conn.execute("SELECT num_attendees FROM tournaments WHERE id=1").fetchone()
        assert row[0] == 200
    finally:
        db.close()
        path.unlink(missing_ok=True)


def test_upsert_player_updates_tag():
    db, path = _temp_db()
    try:
        db.upsert_player(1, "OldTag")
        db.upsert_player(1, "NewTag")
        db.commit()
        assert db.get_stats()["players"] == 1
        row = db.conn.execute("SELECT gamer_tag FROM players WHERE id=1").fetchone()
        assert row[0] == "NewTag"
    finally:
        db.close()
        path.unlink(missing_ok=True)


def test_event_has_sets():
    db, path = _temp_db()
    try:
        assert not db.event_has_sets(999)

        db.upsert_tournament({"id": 1, "name": "T", "slug": "t"})
        db.upsert_event({"id": 10, "name": "E", "slug": "e", "numEntrants": 50}, 1)
        db.upsert_player(100, "Alice")
        db.upsert_player(200, "Bob")
        db.upsert_set({
            "id": 1, "event_id": 10, "round": 1, "player1_id": 100,
            "player2_id": 200, "winner_player_id": 100, "state": 3,
        })
        db.commit()
        assert db.event_has_sets(10)
        assert not db.event_has_sets(99)
    finally:
        db.close()
        path.unlink(missing_ok=True)
