"""SQLite database for storing tournament data."""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "raw" / "melee.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tournaments (
    id              INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    slug            TEXT UNIQUE,
    num_attendees   INTEGER,
    start_at        INTEGER,  -- Unix timestamp
    end_at          INTEGER,
    city            TEXT,
    state           TEXT,
    country_code    TEXT,
    is_online       BOOLEAN
);

CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY,
    tournament_id   INTEGER NOT NULL REFERENCES tournaments(id),
    name            TEXT NOT NULL,
    slug            TEXT UNIQUE,
    num_entrants    INTEGER
);

CREATE TABLE IF NOT EXISTS players (
    id              INTEGER PRIMARY KEY,  -- start.gg player ID (stable across tournaments)
    gamer_tag       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sets (
    id              INTEGER PRIMARY KEY,
    event_id        INTEGER NOT NULL REFERENCES events(id),
    round           INTEGER,
    full_round_text TEXT,
    phase_name      TEXT,
    completed_at    INTEGER,
    total_games     INTEGER,
    display_score   TEXT,
    -- Entrant IDs (event-specific, used to link to winner/loser)
    entrant1_id     INTEGER,
    entrant2_id     INTEGER,
    -- Stable player IDs
    player1_id      INTEGER REFERENCES players(id),
    player2_id      INTEGER REFERENCES players(id),
    winner_player_id INTEGER REFERENCES players(id),
    -- Scores
    player1_score   INTEGER,
    player2_score   INTEGER,
    -- Seeds
    player1_seed    INTEGER,
    player2_seed    INTEGER,
    -- State: 3 = completed
    state           INTEGER
);

CREATE TABLE IF NOT EXISTS games (
    id              INTEGER PRIMARY KEY,
    set_id          INTEGER NOT NULL REFERENCES sets(id),
    game_number     INTEGER NOT NULL,
    winner_entrant_id INTEGER,
    stage_id        INTEGER,
    stage_name      TEXT,
    -- Character selections (stored as start.gg character IDs)
    player1_character_id    INTEGER,
    player1_character_name  TEXT,
    player2_character_id    INTEGER,
    player2_character_name  TEXT
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_sets_event ON sets(event_id);
CREATE INDEX IF NOT EXISTS idx_sets_player1 ON sets(player1_id);
CREATE INDEX IF NOT EXISTS idx_sets_player2 ON sets(player2_id);
CREATE INDEX IF NOT EXISTS idx_sets_winner ON sets(winner_player_id);
CREATE INDEX IF NOT EXISTS idx_games_set ON games(set_id);
CREATE INDEX IF NOT EXISTS idx_events_tournament ON events(tournament_id);
"""


class MeleeDB:
    """SQLite database wrapper for Melee tournament data."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def upsert_tournament(self, t: dict):
        self.conn.execute(
            """
            INSERT INTO tournaments (id, name, slug, num_attendees, start_at, end_at,
                                     city, state, country_code, is_online)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name, slug=excluded.slug,
                num_attendees=excluded.num_attendees,
                start_at=excluded.start_at, end_at=excluded.end_at,
                city=excluded.city, state=excluded.state,
                country_code=excluded.country_code, is_online=excluded.is_online
            """,
            (
                t["id"], t["name"], t.get("slug"), t.get("numAttendees"),
                t.get("startAt"), t.get("endAt"), t.get("city"),
                t.get("addrState"), t.get("countryCode"), t.get("isOnline"),
            ),
        )

    def upsert_event(self, event: dict, tournament_id: int):
        self.conn.execute(
            """
            INSERT INTO events (id, tournament_id, name, slug, num_entrants)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name, slug=excluded.slug,
                num_entrants=excluded.num_entrants
            """,
            (
                event["id"], tournament_id, event["name"],
                event.get("slug"), event.get("numEntrants"),
            ),
        )

    def upsert_player(self, player_id: int, gamer_tag: str):
        self.conn.execute(
            """
            INSERT INTO players (id, gamer_tag)
            VALUES (?, ?)
            ON CONFLICT(id) DO UPDATE SET gamer_tag=excluded.gamer_tag
            """,
            (player_id, gamer_tag),
        )

    def upsert_set(self, s: dict):
        self.conn.execute(
            """
            INSERT INTO sets (id, event_id, round, full_round_text, phase_name,
                              completed_at, total_games, display_score,
                              entrant1_id, entrant2_id,
                              player1_id, player2_id, winner_player_id,
                              player1_score, player2_score,
                              player1_seed, player2_seed, state)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                round=excluded.round, full_round_text=excluded.full_round_text,
                phase_name=excluded.phase_name, completed_at=excluded.completed_at,
                total_games=excluded.total_games, display_score=excluded.display_score,
                entrant1_id=excluded.entrant1_id, entrant2_id=excluded.entrant2_id,
                player1_id=excluded.player1_id, player2_id=excluded.player2_id,
                winner_player_id=excluded.winner_player_id,
                player1_score=excluded.player1_score, player2_score=excluded.player2_score,
                player1_seed=excluded.player1_seed, player2_seed=excluded.player2_seed,
                state=excluded.state
            """,
            (
                s["id"], s["event_id"], s.get("round"), s.get("full_round_text"),
                s.get("phase_name"), s.get("completed_at"), s.get("total_games"),
                s.get("display_score"),
                s.get("entrant1_id"), s.get("entrant2_id"),
                s.get("player1_id"), s.get("player2_id"), s.get("winner_player_id"),
                s.get("player1_score"), s.get("player2_score"),
                s.get("player1_seed"), s.get("player2_seed"), s.get("state"),
            ),
        )

    def upsert_game(self, g: dict):
        self.conn.execute(
            """
            INSERT INTO games (id, set_id, game_number, winner_entrant_id,
                               stage_id, stage_name,
                               player1_character_id, player1_character_name,
                               player2_character_id, player2_character_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                game_number=excluded.game_number, winner_entrant_id=excluded.winner_entrant_id,
                stage_id=excluded.stage_id, stage_name=excluded.stage_name,
                player1_character_id=excluded.player1_character_id,
                player1_character_name=excluded.player1_character_name,
                player2_character_id=excluded.player2_character_id,
                player2_character_name=excluded.player2_character_name
            """,
            (
                g["id"], g["set_id"], g["game_number"], g.get("winner_entrant_id"),
                g.get("stage_id"), g.get("stage_name"),
                g.get("player1_character_id"), g.get("player1_character_name"),
                g.get("player2_character_id"), g.get("player2_character_name"),
            ),
        )

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.commit()
        self.conn.close()

    def tournament_exists(self, tournament_id: int) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM tournaments WHERE id = ?", (tournament_id,)
        ).fetchone()
        return row is not None

    def event_has_sets(self, event_id: int) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM sets WHERE event_id = ? LIMIT 1", (event_id,)
        ).fetchone()
        return row is not None

    def get_stats(self) -> dict:
        """Return counts of all stored data."""
        stats = {}
        for table in ["tournaments", "events", "players", "sets", "games"]:
            row = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            stats[table] = row[0]
        return stats
