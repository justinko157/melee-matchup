"""Gold table identifiers and their Arrow schemas."""

import pyarrow as pa

GOLD_NAMESPACE = "gold"
SET_FEATURES = f"{GOLD_NAMESPACE}.set_features"
PLAYER_RATINGS = f"{GOLD_NAMESPACE}.player_ratings"
HEAD_TO_HEAD = f"{GOLD_NAMESPACE}.head_to_head"
SILVER_SNAPSHOT_PROPERTY = "melee.silver_sets_snapshot"

TS = pa.timestamp("us", tz="UTC")
INT, FLOAT = pa.int64(), pa.float64()

SET_FEATURES_SCHEMA = pa.schema(
    [
        ("set_id", INT),
        ("event_id", INT),
        ("tournament_id", INT),
        ("tournament_name", pa.string()),
        ("completed_at", TS),
        ("player1_id", INT),
        ("player2_id", INT),
        ("p1_elo", FLOAT),
        ("p2_elo", FLOAT),
        ("elo_diff", FLOAT),
        ("p1_expected", FLOAT),
        ("p1_sets_played", INT),
        ("p2_sets_played", INT),
        ("p1_recent_wr", FLOAT),
        ("p2_recent_wr", FLOAT),
        ("recent_wr_diff", FLOAT),
        ("p1_h2h_wins", INT),
        ("p2_h2h_wins", INT),
        ("p1_h2h_wr", FLOAT),
        ("h2h_total", INT),
        ("p1_seed", INT),
        ("p2_seed", INT),
        ("seed_diff", INT),
        ("num_attendees", INT),
        ("p1_score", FLOAT),
        ("p2_score", FLOAT),
        ("p1_won", INT),
    ]
)
PLAYER_RATINGS_SCHEMA = pa.schema(
    [
        ("player_id", INT),
        ("gamer_tag", pa.string()),
        ("elo", FLOAT),
        ("sets_played", INT),
        ("recent_wr", FLOAT),
        ("wins", INT),
        ("losses", INT),
        ("last_played_at", TS),
    ]
)
HEAD_TO_HEAD_SCHEMA = pa.schema(
    [
        ("player_a_id", INT),
        ("player_b_id", INT),
        ("a_wins", INT),
        ("b_wins", INT),
        ("total", INT),
        ("last_played_at", TS),
    ]
)
# identifier -> (schema, column partitioned by year or None)
GOLD_TABLES: dict[str, tuple[pa.Schema, str | None]] = {
    SET_FEATURES: (SET_FEATURES_SCHEMA, "completed_at"),
    PLAYER_RATINGS: (PLAYER_RATINGS_SCHEMA, None),
    HEAD_TO_HEAD: (HEAD_TO_HEAD_SCHEMA, None),
}
