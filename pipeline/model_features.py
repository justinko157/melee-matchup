"""Model feature names shared by gold, training and the app. No third-party imports."""

SEED_FEATURES = ["seed_diff"]
CORE_FEATURES = [
    "elo_diff",
    "p1_expected",
    "p1_sets_played",
    "p2_sets_played",
    "recent_wr_diff",
    "p1_h2h_wr",
    "h2h_total",
    "seed_diff",
    "num_attendees",
]
TARGET = "p1_won"
