"""Pandera schema for gold.set_features (moved from src/validation.py)."""

import pandera.pandas as pandera

features_schema = pandera.DataFrameSchema(
    columns={
        "set_id": pandera.Column("int64", nullable=False, unique=True, coerce=True),
        "event_id": pandera.Column("int64", nullable=False, coerce=True),
        "tournament_id": pandera.Column("int64", nullable=False, coerce=True),
        "tournament_name": pandera.Column(str, nullable=False, coerce=True),
        "completed_at": pandera.Column("datetime64[us, UTC]", nullable=False, coerce=True),
        "player1_id": pandera.Column("int64", nullable=False, coerce=True),
        "player2_id": pandera.Column("int64", nullable=False, coerce=True),
        "p1_elo": pandera.Column(float, pandera.Check.gt(0), nullable=False),
        "p2_elo": pandera.Column(float, pandera.Check.gt(0), nullable=False),
        "elo_diff": pandera.Column(float, nullable=False),
        "p1_expected": pandera.Column(float, pandera.Check.in_range(0, 1), nullable=False),
        "p1_sets_played": pandera.Column("int64", pandera.Check.ge(0), coerce=True),
        "p2_sets_played": pandera.Column("int64", pandera.Check.ge(0), coerce=True),
        "p1_recent_wr": pandera.Column(float, pandera.Check.in_range(0, 1), nullable=False),
        "p2_recent_wr": pandera.Column(float, pandera.Check.in_range(0, 1), nullable=False),
        "recent_wr_diff": pandera.Column(float, pandera.Check.in_range(-1, 1), nullable=False),
        "p1_h2h_wins": pandera.Column("int64", pandera.Check.ge(0), coerce=True),
        "p2_h2h_wins": pandera.Column("int64", pandera.Check.ge(0), coerce=True),
        "p1_h2h_wr": pandera.Column(float, pandera.Check.in_range(0, 1), nullable=False),
        "h2h_total": pandera.Column("int64", pandera.Check.ge(0), coerce=True),
        "p1_seed": pandera.Column("Int64", nullable=True, coerce=True),
        "p2_seed": pandera.Column("Int64", nullable=True, coerce=True),
        "seed_diff": pandera.Column("int64", nullable=False, coerce=True),
        "num_attendees": pandera.Column("int64", pandera.Check.ge(0), coerce=True),
        "p1_score": pandera.Column(float, nullable=True),
        "p2_score": pandera.Column(float, nullable=True),
        "p1_won": pandera.Column("int64", pandera.Check.isin([0, 1]), coerce=True),
    },
    checks=[
        pandera.Check(
            lambda df: (df["player1_id"] != df["player2_id"]).all(),
            error="player1_id and player2_id must differ",
        ),
        pandera.Check(
            lambda df: (df["h2h_total"] == df["p1_h2h_wins"] + df["p2_h2h_wins"]).all(),
            error="h2h_total must equal p1_h2h_wins + p2_h2h_wins",
        ),
        pandera.Check(
            lambda df: (df["seed_diff"] == df["p1_seed"].fillna(0) - df["p2_seed"].fillna(0)).all(),
            error="seed_diff must equal p1_seed - p2_seed, with a missing seed as 0",
        ),
    ],
    ordered=True,
)
