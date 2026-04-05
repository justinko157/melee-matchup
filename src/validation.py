"""Pandera schema validation for the features DataFrame.

Catches data quality issues (nulls, out-of-range values, type mismatches)
before they silently corrupt model training.

Usage:
    python -m src.validation                    # validate default features file
    python -m src.validation path/to/file.parquet
"""

import logging
import sys
from pathlib import Path

import pandera.pandas as pa

logger = logging.getLogger(__name__)

FEATURES_PATH = Path(__file__).parent.parent / "data" / "processed" / "features.parquet"

features_schema = pa.DataFrameSchema(
    columns={
        "set_id": pa.Column(int, nullable=False, unique=True),
        "event_id": pa.Column(int, nullable=False),
        "tournament_id": pa.Column(int, nullable=False),
        "tournament_name": pa.Column(str, nullable=False),
        "completed_at": pa.Column(int, pa.Check.gt(0), nullable=False),
        "player1_id": pa.Column(int, nullable=False),
        "player2_id": pa.Column(int, nullable=False),
        "p1_elo": pa.Column(float, pa.Check.gt(0), nullable=False),
        "p2_elo": pa.Column(float, pa.Check.gt(0), nullable=False),
        "elo_diff": pa.Column(float, nullable=False),
        "p1_expected": pa.Column(
            float,
            pa.Check.in_range(0, 1),
            nullable=False,
        ),
        "p1_sets_played": pa.Column(int, pa.Check.ge(0), nullable=False),
        "p2_sets_played": pa.Column(int, pa.Check.ge(0), nullable=False),
        "p1_recent_wr": pa.Column(
            float,
            pa.Check.in_range(0, 1),
            nullable=False,
        ),
        "p2_recent_wr": pa.Column(
            float,
            pa.Check.in_range(0, 1),
            nullable=False,
        ),
        "recent_wr_diff": pa.Column(
            float,
            pa.Check.in_range(-1, 1),
            nullable=False,
        ),
        "p1_h2h_wins": pa.Column(int, pa.Check.ge(0), nullable=False),
        "p2_h2h_wins": pa.Column(int, pa.Check.ge(0), nullable=False),
        "p1_h2h_wr": pa.Column(
            float,
            pa.Check.in_range(0, 1),
            nullable=False,
        ),
        "h2h_total": pa.Column(int, pa.Check.ge(0), nullable=False),
        "p1_seed": pa.Column(int, nullable=False),
        "p2_seed": pa.Column(int, nullable=False),
        "seed_diff": pa.Column(int, nullable=False),
        "num_attendees": pa.Column(int, pa.Check.gt(0), nullable=False),
        "p1_score": pa.Column(float, nullable=True),
        "p2_score": pa.Column(float, nullable=True),
        "p1_won": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
    },
    checks=[
        # Players in a set must be different
        pa.Check(
            lambda df: (df["player1_id"] != df["player2_id"]).all(),
            error="player1_id and player2_id must differ",
        ),
        # H2H totals must equal sum of wins
        pa.Check(
            lambda df: (df["h2h_total"] == df["p1_h2h_wins"] + df["p2_h2h_wins"]).all(),
            error="h2h_total must equal p1_h2h_wins + p2_h2h_wins",
        ),
        # Seed diff must be consistent
        pa.Check(
            lambda df: (df["seed_diff"] == df["p1_seed"] - df["p2_seed"]).all(),
            error="seed_diff must equal p1_seed - p2_seed",
        ),
    ],
    ordered=True,
)


def validate(path: Path | str = FEATURES_PATH) -> bool:
    """Validate the features parquet file against the schema.

    Returns True if valid, raises SchemaError otherwise.
    """
    import pandas as pd

    df = pd.read_parquet(path)
    logger.info(f"Validating {len(df):,} rows from {path}")
    features_schema.validate(df)
    logger.info("Validation passed")
    return True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    path = sys.argv[1] if len(sys.argv) > 1 else FEATURES_PATH
    try:
        validate(path)
    except pa.errors.SchemaError as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)
