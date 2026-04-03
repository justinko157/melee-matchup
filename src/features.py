"""Feature engineering pipeline for Melee set prediction.

Processes sets chronologically and computes per-set features:
- Elo ratings (custom implementation)
- Head-to-head records
- Recent form (rolling win rate)
- Seed differential
- Tournament size/tier

All features use ONLY data available BEFORE each set to avoid leakage.
"""

import logging
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .database import DEFAULT_DB_PATH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Elo rating engine
# ---------------------------------------------------------------------------

DEFAULT_ELO = 1500
K_FACTOR = 32
# Scale factor: 400 means a 400-point gap = 10x expected performance
ELO_SCALE = 400


class EloEngine:
    """Tracks Elo ratings for all players over time."""

    def __init__(self, default_rating: float = DEFAULT_ELO, k: float = K_FACTOR):
        self.default_rating = default_rating
        self.k = k
        self.ratings: dict[int, float] = {}
        self.games_played: dict[int, int] = defaultdict(int)

    def get_rating(self, player_id: int) -> float:
        return self.ratings.get(player_id, self.default_rating)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Probability that player A beats player B."""
        return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / ELO_SCALE))

    def update(self, winner_id: int, loser_id: int) -> tuple[float, float]:
        """Update ratings after a match. Returns (new_winner_rating, new_loser_rating).

        Uses a dynamic K-factor: higher for players with fewer games
        to allow faster convergence for newcomers.
        """
        r_w = self.get_rating(winner_id)
        r_l = self.get_rating(loser_id)

        expected_w = self.expected_score(r_w, r_l)

        # Dynamic K: higher for newer players (first 30 games)
        k_w = self.k * max(1.0, 2.0 - self.games_played[winner_id] / 30)
        k_l = self.k * max(1.0, 2.0 - self.games_played[loser_id] / 30)

        self.ratings[winner_id] = r_w + k_w * (1.0 - expected_w)
        self.ratings[loser_id] = r_l + k_l * (0.0 - (1.0 - expected_w))

        self.games_played[winner_id] += 1
        self.games_played[loser_id] += 1

        return self.ratings[winner_id], self.ratings[loser_id]


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------


class FeatureBuilder:
    """Builds per-set features from the database, processing chronologically."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def load_sets(self) -> pd.DataFrame:
        """Load all completed sets with tournament metadata, ordered by time."""
        query = """
            SELECT
                s.id as set_id,
                s.event_id,
                s.completed_at,
                s.player1_id,
                s.player2_id,
                s.winner_player_id,
                s.player1_score,
                s.player2_score,
                s.player1_seed,
                s.player2_seed,
                s.full_round_text,
                s.total_games,
                t.id as tournament_id,
                t.name as tournament_name,
                t.num_attendees,
                t.start_at as tournament_date
            FROM sets s
            JOIN events e ON s.event_id = e.id
            JOIN tournaments t ON e.tournament_id = t.id
            WHERE s.winner_player_id IS NOT NULL
              AND s.completed_at IS NOT NULL
            ORDER BY s.completed_at ASC
        """
        df = pd.read_sql(query, self.conn)
        logger.info(f"Loaded {len(df):,} sets")
        return df

    def build_features(self) -> pd.DataFrame:
        """Build the full feature matrix.

        Processes sets chronologically. For each set, features are computed
        from data available BEFORE that set (no leakage).
        """
        sets_df = self.load_sets()

        elo = EloEngine()
        # Rolling stats per player
        recent_wins: dict[int, list[bool]] = defaultdict(list)
        # Head-to-head records: (p1, p2) -> [p1_wins, p2_wins] where p1 < p2
        h2h: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])

        RECENT_WINDOW = 30  # last N sets for rolling win rate

        features = []

        for row in tqdm(sets_df.itertuples(), total=len(sets_df), desc="Building features"):
            p1 = row.player1_id
            p2 = row.player2_id
            winner = row.winner_player_id

            # --- FEATURES (computed BEFORE this set) ---

            # Elo ratings
            p1_elo = elo.get_rating(p1)
            p2_elo = elo.get_rating(p2)
            elo_diff = p1_elo - p2_elo
            p1_expected = elo.expected_score(p1_elo, p2_elo)

            # Games played (proxy for experience/reliability)
            p1_games = elo.games_played[p1]
            p2_games = elo.games_played[p2]

            # Recent form (rolling win rate over last N sets)
            p1_recent = recent_wins[p1][-RECENT_WINDOW:]
            p2_recent = recent_wins[p2][-RECENT_WINDOW:]
            p1_recent_wr = sum(p1_recent) / len(p1_recent) if p1_recent else 0.5
            p2_recent_wr = sum(p2_recent) / len(p2_recent) if p2_recent else 0.5

            # Head-to-head
            pair = (min(p1, p2), max(p1, p2))
            h2h_record = h2h[pair]
            if p1 == pair[0]:
                p1_h2h_wins, p2_h2h_wins = h2h_record[0], h2h_record[1]
            else:
                p1_h2h_wins, p2_h2h_wins = h2h_record[1], h2h_record[0]
            h2h_total = p1_h2h_wins + p2_h2h_wins
            p1_h2h_wr = p1_h2h_wins / h2h_total if h2h_total > 0 else 0.5

            # Seed differential
            seed_diff = (row.player1_seed or 0) - (row.player2_seed or 0)

            # Tournament features
            num_attendees = row.num_attendees or 0

            # --- TARGET ---
            p1_won = int(winner == p1)

            features.append({
                "set_id": row.set_id,
                "event_id": row.event_id,
                "tournament_id": row.tournament_id,
                "tournament_name": row.tournament_name,
                "completed_at": row.completed_at,
                "player1_id": p1,
                "player2_id": p2,
                # Elo features
                "p1_elo": p1_elo,
                "p2_elo": p2_elo,
                "elo_diff": elo_diff,
                "p1_expected": p1_expected,
                # Experience
                "p1_sets_played": p1_games,
                "p2_sets_played": p2_games,
                # Recent form
                "p1_recent_wr": p1_recent_wr,
                "p2_recent_wr": p2_recent_wr,
                "recent_wr_diff": p1_recent_wr - p2_recent_wr,
                # Head-to-head
                "p1_h2h_wins": p1_h2h_wins,
                "p2_h2h_wins": p2_h2h_wins,
                "p1_h2h_wr": p1_h2h_wr,
                "h2h_total": h2h_total,
                # Seed
                "p1_seed": row.player1_seed,
                "p2_seed": row.player2_seed,
                "seed_diff": seed_diff,
                # Tournament context
                "num_attendees": num_attendees,
                # Scores
                "p1_score": row.player1_score,
                "p2_score": row.player2_score,
                # Target
                "p1_won": p1_won,
            })

            # --- UPDATE STATE (after this set) ---

            # Update Elo
            loser = p2 if winner == p1 else p1
            elo.update(winner, loser)

            # Update recent form
            recent_wins[p1].append(p1_won == 1)
            recent_wins[p2].append(p1_won == 0)

            # Update head-to-head
            if winner == pair[0]:
                h2h[pair][0] += 1
            else:
                h2h[pair][1] += 1

        result = pd.DataFrame(features)
        logger.info(f"Built {len(result):,} feature rows")
        return result

    def close(self):
        self.conn.close()


def build_and_export(
    db_path: Path | str = DEFAULT_DB_PATH,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Build features and export to parquet.

    Args:
        db_path: Path to the SQLite database.
        output_path: Where to save the parquet file. Defaults to data/processed/features.parquet.

    Returns:
        The feature DataFrame.
    """
    if output_path is None:
        output_path = Path(db_path).parent.parent / "processed" / "features.parquet"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    builder = FeatureBuilder(db_path)
    try:
        df = builder.build_features()
        df.to_parquet(output_path, index=False)
        logger.info(f"Exported {len(df):,} rows to {output_path}")
        return df
    finally:
        builder.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    build_and_export()
