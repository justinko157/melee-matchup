"""Chronological feature pass over silver sets (moved from src/features.py).

Every feature for a set uses only sets completed before it. Sets are processed
in (completed_at, set_id) order so ties resolve the same way on every run.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

DEFAULT_ELO = 1500
K_FACTOR = 32
# Scale factor: 400 means a 400-point gap = 10x expected performance
ELO_SCALE = 400
RECENT_WINDOW = 30  # last N sets for rolling win rate

SET_INPUT_COLUMNS = (
    "set_id",
    "event_id",
    "completed_at",
    "player1_id",
    "player2_id",
    "winner_player_id",
    "player1_score",
    "player2_score",
    "player1_seed",
    "player2_seed",
    "tournament_id",
    "tournament_name",
    "num_attendees",
)
FEATURE_COLUMNS = (
    "set_id",
    "event_id",
    "tournament_id",
    "tournament_name",
    "completed_at",
    "player1_id",
    "player2_id",
    "p1_elo",
    "p2_elo",
    "elo_diff",
    "p1_expected",
    "p1_sets_played",
    "p2_sets_played",
    "p1_recent_wr",
    "p2_recent_wr",
    "recent_wr_diff",
    "p1_h2h_wins",
    "p2_h2h_wins",
    "p1_h2h_wr",
    "h2h_total",
    "p1_seed",
    "p2_seed",
    "seed_diff",
    "num_attendees",
    "p1_score",
    "p2_score",
    "p1_won",
)


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


@dataclass
class FeatureState:
    """Everything the pass knows after the last set; gold's player and h2h tables use it."""

    elo: EloEngine = field(default_factory=EloEngine)
    recent_wins: dict[int, list[bool]] = field(default_factory=lambda: defaultdict(list))
    # (lower id, higher id) -> [lower id's wins, higher id's wins]
    h2h: dict[tuple[int, int], list[int]] = field(
        default_factory=lambda: defaultdict(lambda: [0, 0])
    )
    wins: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    losses: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    last_played: dict[int, pd.Timestamp] = field(default_factory=dict)
    pair_last_played: dict[tuple[int, int], pd.Timestamp] = field(default_factory=dict)


def _int_or_none(value) -> int | None:
    return None if value is None or pd.isna(value) else int(value)


def _float_or_none(value) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def eligible_sets(sets: pd.DataFrame) -> pd.DataFrame:
    """Sets with a winner and a completion time, in (completed_at, set_id) order."""
    mask = sets["winner_player_id"].notna() & sets["completed_at"].notna()
    ordered = sets[mask].sort_values(["completed_at", "set_id"], kind="mergesort")
    return ordered.reset_index(drop=True)


def build_features(sets: pd.DataFrame) -> tuple[pd.DataFrame, FeatureState]:
    """Build the feature matrix (FEATURE_COLUMNS) and return the final state."""
    state = FeatureState()
    rows = []
    for row in eligible_sets(sets).itertuples(index=False):
        p1, p2 = int(row.player1_id), int(row.player2_id)
        winner = int(row.winner_player_id)

        # --- FEATURES (computed BEFORE this set) ---
        p1_elo = state.elo.get_rating(p1)
        p2_elo = state.elo.get_rating(p2)
        p1_games = state.elo.games_played[p1]
        p2_games = state.elo.games_played[p2]

        p1_recent = state.recent_wins[p1][-RECENT_WINDOW:]
        p2_recent = state.recent_wins[p2][-RECENT_WINDOW:]
        p1_recent_wr = sum(p1_recent) / len(p1_recent) if p1_recent else 0.5
        p2_recent_wr = sum(p2_recent) / len(p2_recent) if p2_recent else 0.5

        pair = (min(p1, p2), max(p1, p2))
        record = state.h2h[pair]
        if p1 == pair[0]:
            p1_h2h_wins, p2_h2h_wins = record[0], record[1]
        else:
            p1_h2h_wins, p2_h2h_wins = record[1], record[0]
        h2h_total = p1_h2h_wins + p2_h2h_wins

        p1_seed, p2_seed = _int_or_none(row.player1_seed), _int_or_none(row.player2_seed)
        p1_won = int(winner == p1)

        rows.append(
            {
                "set_id": int(row.set_id),
                "event_id": int(row.event_id),
                "tournament_id": int(row.tournament_id),
                "tournament_name": row.tournament_name,
                "completed_at": row.completed_at,
                "player1_id": p1,
                "player2_id": p2,
                "p1_elo": p1_elo,
                "p2_elo": p2_elo,
                "elo_diff": p1_elo - p2_elo,
                "p1_expected": state.elo.expected_score(p1_elo, p2_elo),
                "p1_sets_played": p1_games,
                "p2_sets_played": p2_games,
                "p1_recent_wr": p1_recent_wr,
                "p2_recent_wr": p2_recent_wr,
                "recent_wr_diff": p1_recent_wr - p2_recent_wr,
                "p1_h2h_wins": p1_h2h_wins,
                "p2_h2h_wins": p2_h2h_wins,
                "p1_h2h_wr": p1_h2h_wins / h2h_total if h2h_total > 0 else 0.5,
                "h2h_total": h2h_total,
                "p1_seed": p1_seed,
                "p2_seed": p2_seed,
                # A missing seed counts as 0, as in the original pipeline.
                "seed_diff": (p1_seed or 0) - (p2_seed or 0),
                "num_attendees": _int_or_none(row.num_attendees) or 0,
                "p1_score": _float_or_none(row.player1_score),
                "p2_score": _float_or_none(row.player2_score),
                "p1_won": p1_won,
            }
        )

        # --- UPDATE STATE (after this set) ---
        loser = p2 if winner == p1 else p1
        state.elo.update(winner, loser)
        state.recent_wins[p1].append(p1_won == 1)
        state.recent_wins[p2].append(p1_won == 0)
        state.h2h[pair][0 if winner == pair[0] else 1] += 1
        state.wins[winner] += 1
        state.losses[loser] += 1
        state.last_played[p1] = state.last_played[p2] = row.completed_at
        state.pair_last_played[pair] = row.completed_at

    return pd.DataFrame(rows, columns=list(FEATURE_COLUMNS)), state
