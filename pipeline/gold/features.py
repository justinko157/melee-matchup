"""Chronological feature pass over silver sets (moved from src/features.py).

Every feature for a set uses only sets completed before it. Sets are processed
in (completed_at, set_id) order so ties resolve the same way on every run.
"""

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
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
    # Only the last RECENT_WINDOW results matter, so older ones are dropped.
    recent_wins: dict[int, deque[bool]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=RECENT_WINDOW))
    )
    # (lower id, higher id) -> [lower id's wins, higher id's wins]
    h2h: dict[tuple[int, int], list[int]] = field(
        default_factory=lambda: defaultdict(lambda: [0, 0])
    )
    wins: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    losses: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    last_played: dict[int, pd.Timestamp] = field(default_factory=dict)
    pair_last_played: dict[tuple[int, int], pd.Timestamp] = field(default_factory=dict)


def eligible_sets(sets: pd.DataFrame) -> pd.DataFrame:
    """Sets with a winner and a completion time, in (completed_at, set_id) order."""
    mask = sets["winner_player_id"].notna() & sets["completed_at"].notna()
    ordered = sets[mask].sort_values(["completed_at", "set_id"], kind="mergesort")
    return ordered.reset_index(drop=True)


_FLOAT_COLUMNS = (
    "p1_elo",
    "p2_elo",
    "elo_diff",
    "p1_expected",
    "p1_recent_wr",
    "p2_recent_wr",
    "recent_wr_diff",
    "p1_h2h_wr",
    "p1_score",
    "p2_score",
)
_INT_COLUMNS = (
    "set_id",
    "event_id",
    "tournament_id",
    "player1_id",
    "player2_id",
    "p1_sets_played",
    "p2_sets_played",
    "p1_h2h_wins",
    "p2_h2h_wins",
    "h2h_total",
    "seed_diff",
    "num_attendees",
    "p1_won",
)


def _typed(frame: pd.DataFrame) -> pd.DataFrame:
    """Give every column the dtype features_schema expects, also for all-default or empty input."""
    frame = frame.astype(
        {
            **dict.fromkeys(_FLOAT_COLUMNS, "float64"),
            **dict.fromkeys(_INT_COLUMNS, "int64"),
            "p1_seed": "Int64",
            "p2_seed": "Int64",
            "tournament_name": "object",
        }
    )
    frame["completed_at"] = pd.to_datetime(frame["completed_at"], utc=True).astype(
        "datetime64[us, UTC]"
    )
    return frame


def build_features(sets: pd.DataFrame) -> tuple[pd.DataFrame, FeatureState]:
    """Build the feature matrix (FEATURE_COLUMNS) and return the final state.

    Per-set values go into preallocated arrays rather than a list of row dicts,
    which kept ~2 KB of Python objects alive per set until the frame was built,
    and the frame wraps those arrays without copying them.
    """
    state = FeatureState()
    ordered = eligible_sets(sets)
    n = len(ordered)
    p1_elo, p2_elo, p1_expected = np.empty(n), np.empty(n), np.empty(n)
    p1_recent_wr, p2_recent_wr, p1_h2h_wr = np.empty(n), np.empty(n), np.empty(n)
    p1_sets_played, p2_sets_played = np.empty(n, "int64"), np.empty(n, "int64")
    p1_h2h_wins, p2_h2h_wins = np.empty(n, "int64"), np.empty(n, "int64")
    p1_won = np.empty(n, "int64")

    p1_ids = ordered["player1_id"].astype("int64").to_numpy()
    p2_ids = ordered["player2_id"].astype("int64").to_numpy()
    winners = ordered["winner_player_id"].astype("int64").to_numpy()
    for i, completed_at in enumerate(ordered["completed_at"]):
        p1, p2, winner = int(p1_ids[i]), int(p2_ids[i]), int(winners[i])

        # --- FEATURES (computed BEFORE this set) ---
        elo1, elo2 = state.elo.get_rating(p1), state.elo.get_rating(p2)
        p1_elo[i], p2_elo[i] = elo1, elo2
        p1_expected[i] = state.elo.expected_score(elo1, elo2)
        p1_sets_played[i] = state.elo.games_played[p1]
        p2_sets_played[i] = state.elo.games_played[p2]

        p1_recent, p2_recent = state.recent_wins[p1], state.recent_wins[p2]
        p1_recent_wr[i] = sum(p1_recent) / len(p1_recent) if p1_recent else 0.5
        p2_recent_wr[i] = sum(p2_recent) / len(p2_recent) if p2_recent else 0.5

        pair = (min(p1, p2), max(p1, p2))
        record = state.h2h[pair]
        if p1 == pair[0]:
            wins1, wins2 = record[0], record[1]
        else:
            wins1, wins2 = record[1], record[0]
        p1_h2h_wins[i], p2_h2h_wins[i] = wins1, wins2
        p1_h2h_wr[i] = wins1 / (wins1 + wins2) if wins1 + wins2 > 0 else 0.5

        won = winner == p1
        p1_won[i] = int(won)

        # --- UPDATE STATE (after this set) ---
        loser = p2 if won else p1
        state.elo.update(winner, loser)
        state.recent_wins[p1].append(won)
        state.recent_wins[p2].append(not won)
        state.h2h[pair][0 if winner == pair[0] else 1] += 1
        state.wins[winner] += 1
        state.losses[loser] += 1
        state.last_played[p1] = state.last_played[p2] = completed_at
        state.pair_last_played[pair] = completed_at

    p1_seed = ordered["player1_seed"].astype("Int64")
    p2_seed = ordered["player2_seed"].astype("Int64")
    frame = pd.DataFrame(
        {
            "set_id": ordered["set_id"].to_numpy(),
            "event_id": ordered["event_id"].to_numpy(),
            "tournament_id": ordered["tournament_id"].to_numpy(),
            "tournament_name": ordered["tournament_name"].to_numpy(),
            "completed_at": ordered["completed_at"],
            "player1_id": p1_ids,
            "player2_id": p2_ids,
            "p1_elo": p1_elo,
            "p2_elo": p2_elo,
            "elo_diff": p1_elo - p2_elo,
            "p1_expected": p1_expected,
            "p1_sets_played": p1_sets_played,
            "p2_sets_played": p2_sets_played,
            "p1_recent_wr": p1_recent_wr,
            "p2_recent_wr": p2_recent_wr,
            "recent_wr_diff": p1_recent_wr - p2_recent_wr,
            "p1_h2h_wins": p1_h2h_wins,
            "p2_h2h_wins": p2_h2h_wins,
            "p1_h2h_wr": p1_h2h_wr,
            "h2h_total": p1_h2h_wins + p2_h2h_wins,
            "p1_seed": p1_seed,
            "p2_seed": p2_seed,
            # A missing seed counts as 0, as in the original pipeline.
            "seed_diff": (p1_seed.fillna(0) - p2_seed.fillna(0)).to_numpy("int64"),
            "num_attendees": ordered["num_attendees"].fillna(0).to_numpy(),
            "p1_score": ordered["player1_score"].to_numpy(),
            "p2_score": ordered["player2_score"].to_numpy(),
            "p1_won": p1_won,
        },
        columns=list(FEATURE_COLUMNS),
        copy=False,  # wrap the arrays instead of copying them into new blocks
    )
    return _typed(frame), state
