"""Synthetic gold.set_features rows where Elo predicts the winner and seeds are noise."""

import numpy as np
import pandas as pd

from pipeline.gold.features import FEATURE_COLUMNS


def synthetic_gold_features(n: int = 800, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    elo_diff = rng.normal(0, 250, n)
    expected = 1 / (1 + 10 ** (-elo_diff / 400))
    p1_won = (rng.random(n) < expected).astype("int64")
    p1_seed = rng.integers(1, 64, n)
    p2_seed = rng.integers(1, 64, n)
    frame = pd.DataFrame(
        {
            "set_id": np.arange(n, dtype="int64"),
            "event_id": 10,
            "tournament_id": 1,
            "tournament_name": "Synthetic",
            "completed_at": pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC"),
            "player1_id": rng.integers(1, 50, n),
            "player2_id": rng.integers(51, 100, n),
            "p1_elo": 1500 + elo_diff / 2,
            "p2_elo": 1500 - elo_diff / 2,
            "elo_diff": elo_diff,
            "p1_expected": expected,
            "p1_sets_played": rng.integers(5, 200, n),
            "p2_sets_played": rng.integers(5, 200, n),
            "p1_recent_wr": 0.5,
            "p2_recent_wr": 0.5,
            "recent_wr_diff": 0.0,
            "p1_h2h_wins": 0,
            "p2_h2h_wins": 0,
            "p1_h2h_wr": 0.5,
            "h2h_total": 0,
            "p1_seed": p1_seed,
            "p2_seed": p2_seed,
            "seed_diff": p1_seed - p2_seed,
            "num_attendees": 200,
            "p1_score": 2.0,
            "p2_score": 1.0,
            "p1_won": p1_won,
        }
    )
    return frame[list(FEATURE_COLUMNS)]
