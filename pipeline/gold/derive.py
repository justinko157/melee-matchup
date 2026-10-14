"""Player and head-to-head tables from the feature pass's final state.

Both build columns directly instead of a list of row dicts: head_to_head has
roughly one row per set, so per-row dicts would cost hundreds of MB.
"""

import numpy as np
import pandas as pd

from pipeline.gold.features import FeatureState
from pipeline.gold.tables import HEAD_TO_HEAD_SCHEMA, PLAYER_RATINGS_SCHEMA


def _tag(tag, player_id: int) -> str:
    return tag if isinstance(tag, str) and tag else str(player_id)


def _timestamps(values: list) -> pd.Series:
    return pd.to_datetime(pd.Series(values, dtype=object), utc=True).astype("datetime64[us, UTC]")


def player_ratings(state: FeatureState, players: pd.DataFrame) -> pd.DataFrame:
    tags = dict(zip(players["player_id"], players["gamer_tag"], strict=True))
    player_ids = sorted(state.elo.ratings)
    recent_wr = np.empty(len(player_ids))
    for i, player_id in enumerate(player_ids):
        recent = state.recent_wins[player_id]  # already capped at RECENT_WINDOW
        recent_wr[i] = sum(recent) / len(recent) if recent else 0.5
    columns = {
        "player_id": np.array(player_ids, dtype="int64"),
        "gamer_tag": [_tag(tags.get(p), p) for p in player_ids],
        "elo": np.array([state.elo.get_rating(p) for p in player_ids], dtype="float64"),
        "sets_played": np.array([state.elo.games_played[p] for p in player_ids], dtype="int64"),
        "recent_wr": recent_wr,
        "wins": np.array([state.wins[p] for p in player_ids], dtype="int64"),
        "losses": np.array([state.losses[p] for p in player_ids], dtype="int64"),
        "last_played_at": _timestamps([state.last_played[p] for p in player_ids]),
    }
    return pd.DataFrame(columns, columns=PLAYER_RATINGS_SCHEMA.names)


def head_to_head(state: FeatureState) -> pd.DataFrame:
    n = len(state.h2h)
    a_ids, b_ids = np.empty(n, "int64"), np.empty(n, "int64")
    a_wins, b_wins = np.empty(n, "int64"), np.empty(n, "int64")
    last_played = [None] * n
    for i, ((a, b), record) in enumerate(state.h2h.items()):
        a_ids[i], b_ids[i] = a, b
        a_wins[i], b_wins[i] = record
        last_played[i] = state.pair_last_played[(a, b)]
    order = np.lexsort((b_ids, a_ids))  # sorted by (player_a_id, player_b_id)
    columns = {
        "player_a_id": a_ids[order],
        "player_b_id": b_ids[order],
        "a_wins": a_wins[order],
        "b_wins": b_wins[order],
        "total": (a_wins + b_wins)[order],
        "last_played_at": _timestamps(last_played).take(order).reset_index(drop=True),
    }
    return pd.DataFrame(columns, columns=HEAD_TO_HEAD_SCHEMA.names)
