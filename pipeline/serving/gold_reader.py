"""Gold tables the app reads, through PyIceberg."""

import pandas as pd
from pyiceberg.catalog import Catalog

from pipeline.gold.tables import HEAD_TO_HEAD, PLAYER_RATINGS
from pipeline.lakehouse import read_table

PLAYER_COLUMNS = (
    "player_id",
    "gamer_tag",
    "elo",
    "sets_played",
    "recent_wr",
    "wins",
    "losses",
    "last_played_at",
)
H2H_COLUMNS = ("player_a_id", "player_b_id", "a_wins", "b_wins", "total", "last_played_at")


def load_player_ratings(catalog: Catalog, min_sets: int = 5) -> pd.DataFrame:
    """Players with at least min_sets sets, strongest first, with career totals."""
    players = read_table(catalog, PLAYER_RATINGS, PLAYER_COLUMNS)
    players = players[players["sets_played"] >= min_sets].copy()
    players["total_sets"] = players["wins"] + players["losses"]
    players["win_rate"] = (players["wins"] / players["total_sets"]).where(
        players["total_sets"] > 0, 0.0
    )
    return players.sort_values("elo", ascending=False).reset_index(drop=True)


def load_head_to_head(catalog: Catalog) -> pd.DataFrame:
    return read_table(catalog, HEAD_TO_HEAD, H2H_COLUMNS)


def lookup_h2h(h2h: pd.DataFrame, pid1: int, pid2: int) -> tuple[int, int]:
    """(pid1's wins, pid2's wins) against each other."""
    a, b = min(pid1, pid2), max(pid1, pid2)
    row = h2h[(h2h["player_a_id"] == a) & (h2h["player_b_id"] == b)]
    if row.empty:
        return 0, 0
    record = row.iloc[0]
    a_wins, b_wins = int(record["a_wins"]), int(record["b_wins"])
    return (a_wins, b_wins) if pid1 == a else (b_wins, a_wins)
