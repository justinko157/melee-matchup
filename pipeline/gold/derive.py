"""Player and head-to-head tables from the feature pass's final state."""

import pandas as pd

from pipeline.gold.features import RECENT_WINDOW, FeatureState
from pipeline.gold.tables import HEAD_TO_HEAD_SCHEMA, PLAYER_RATINGS_SCHEMA


def _tag(tag, player_id: int) -> str:
    return tag if isinstance(tag, str) and tag else str(player_id)


def player_ratings(state: FeatureState, players: pd.DataFrame) -> pd.DataFrame:
    tags = dict(zip(players["player_id"], players["gamer_tag"], strict=True))
    rows = []
    for player_id in sorted(state.elo.ratings):
        recent = state.recent_wins[player_id][-RECENT_WINDOW:]
        rows.append(
            {
                "player_id": player_id,
                "gamer_tag": _tag(tags.get(player_id), player_id),
                "elo": state.elo.get_rating(player_id),
                "sets_played": state.elo.games_played[player_id],
                "recent_wr": sum(recent) / len(recent) if recent else 0.5,
                "wins": state.wins[player_id],
                "losses": state.losses[player_id],
                "last_played_at": state.last_played[player_id],
            }
        )
    return pd.DataFrame(rows, columns=PLAYER_RATINGS_SCHEMA.names)


def head_to_head(state: FeatureState) -> pd.DataFrame:
    rows = [
        {
            "player_a_id": a,
            "player_b_id": b,
            "a_wins": record[0],
            "b_wins": record[1],
            "total": record[0] + record[1],
            "last_played_at": state.pair_last_played[(a, b)],
        }
        for (a, b), record in sorted(state.h2h.items())
    ]
    return pd.DataFrame(rows, columns=HEAD_TO_HEAD_SCHEMA.names)
