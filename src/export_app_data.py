"""Export pre-computed data for the Streamlit app.

Generates:
- data/app/model.json — XGBoost model
- data/app/players.parquet — latest Elo, form, and stats per player
- data/app/h2h.parquet — head-to-head records for all player pairs
"""

import logging
import sqlite3
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .database import DEFAULT_DB_PATH
from .model import (
    CORE_FEATURES,
    export_shap_metadata,
    load_data,
    temporal_split,
    train_xgboost,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

APP_DATA_DIR = Path(__file__).parent.parent / "data" / "app"

FEATURES_PATH = Path(__file__).parent.parent / "data" / "processed" / "features.parquet"

P_COLS = ["player_id", "elo", "sets_played", "recent_wr"]
RENAME_P1 = {
    "player1_id": "player_id",
    "p1_elo": "elo",
    "p1_sets_played": "sets_played",
    "p1_recent_wr": "recent_wr",
}
RENAME_P2 = {
    "player2_id": "player_id",
    "p2_elo": "elo",
    "p2_sets_played": "sets_played",
    "p2_recent_wr": "recent_wr",
}


def export_model():
    """Train XGBoost and export model + SHAP metadata."""
    df = load_data()
    train, test = temporal_split(df)
    result = train_xgboost(train, test)

    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    model_path = APP_DATA_DIR / "model.json"
    result.model.save_model(str(model_path))
    logger.info(f"Exported model to {model_path}")

    # Export SHAP metadata for per-prediction explanations
    X_test = test[CORE_FEATURES]
    shap_path = APP_DATA_DIR / "shap_metadata.json"
    export_shap_metadata(result.model, X_test, shap_path)


def export_player_snapshots():
    """Export latest stats for every player."""
    features = pd.read_parquet(FEATURES_PATH)

    # Get latest stats per player from both sides of each set
    src_cols_p1 = list(RENAME_P1.keys()) + ["completed_at"]
    src_cols_p2 = list(RENAME_P2.keys()) + ["completed_at"]
    p1 = features[src_cols_p1].rename(columns=RENAME_P1)
    p2 = features[src_cols_p2].rename(columns=RENAME_P2)

    all_player_rows = pd.concat([p1, p2])
    latest = all_player_rows.sort_values("completed_at").groupby("player_id").last().reset_index()

    # Add gamer tags
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    names = pd.read_sql("SELECT id as player_id, gamer_tag FROM players", conn)
    conn.close()
    latest = latest.merge(names, on="player_id", how="left")

    # Compute overall win rate from the DB
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    wins = pd.read_sql(
        """
        SELECT winner_player_id as player_id, COUNT(*) as wins
        FROM sets WHERE winner_player_id IS NOT NULL
        GROUP BY winner_player_id
    """,
        conn,
    )
    total_as_p1 = pd.read_sql(
        "SELECT player1_id as player_id, COUNT(*) as c " "FROM sets GROUP BY player1_id",
        conn,
    )
    total_as_p2 = pd.read_sql(
        "SELECT player2_id as player_id, COUNT(*) as c " "FROM sets GROUP BY player2_id",
        conn,
    )
    conn.close()

    totals = (
        total_as_p1.set_index("player_id")["c"]
        .add(total_as_p2.set_index("player_id")["c"], fill_value=0)
        .reset_index()
    )
    totals.columns = ["player_id", "total_sets"]
    wins.columns = ["player_id", "wins"]

    latest = latest.merge(totals, on="player_id", how="left")
    latest = latest.merge(wins, on="player_id", how="left")
    latest["wins"] = latest["wins"].fillna(0).astype(int)
    latest["total_sets"] = latest["total_sets"].fillna(0).astype(int)
    latest["losses"] = latest["total_sets"] - latest["wins"]
    latest["win_rate"] = (latest["wins"] / latest["total_sets"]).fillna(0)

    out_path = APP_DATA_DIR / "players.parquet"
    latest.to_parquet(out_path, index=False)
    logger.info(f"Exported {len(latest):,} player snapshots to {out_path}")


def export_h2h():
    """Export head-to-head records for all player pairs with 2+ sets."""
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    sets_df = pd.read_sql(
        """
        SELECT player1_id, player2_id, winner_player_id
        FROM sets WHERE winner_player_id IS NOT NULL
    """,
        conn,
    )
    conn.close()

    h2h: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
    for _, row in sets_df.iterrows():
        p1, p2, w = row["player1_id"], row["player2_id"], row["winner_player_id"]
        pair = (min(p1, p2), max(p1, p2))
        if w == pair[0]:
            h2h[pair][0] += 1
        else:
            h2h[pair][1] += 1

    rows = []
    for (pa, pb), (wa, wb) in h2h.items():
        if wa + wb >= 2:
            rows.append({"player_a": pa, "player_b": pb, "a_wins": wa, "b_wins": wb})

    h2h_df = pd.DataFrame(rows)
    out_path = APP_DATA_DIR / "h2h.parquet"
    h2h_df.to_parquet(out_path, index=False)
    logger.info(f"Exported {len(h2h_df):,} H2H records to {out_path}")


if __name__ == "__main__":
    export_model()
    export_player_snapshots()
    export_h2h()
