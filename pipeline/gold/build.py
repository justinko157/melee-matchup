"""Recompute every gold table from silver. Run directly for a one-off build."""

import json
import logging
from dataclasses import dataclass

import pandas as pd
from pyiceberg.catalog import Catalog

from pipeline.gold.derive import head_to_head, player_ratings
from pipeline.gold.features import build_features
from pipeline.gold.tables import HEAD_TO_HEAD, PLAYER_RATINGS, SET_FEATURES
from pipeline.gold.writer import write_gold
from pipeline.lakehouse import current_snapshot_id, load_lakehouse_catalog, read_table
from pipeline.model_features import CORE_FEATURES

SET_COLUMNS = (
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
)


@dataclass
class GoldBuild:
    frames: dict[str, pd.DataFrame]
    eligible_silver_sets: int
    silver_sets_snapshot: int | None
    snapshots: dict[str, int | None]


def load_silver(catalog: Catalog) -> tuple[pd.DataFrame, pd.DataFrame, int | None, int]:
    """(sets joined to their tournament, players, silver.sets snapshot, eligible set count).

    Sets whose event or tournament is missing drop out of the join, as they did in
    the original SQL. The eligible count is taken before the join, so a gap shows
    up in the features_match_silver check.
    """
    snapshot = current_snapshot_id(catalog, "silver.sets")
    sets = read_table(catalog, "silver.sets", SET_COLUMNS, snapshot_id=snapshot)
    events = read_table(catalog, "silver.events", ("event_id", "tournament_id"))
    tournaments = read_table(
        catalog, "silver.tournaments", ("tournament_id", "name", "num_attendees")
    ).rename(columns={"name": "tournament_name"})
    players = read_table(catalog, "silver.players", ("player_id", "gamer_tag"))
    eligible = int((sets["winner_player_id"].notna() & sets["completed_at"].notna()).sum())
    joined = sets.merge(events, on="event_id", how="inner").merge(
        tournaments, on="tournament_id", how="inner"
    )
    return joined, players, snapshot, eligible


def build_gold(catalog: Catalog) -> GoldBuild:
    sets, players, snapshot, eligible = load_silver(catalog)
    features, state = build_features(sets)
    frames = {
        SET_FEATURES: features,
        PLAYER_RATINGS: player_ratings(state, players),
        HEAD_TO_HEAD: head_to_head(state),
    }
    snapshots = write_gold(catalog, frames, snapshot)
    return GoldBuild(frames, eligible, snapshot, snapshots)


def gold_checks(build: GoldBuild) -> dict[str, tuple[bool, dict]]:
    features = build.frames[SET_FEATURES]
    nulls = int(features[CORE_FEATURES].isna().sum().sum())
    ordered = bool(features["completed_at"].is_monotonic_increasing)
    return {
        "features_match_silver": (
            len(features) == build.eligible_silver_sets,
            {"gold_rows": len(features), "eligible_silver_sets": build.eligible_silver_sets},
        ),
        "no_null_features": (nulls == 0, {"null_feature_values": nulls}),
        "completed_at_non_decreasing": (ordered, {}),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    build = build_gold(load_lakehouse_catalog())
    summary = {name: len(frame) for name, frame in build.frames.items()}
    summary["checks"] = {name: passed for name, (passed, _) in gold_checks(build).items()}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
