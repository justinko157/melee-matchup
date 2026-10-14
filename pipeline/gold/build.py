"""Recompute every gold table from silver. Run directly for a one-off build.

A build is skipped when every gold table was already written from the current
snapshots of all silver inputs.
"""

import argparse
import json
import logging
from dataclasses import dataclass

import pandas as pd
from pyiceberg.catalog import Catalog

from pipeline.gold.derive import head_to_head, player_ratings
from pipeline.gold.features import build_features
from pipeline.gold.tables import (
    CHECKS_PROPERTY,
    GOLD_TABLES,
    HEAD_TO_HEAD,
    PLAYER_RATINGS,
    SET_FEATURES,
    SILVER_INPUTS_PROPERTY,
    SILVER_SNAPSHOT_PROPERTY,
)
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
# silver.sets first: it is pinned, and the parents read after it are at least as new.
SILVER_INPUTS = ("silver.sets", "silver.events", "silver.tournaments", "silver.players")

logger = logging.getLogger(__name__)


Checks = dict[str, tuple[bool, dict]]


@dataclass
class GoldBuild:
    """What a build wrote. Holds only scalars: the frames are released as they are written."""

    row_counts: dict[str, int]
    checks: Checks
    eligible_silver_sets: int
    silver_sets_snapshot: int | None
    snapshots: dict[str, int | None]
    skipped: bool = False


def silver_fingerprint(catalog: Catalog) -> dict[str, int | None]:
    """The current snapshot of every silver table gold reads."""
    return {identifier: current_snapshot_id(catalog, identifier) for identifier in SILVER_INPUTS}


def recorded_build(catalog: Catalog, fingerprint: dict[str, int | None]) -> GoldBuild | None:
    """The build stored on gold, if every gold table was written from ``fingerprint``."""
    expected = json.dumps(fingerprint, sort_keys=True)
    row_counts: dict[str, int] = {}
    snapshots: dict[str, int | None] = {}
    stored_checks = set()
    for identifier in GOLD_TABLES:
        if not catalog.table_exists(identifier):
            return None
        table = catalog.load_table(identifier)
        properties = table.properties
        if properties.get(SILVER_INPUTS_PROPERTY) != expected or CHECKS_PROPERTY not in properties:
            return None
        stored_checks.add(properties[CHECKS_PROPERTY])
        snapshot = table.current_snapshot()
        snapshots[identifier] = None if snapshot is None else snapshot.snapshot_id
        records = None if snapshot is None else snapshot.summary.get("total-records")
        row_counts[identifier] = int(records or 0)
    if len(stored_checks) != 1:
        return None
    checks = {name: (passed, meta) for name, (passed, meta) in json.loads(*stored_checks).items()}
    eligible = checks["features_match_silver"][1]["eligible_silver_sets"]
    sets_snapshot = fingerprint["silver.sets"]
    return GoldBuild(row_counts, checks, eligible, sets_snapshot, snapshots, skipped=True)


def load_silver(catalog: Catalog, snapshot: int | None) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """(sets joined to their tournament, players, eligible set count).

    Sets whose event or tournament is missing drop out of the join, as they did in
    the original SQL. The eligible count is taken before the join, so a gap shows
    up in the features_match_silver check.
    """
    # Read silver.sets (pinned to ``snapshot``) before its parents on purpose: the silver
    # job MERGEs parents first, so parents read afterwards are at least as new as the sets.
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
    return joined, players, eligible


def feature_checks(features: pd.DataFrame, eligible_silver_sets: int) -> Checks:
    nulls = int(features[CORE_FEATURES].isna().sum().sum())
    ordered = bool(features["completed_at"].is_monotonic_increasing)
    return {
        "features_match_silver": (
            len(features) == eligible_silver_sets,
            {"gold_rows": len(features), "eligible_silver_sets": eligible_silver_sets},
        ),
        "no_null_features": (nulls == 0, {"null_feature_values": nulls}),
        "completed_at_non_decreasing": (ordered, {}),
    }


def build_gold(catalog: Catalog, *, force: bool = False) -> GoldBuild:
    """Rebuild gold from silver, unless it was already built from the same silver snapshots."""
    fingerprint = silver_fingerprint(catalog)
    if not force and (recorded := recorded_build(catalog, fingerprint)) is not None:
        logger.info("Gold is current for silver snapshots %s; skipping the build", fingerprint)
        return recorded
    snapshot = fingerprint["silver.sets"]
    sets, players, eligible = load_silver(catalog, snapshot)
    features, state = build_features(sets)
    del sets
    checks = feature_checks(features, eligible)
    frames = {
        SET_FEATURES: features,
        PLAYER_RATINGS: player_ratings(state, players),
        HEAD_TO_HEAD: head_to_head(state),
    }
    del features, state, players
    row_counts = {identifier: len(frame) for identifier, frame in frames.items()}
    properties = {
        SILVER_SNAPSHOT_PROPERTY: "" if snapshot is None else str(snapshot),
        SILVER_INPUTS_PROPERTY: json.dumps(fingerprint, sort_keys=True),
        CHECKS_PROPERTY: json.dumps(checks, sort_keys=True),
    }
    snapshots = write_gold(catalog, frames, properties)  # consumes frames
    return GoldBuild(row_counts, checks, eligible, snapshot, snapshots)


def gold_checks(build: GoldBuild) -> Checks:
    """Check results, computed from set_features before it was written and released."""
    return build.checks


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild even if gold is current")
    args = parser.parse_args()
    build = build_gold(load_lakehouse_catalog(), force=args.force)
    summary: dict = {**build.row_counts, "skipped": build.skipped}
    summary["checks"] = {name: passed for name, (passed, _) in gold_checks(build).items()}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
