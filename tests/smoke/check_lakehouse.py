"""Assert silver, gold and MLflow after the smoke test's runs.

Usage: python -m tests.smoke.check_lakehouse --fixture PATH --week KEY --copies N
"""

import argparse

import dagster as dg
from mlflow import MlflowClient

from pipeline.gold.tables import HEAD_TO_HEAD, PLAYER_RATINGS, SET_FEATURES
from pipeline.ingest.collect import collect_week
from pipeline.ingest.fixtures import FixtureClient
from pipeline.lakehouse import current_snapshot_id, load_lakehouse_catalog
from pipeline.ml.names import GOLD_SNAPSHOT_TAG, MODEL_NAME
from pipeline.serving.gold_reader import load_head_to_head, load_player_ratings, lookup_h2h


class CapturingPublisher:
    def __init__(self):
        self.set_payloads: list[dict] = []

    def publish(self, topic: str, key: str, value: dict) -> None:
        if value["entity"] == "set":
            self.set_payloads.append(value["payload"])


def slot_player_id(slot: dict):
    participants = (slot.get("entrant") or {}).get("participants") or [{}]
    return ((participants[0] or {}).get("player") or {}).get("id")


def expected_kept_sets(payloads: list[dict]) -> dict[int, dict]:
    """The silver set rules, reimplemented independently as the smoke test's oracle."""
    kept = {}
    for node in payloads:
        slots = node.get("slots") or []
        if len(slots) < 2:
            continue
        if not all(slot_player_id(slot) for slot in slots[:2]):
            continue
        if node.get("state") != 3 or not str(node.get("id")).isdigit():
            continue
        entrants = [(slot.get("entrant") or {}).get("id") for slot in slots[:2]]
        if node.get("winnerId") not in entrants:
            continue
        kept[int(node["id"])] = node
    return kept


def row_count(catalog, identifier: str) -> int:
    return catalog.load_table(identifier).scan().to_arrow().num_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--week", required=True)
    parser.add_argument("--copies", type=int, required=True)
    args = parser.parse_args()

    publisher = CapturingPublisher()
    stats = collect_week(FixtureClient(args.fixture), publisher, args.week, run_id="check")
    kept = expected_kept_sets(publisher.set_payloads)
    games = {g["id"] for node in kept.values() for g in (node.get("games") or []) if g.get("id")}
    eligible = sum(1 for node in kept.values() if node.get("completedAt") is not None)
    assert kept and eligible, f"fixture has no eligible sets for {args.week}"

    catalog = load_lakehouse_catalog()
    # Replays must not change silver or gold: the counts are the same on every pass.
    exact = {
        "silver.tournaments": stats.tournaments,
        "silver.events": stats.events,
        "silver.sets": len(kept),
        "silver.games": len(games),
        SET_FEATURES: eligible,
    }
    for identifier, expected in exact.items():
        actual = row_count(catalog, identifier)
        assert actual == expected, f"{identifier}: {actual} rows, expected {expected}"
        print(f"{identifier}: {actual} rows")
    for identifier in ("silver.players", PLAYER_RATINGS, HEAD_TO_HEAD):
        actual = row_count(catalog, identifier)
        assert actual > 0, f"{identifier} is empty"
        print(f"{identifier}: {actual} rows")

    # The app's own readers must work against these tables.
    assert len(load_player_ratings(catalog, min_sets=1)) > 0, "app reader returned no players"
    first = next(iter(kept.values()))
    p1, p2 = (int(slot_player_id(slot)) for slot in first["slots"][:2])
    assert sum(lookup_h2h(load_head_to_head(catalog), p1, p2)) >= 1, f"no h2h for {p1} vs {p2}"

    versions = MlflowClient().search_model_versions(f"name='{MODEL_NAME}'")
    assert len(versions) == args.copies, f"{len(versions)} model versions, expected {args.copies}"
    latest = max(versions, key=lambda version: int(version.version))
    gold_snapshot = str(current_snapshot_id(catalog, SET_FEATURES))
    tagged = latest.tags.get(GOLD_SNAPSHOT_TAG)
    assert tagged == gold_snapshot, (
        f"model v{latest.version} tagged {tagged}, gold is {gold_snapshot}"
    )
    event = dg.DagsterInstance.get().get_latest_materialization_event(dg.AssetKey("matchup_model"))
    decision = event.asset_materialization.metadata["promotion_decision"].value
    print(f"model v{latest.version}: promotion decision {decision}")


if __name__ == "__main__":
    main()
