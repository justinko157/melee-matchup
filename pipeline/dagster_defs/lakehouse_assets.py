"""Silver and gold assets. Both run on declarative automation after their upstream."""

import dagster as dg

from pipeline.dagster_defs.assets import bronze_startgg
from pipeline.dagster_defs.pools import SPARK_LAKEHOUSE
from pipeline.dagster_defs.resources import LakehouseResource, SparkJobResource
from pipeline.gold.build import build_gold, gold_checks
from pipeline.gold.tables import HEAD_TO_HEAD, PLAYER_RATINGS, SET_FEATURES

MAX_DROP_RATE = 0.10
SILVER_CHECKS = ("no_orphan_sets", "drop_rate_below_10pct", "winner_is_a_player")
GOLD_CHECKS = ("features_match_silver", "no_null_features", "completed_at_non_decreasing")


def drop_rate(summary: dict) -> float:
    """Dropped bronze set rows / bronze set rows read in this run (0 when nothing was read)."""
    read = summary["bronze_rows_read"]
    return sum(summary["dropped"].values()) / read if read else 0.0


@dg.asset(
    deps=[bronze_startgg],
    pool=SPARK_LAKEHOUSE,
    group_name="silver",
    kinds={"spark", "iceberg"},
    automation_condition=dg.AutomationCondition.eager(),
    retry_policy=dg.RetryPolicy(max_retries=2, delay=30, backoff=dg.Backoff.EXPONENTIAL),
    check_specs=[dg.AssetCheckSpec(name, asset="silver_startgg") for name in SILVER_CHECKS],
)
def silver_startgg(spark_job: SparkJobResource) -> dg.MaterializeResult:
    """New bronze rows, parsed and merged into the silver tables."""
    summary = spark_job.run_silver_job()
    rate = drop_rate(summary)
    metadata = {
        f"{entity}_{kind}": counts[kind]
        for entity, counts in summary["tables"].items()
        for kind in ("inserted", "updated")
    }
    metadata.update(
        {
            "bronze_rows_read": summary["bronze_rows_read"],
            "drop_rate": rate,
            "dropped": dg.MetadataValue.json(summary["dropped"]),
        }
    )
    return dg.MaterializeResult(
        metadata=metadata,
        check_results=[
            dg.AssetCheckResult(
                check_name="no_orphan_sets",
                passed=summary["orphan_sets"] == 0,
                metadata={"orphan_sets": summary["orphan_sets"]},
            ),
            dg.AssetCheckResult(
                check_name="drop_rate_below_10pct",
                passed=rate < MAX_DROP_RATE,
                metadata={"drop_rate": rate, **summary["dropped"]},
            ),
            dg.AssetCheckResult(
                check_name="winner_is_a_player",
                passed=summary["winner_not_player"] == 0,
                metadata={"winner_not_player": summary["winner_not_player"]},
            ),
        ],
    )


@dg.asset(
    deps=[silver_startgg],
    pool=SPARK_LAKEHOUSE,
    group_name="gold",
    kinds={"python", "iceberg"},
    automation_condition=dg.AutomationCondition.eager(),
    check_specs=[dg.AssetCheckSpec(name, asset="gold_startgg") for name in GOLD_CHECKS],
)
def gold_startgg(lakehouse: LakehouseResource) -> dg.MaterializeResult:
    """Feature, player and head-to-head tables recomputed from silver."""
    build = build_gold(lakehouse.get_catalog())
    return dg.MaterializeResult(
        metadata={
            "set_features_rows": build.row_counts[SET_FEATURES],
            "player_ratings_rows": build.row_counts[PLAYER_RATINGS],
            "head_to_head_rows": build.row_counts[HEAD_TO_HEAD],
            "silver_sets_snapshot": str(build.silver_sets_snapshot),
            "set_features_snapshot": str(build.snapshots[SET_FEATURES]),
        },
        check_results=[
            dg.AssetCheckResult(check_name=name, passed=passed, metadata=metadata)
            for name, (passed, metadata) in gold_checks(build).items()
        ],
    )
