"""Upsert new bronze rows into the silver tables.

Run with spark-submit. The last line of stdout is a JSON summary.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from pipeline.spark_jobs.bronze import BRONZE_TABLES
from pipeline.spark_jobs.silver import (
    DROP_REASONS,
    KEYS,
    MERGE_ORDER,
    SILVER_NAMESPACE,
    WATERMARKS_DDL,
    WATERMARKS_TABLE,
    drop_counts,
    event_rows,
    finalize,
    game_rows,
    merge_sql,
    newest_per_key,
    parse_sets,
    parse_tournaments,
    player_rows,
    set_rows,
    silver_table_ddl,
    table_name,
    tournament_rows,
)
from pipeline.spark_jobs.watermarks import ReadPlan, Snapshot, plan_read

BRONZE_SOURCES = {"tournaments": BRONZE_TABLES["tournament"], "sets": BRONZE_TABLES["set"]}


def ensure_tables(spark: SparkSession) -> None:
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {SILVER_NAMESPACE}")
    for entity in MERGE_ORDER:
        spark.sql(silver_table_ddl(entity))
    spark.sql(WATERMARKS_DDL)


def read_watermarks(spark: SparkSession) -> dict[str, int]:
    return {row.bronze_table: row.snapshot_id for row in spark.table(WATERMARKS_TABLE).collect()}


def snapshot_lineage(spark: SparkSession, table: str) -> list[Snapshot]:
    """Live snapshots on the current branch, oldest first ([] if the table doesn't exist)."""
    if not spark.catalog.tableExists(table):
        return []
    rows = spark.sql(
        f"""
        SELECT h.snapshot_id, s.operation
        FROM {table}.history AS h
        JOIN {table}.snapshots AS s ON h.snapshot_id = s.snapshot_id
        WHERE h.is_current_ancestor
        ORDER BY h.made_current_at
        """
    ).collect()
    return [Snapshot(row.snapshot_id, row.operation) for row in rows]


def read_bronze(spark: SparkSession, table: str, plan: ReadPlan) -> DataFrame:
    reader = spark.read.format("iceberg")
    if plan.kind == "incremental":
        reader = reader.option("start-snapshot-id", str(plan.start)).option(
            "end-snapshot-id", str(plan.end)
        )
    else:
        reader = reader.option("snapshot-id", str(plan.end))
    return reader.load(table)


def count_changes(spark: SparkSession, entity: str, staged: DataFrame) -> dict[str, int]:
    """How many staged rows will be inserted, and how many will update an existing row."""
    key = KEYS[entity]
    existing = spark.table(table_name(entity)).select(
        F.col(key), F.col("source_ingested_at").alias("_target_ts")
    )
    row = (
        staged.join(existing, key, "left")
        .agg(
            F.sum(F.when(F.col("_target_ts").isNull(), 1).otherwise(0)).alias("inserted"),
            F.sum(F.when(F.col("source_ingested_at") >= F.col("_target_ts"), 1).otherwise(0)).alias(
                "updated"
            ),
        )
        .first()
    )
    return {"inserted": int(row.inserted or 0), "updated": int(row.updated or 0)}


def merge_into(spark: SparkSession, entity: str, staged: DataFrame) -> dict[str, int]:
    staged = staged.persist()
    try:
        changes = count_changes(spark, entity, staged)
        view = f"staged_{entity}"
        staged.createOrReplaceTempView(view)
        spark.sql(merge_sql(entity, view))
        return changes
    finally:
        staged.unpersist()


def write_watermarks(spark: SparkSession, snapshots: dict[str, int]) -> None:
    new = spark.createDataFrame(
        list(snapshots.items()), "bronze_table STRING, snapshot_id BIGINT"
    ).withColumn("updated_at", F.current_timestamp())
    new.createOrReplaceTempView("new_watermarks")
    spark.sql(
        f"""
        MERGE INTO {WATERMARKS_TABLE} AS t USING new_watermarks AS s
        ON t.bronze_table = s.bronze_table
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
        """
    )


def integrity_counts(spark: SparkSession) -> dict[str, int]:
    sets, events = table_name("sets"), table_name("events")
    orphans = spark.sql(
        f"SELECT count(*) AS n FROM {sets} AS s LEFT ANTI JOIN {events} AS e "
        f"ON s.event_id = e.event_id"
    ).first()
    bad_winners = spark.sql(
        f"SELECT count(*) AS n FROM {sets} WHERE winner_player_id IS NOT NULL "
        f"AND winner_player_id NOT IN (player1_id, player2_id)"
    ).first()
    return {"orphan_sets": int(orphans.n), "winner_not_player": int(bad_winners.n)}


def run_silver(spark: SparkSession) -> dict:
    """Read new bronze snapshots, MERGE them into silver, then advance the watermarks."""
    ensure_tables(spark)
    watermarks = read_watermarks(spark)
    plans = {
        source: plan_read(watermarks.get(table), snapshot_lineage(spark, table))
        for source, table in BRONZE_SOURCES.items()
    }
    summary = {
        "tables": {entity: {"inserted": 0, "updated": 0} for entity in MERGE_ORDER},
        "dropped": {reason: 0 for reason in DROP_REASONS},
        "bronze_rows_read": 0,
        "bronze_tournament_rows_read": 0,
    }
    staged: dict[str, DataFrame] = {}
    # Parsed bronze is persisted so from_json runs once, not once per Spark action
    # (row counts, drop counts and one MERGE per derived entity).
    persisted: list[DataFrame] = []

    try:
        plan = plans["tournaments"]
        if plan.kind != "skip":
            bronze = read_bronze(spark, BRONZE_SOURCES["tournaments"], plan)
            summary["bronze_tournament_rows_read"] = bronze.count()
            parsed = parse_tournaments(bronze)
            persisted.append(parsed.persist())
            staged["tournaments"] = finalize(
                newest_per_key(tournament_rows(parsed), "tournament_id"), "tournaments", plan.end
            )
            staged["events"] = finalize(
                newest_per_key(event_rows(parsed), "event_id"), "events", plan.end
            )

        plan = plans["sets"]
        if plan.kind != "skip":
            parsed = parse_sets(read_bronze(spark, BRONZE_SOURCES["sets"], plan))
            persisted.append(parsed.persist())
            summary["bronze_rows_read"], summary["dropped"] = drop_counts(parsed)
            staged["players"] = finalize(
                newest_per_key(player_rows(parsed), "player_id"), "players", plan.end
            )
            staged["sets"] = finalize(newest_per_key(set_rows(parsed), "set_id"), "sets", plan.end)
            staged["games"] = finalize(
                newest_per_key(game_rows(parsed), "game_id"), "games", plan.end
            )

        for entity in MERGE_ORDER:
            if entity in staged:
                summary["tables"][entity] = merge_into(spark, entity, staged[entity])
    finally:
        for df in persisted:
            df.unpersist()

    advanced = {BRONZE_SOURCES[s]: p.end for s, p in plans.items() if p.kind != "skip"}
    if advanced:
        write_watermarks(spark, advanced)
    summary.update(integrity_counts(spark))
    return summary
