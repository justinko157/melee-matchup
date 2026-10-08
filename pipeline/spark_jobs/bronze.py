"""Pure transforms from Kafka source rows to bronze rows and rejects, plus table DDL."""

from functools import reduce

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from pipeline.topics import SETS_TOPIC, TOURNAMENTS_TOPIC

ENVELOPE_FIELDS = (
    "entity",
    "entity_id",
    "ingested_at",
    "dagster_run_id",
    "partition_week",
    "query_name",
    "query_variables",
    "payload",
)
# Declaring nested objects as strings makes from_json keep their raw JSON text.
ENVELOPE_SCHEMA = StructType([StructField(name, StringType()) for name in ENVELOPE_FIELDS])
REQUIRED_FIELDS = ("entity", "entity_id", "ingested_at", "payload")

TOPIC_ENTITIES = {TOURNAMENTS_TOPIC: "tournament", SETS_TOPIC: "set"}

BRONZE_NAMESPACE = "bronze"
BRONZE_TABLES = {"tournament": "bronze.startgg_tournaments", "set": "bronze.startgg_sets"}
REJECTS_TABLE = "bronze.startgg_rejects"

KAFKA_COLUMNS = ("kafka_topic", "kafka_partition", "kafka_offset", "kafka_timestamp")
BRONZE_COLUMNS = (
    *KAFKA_COLUMNS,
    "kafka_key",
    "entity",
    "entity_id",
    "ingested_at",
    "dagster_run_id",
    "partition_week",
    "query_name",
    "query_variables",
    "payload",
)
REJECT_COLUMNS = (*KAFKA_COLUMNS, "raw_key", "raw_value", "reject_reason", "rejected_at")


def bronze_table_ddl(table: str) -> str:
    return f"""
        CREATE TABLE IF NOT EXISTS {table} (
            kafka_topic STRING, kafka_partition INT, kafka_offset BIGINT,
            kafka_timestamp TIMESTAMP, kafka_key STRING,
            entity STRING, entity_id STRING, ingested_at TIMESTAMP,
            dagster_run_id STRING, partition_week STRING,
            query_name STRING, query_variables STRING, payload STRING
        ) USING iceberg
        PARTITIONED BY (days(ingested_at))
    """


REJECTS_DDL = f"""
    CREATE TABLE IF NOT EXISTS {REJECTS_TABLE} (
        kafka_topic STRING, kafka_partition INT, kafka_offset BIGINT,
        kafka_timestamp TIMESTAMP, raw_key BINARY, raw_value BINARY,
        reject_reason STRING, rejected_at TIMESTAMP
    ) USING iceberg
    PARTITIONED BY (days(kafka_timestamp))
"""


def _any_null(columns: list[Column]) -> Column:
    return reduce(lambda a, b: a | b, [c.isNull() for c in columns])


def _all_null(columns: list[Column]) -> Column:
    return reduce(lambda a, b: a & b, [c.isNull() for c in columns])


def parse_envelopes(kafka_df: DataFrame) -> DataFrame:
    """Parse each Kafka value as an envelope and decide whether to reject it."""
    expected_entity = F.create_map(*[F.lit(x) for pair in TOPIC_ENTITIES.items() for x in pair])[
        F.col("topic")
    ]
    parsed = kafka_df.select(
        F.col("topic").alias("kafka_topic"),
        F.col("partition").alias("kafka_partition"),
        F.col("offset").alias("kafka_offset"),
        F.col("timestamp").alias("kafka_timestamp"),
        F.col("key").cast("string").alias("kafka_key"),
        F.col("key").alias("raw_key"),
        F.col("value").alias("raw_value"),
        F.from_json(F.col("value").cast("string"), ENVELOPE_SCHEMA).alias("env"),
        expected_entity.alias("expected_entity"),
    )
    env_fields = [F.col(f"env.{name}") for name in ENVELOPE_FIELDS]
    ingested_ts = F.try_to_timestamp(F.col("env.ingested_at"))
    reject_reason = (
        F.when(F.col("raw_value").isNull(), "null_value")
        .when(F.col("env").isNull() | _all_null(env_fields), "unparseable_envelope")
        .when(
            _any_null([F.col(f"env.{name}") for name in REQUIRED_FIELDS]),
            "missing_required_field",
        )
        .when(ingested_ts.isNull(), "invalid_ingested_at")
        .when(~F.col("env.entity").eqNullSafe(F.col("expected_entity")), "entity_topic_mismatch")
    )
    return parsed.select(
        "*", ingested_ts.alias("ingested_ts"), reject_reason.alias("reject_reason")
    )


def bronze_rows(parsed: DataFrame, entity: str) -> DataFrame:
    """Accepted rows for one entity, in bronze table column order."""
    return parsed.where(
        F.col("reject_reason").isNull() & (F.col("env.entity") == F.lit(entity))
    ).select(
        *KAFKA_COLUMNS,
        "kafka_key",
        F.col("env.entity").alias("entity"),
        F.col("env.entity_id").alias("entity_id"),
        F.col("ingested_ts").alias("ingested_at"),
        F.col("env.dagster_run_id").alias("dagster_run_id"),
        F.col("env.partition_week").alias("partition_week"),
        F.col("env.query_name").alias("query_name"),
        F.col("env.query_variables").alias("query_variables"),
        F.col("env.payload").alias("payload"),
    )


def reject_rows(parsed: DataFrame) -> DataFrame:
    """Rows that failed validation, with the raw bytes kept for inspection."""
    return parsed.where(F.col("reject_reason").isNotNull()).select(
        *KAFKA_COLUMNS,
        "raw_key",
        "raw_value",
        "reject_reason",
        F.current_timestamp().alias("rejected_at"),
    )
