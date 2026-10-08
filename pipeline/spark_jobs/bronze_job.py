"""Land the raw Kafka topics into Iceberg bronze tables, then stop (availableNow).

Run with spark-submit. The last line of stdout is a JSON summary of rows added.
Each target table has its own streaming query and checkpoint, so every write
goes through Iceberg's exactly-once streaming sink.
"""

import argparse
import json
from datetime import UTC, datetime

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.streaming import StreamingQuery

from pipeline.spark_jobs.bronze import (
    BRONZE_NAMESPACE,
    BRONZE_TABLES,
    REJECTS_DDL,
    REJECTS_TABLE,
    bronze_rows,
    bronze_table_ddl,
    parse_envelopes,
    reject_rows,
)
from pipeline.topics import RAW_TOPICS, SETS_TOPIC, TOURNAMENTS_TOPIC


def read_kafka(spark: SparkSession, bootstrap_servers: str, topics: list[str]) -> DataFrame:
    return (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("subscribe", ",".join(topics))
        .option("startingOffsets", "earliest")
        .load()
    )


def write_once(df: DataFrame, table: str, checkpoint: str) -> StreamingQuery:
    return (
        df.writeStream.format("iceberg")
        .outputMode("append")
        .trigger(availableNow=True)
        .option("checkpointLocation", checkpoint)
        .option("fanout-enabled", "true")
        .toTable(table)
    )


def added_rows_since(spark: SparkSession, table: str, since: datetime) -> int:
    """Rows committed to a table since `since`, from Iceberg snapshot summaries."""
    row = spark.sql(
        f"""
        SELECT coalesce(sum(cast(summary['added-records'] AS BIGINT)), 0) AS n
        FROM {table}.snapshots
        WHERE committed_at >= TIMESTAMP '{since:%Y-%m-%d %H:%M:%S.%f}'
        """
    ).first()
    return int(row.n)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-servers", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    args = parser.parse_args(argv)

    spark = SparkSession.builder.appName("bronze-startgg").getOrCreate()
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {BRONZE_NAMESPACE}")
    for table in BRONZE_TABLES.values():
        spark.sql(bronze_table_ddl(table))
    spark.sql(REJECTS_DDL)

    started = datetime.now(UTC)
    servers, root = args.bootstrap_servers, args.checkpoint_root
    queries = [
        write_once(
            bronze_rows(
                parse_envelopes(read_kafka(spark, servers, [TOURNAMENTS_TOPIC])), "tournament"
            ),
            BRONZE_TABLES["tournament"],
            f"{root}/tournaments",
        ),
        write_once(
            bronze_rows(parse_envelopes(read_kafka(spark, servers, [SETS_TOPIC])), "set"),
            BRONZE_TABLES["set"],
            f"{root}/sets",
        ),
        write_once(
            reject_rows(parse_envelopes(read_kafka(spark, servers, list(RAW_TOPICS)))),
            REJECTS_TABLE,
            f"{root}/rejects",
        ),
    ]
    for query in queries:
        query.awaitTermination()

    summary = {
        "tournaments": added_rows_since(spark, BRONZE_TABLES["tournament"], started),
        "sets": added_rows_since(spark, BRONZE_TABLES["set"], started),
        "rejects": added_rows_since(spark, REJECTS_TABLE, started),
    }
    spark.stop()
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
