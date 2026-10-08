"""Ingestion and bronze landing assets."""

import dagster as dg

from pipeline.dagster_defs.partitions import WEEKLY_PARTITIONS
from pipeline.dagster_defs.resources import KafkaResource, SparkJobResource, StartGGResource
from pipeline.ingest.collect import collect_week


@dg.asset(
    partitions_def=WEEKLY_PARTITIONS,
    pool="startgg_api",
    group_name="ingest",
    kinds={"kafka", "python"},
    check_specs=[dg.AssetCheckSpec("no_dlq_messages", asset="startgg_raw")],
)
def startgg_raw(
    context: dg.AssetExecutionContext, startgg: StartGGResource, kafka: KafkaResource
) -> dg.MaterializeResult:
    """One week of start.gg tournaments and sets, published to the raw Kafka topics."""
    client = startgg.get_client()
    publisher = kafka.get_publisher()
    stats = collect_week(
        client,
        publisher,
        context.partition_key,
        run_id=context.run.run_id,
        min_attendees=startgg.min_attendees,
    )
    publisher.flush()
    return dg.MaterializeResult(
        metadata={
            "tournaments": stats.tournaments,
            "events": stats.events,
            "sets": stats.sets,
            "dlq_messages": stats.dlq_messages,
            "api_calls": client.api_calls,
        },
        check_results=[
            dg.AssetCheckResult(
                check_name="no_dlq_messages",
                passed=stats.dlq_messages == 0,
                metadata={"dlq_messages": stats.dlq_messages},
            )
        ],
    )


@dg.asset(
    deps=[startgg_raw],
    pool="spark_bronze",
    group_name="bronze",
    kinds={"spark", "iceberg"},
    retry_policy=dg.RetryPolicy(max_retries=2, delay=30, backoff=dg.Backoff.EXPONENTIAL),
    check_specs=[dg.AssetCheckSpec("no_rejects", asset="bronze_startgg")],
)
def bronze_startgg(spark_job: SparkJobResource) -> dg.MaterializeResult:
    """Everything new in the raw topics, landed into the Iceberg bronze tables."""
    summary = spark_job.run_bronze_job()
    return dg.MaterializeResult(
        metadata={
            "tournament_rows": summary["tournaments"],
            "set_rows": summary["sets"],
            "reject_rows": summary["rejects"],
        },
        check_results=[
            dg.AssetCheckResult(
                check_name="no_rejects",
                passed=summary["rejects"] == 0,
                metadata={"reject_rows": summary["rejects"]},
            )
        ],
    )
