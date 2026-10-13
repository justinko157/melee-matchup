"""Concurrency pools. Each is limited to one run (see the dagster-pools service)."""

STARTGG_API = "startgg_api"  # the start.gg rate limit is shared
SPARK_LAKEHOUSE = "spark_lakehouse"  # bronze, silver and maintenance: one Spark driver at a time
GOLD = "gold"
TRAINING = "training"
ALL_POOLS = (STARTGG_API, SPARK_LAKEHOUSE, GOLD, TRAINING)
