"""Concurrency pools. Each is limited to one run (see the dagster-pools service)."""

STARTGG_API = "startgg_api"  # the start.gg rate limit is shared
# Bronze, silver, gold, training and maintenance: every run executes inside
# dagster-code (3 GB), so only one memory-heavy run (Spark, gold or training) at a time.
SPARK_LAKEHOUSE = "spark_lakehouse"
ALL_POOLS = (STARTGG_API, SPARK_LAKEHOUSE)
