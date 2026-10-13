"""Maximum run times, in seconds, as dagster/max_runtime run tags.

A run whose worker dies stays STARTED and keeps its pool slot. The daemon's run
monitoring (enabled in infra/dagster/dagster.yaml) fails any run that outlives
its tag, which frees the slot. Untagged runs fall back to run_monitoring's
max_runtime_seconds there (the longest limit below).
"""

MAX_RUNTIME_TAG = "dagster/max_runtime"

INGEST = 1800  # one week of start.gg (normally under 6 minutes)
BRONZE = 16200  # Spark: up to three 1-hour attempts plus slack
SILVER = 16200  # Spark: one 4-hour attempt plus slack
GOLD = 3600
TRAINING = 3600
LAKEHOUSE = SILVER + GOLD  # automation usually runs silver and gold together
MAINTENANCE = 9000  # the 2-hour Spark timeout plus slack


def max_runtime(seconds: int) -> dict[str, str]:
    return {MAX_RUNTIME_TAG: str(seconds)}
