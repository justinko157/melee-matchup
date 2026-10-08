"""Partition definitions shared by assets and automation."""

import dagster as dg

# Monday-to-Sunday weeks in UTC. end_offset=1 makes the in-progress week a
# partition too, so the daily schedule can refresh it.
WEEKLY_PARTITIONS = dg.WeeklyPartitionsDefinition(
    start_date="2018-01-01", day_offset=1, end_offset=1, timezone="UTC"
)
