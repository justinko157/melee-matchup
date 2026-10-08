"""Weekly partition windows for start.gg ingestion."""

from datetime import UTC, datetime, timedelta

WEEK = timedelta(days=7)


def week_bounds(partition_key: str) -> tuple[int, int]:
    """Return (after_date, before_date) unix timestamps for a weekly partition.

    The window runs from Monday 00:00:00 UTC to the following Sunday 23:59:59
    UTC, so adjacent weeks never overlap.
    """
    start = datetime.strptime(partition_key, "%Y-%m-%d").replace(tzinfo=UTC)
    if start.weekday() != 0:
        raise ValueError(f"Partition key {partition_key!r} is not a Monday")
    end = start + WEEK
    return int(start.timestamp()), int(end.timestamp()) - 1
