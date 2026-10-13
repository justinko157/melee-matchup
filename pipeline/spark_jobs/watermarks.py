"""Decide how much of a bronze table the silver job must read. No Spark imports."""

from dataclasses import dataclass

# Compaction (replace) snapshots carry no new rows and are skipped by Iceberg's
# incremental append scan. Anything else after the watermark forces a full read.
INCREMENTAL_SAFE = frozenset({"append", "replace"})


@dataclass(frozen=True)
class Snapshot:
    snapshot_id: int
    operation: str


@dataclass(frozen=True)
class ReadPlan:
    kind: str  # "skip" | "full" | "incremental"
    end: int | None = None
    start: int | None = None


def plan_read(watermark: int | None, lineage: list[Snapshot]) -> ReadPlan:
    """lineage: the table's live current-ancestor snapshots, oldest first."""
    if not lineage:
        return ReadPlan("skip")
    current = lineage[-1].snapshot_id
    if watermark == current:
        return ReadPlan("skip", end=current)
    ids = [snapshot.snapshot_id for snapshot in lineage]
    if watermark is None or watermark not in ids:
        return ReadPlan("full", end=current)
    later = lineage[ids.index(watermark) + 1 :]
    if any(snapshot.operation not in INCREMENTAL_SAFE for snapshot in later):
        return ReadPlan("full", end=current)
    return ReadPlan("incremental", end=current, start=watermark)
