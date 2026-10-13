from pipeline.spark_jobs.watermarks import ReadPlan, Snapshot, plan_read

LINEAGE = [Snapshot(1, "append"), Snapshot(2, "append"), Snapshot(3, "append")]


def test_empty_table_is_skipped():
    assert plan_read(None, []) == ReadPlan("skip")


def test_first_run_reads_everything():
    assert plan_read(None, LINEAGE) == ReadPlan("full", end=3)


def test_unchanged_table_is_skipped():
    assert plan_read(3, LINEAGE) == ReadPlan("skip", end=3)


def test_advanced_table_is_read_incrementally():
    assert plan_read(1, LINEAGE) == ReadPlan("incremental", end=3, start=1)


def test_expired_watermark_falls_back_to_full_scan():
    assert plan_read(99, LINEAGE) == ReadPlan("full", end=3)


def test_overwrite_after_the_watermark_forces_a_full_scan():
    lineage = [Snapshot(1, "append"), Snapshot(2, "overwrite"), Snapshot(3, "append")]
    assert plan_read(1, lineage) == ReadPlan("full", end=3)
