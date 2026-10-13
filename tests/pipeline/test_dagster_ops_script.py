import importlib.util
from datetime import date
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("dagster_ops", Path("scripts/dagster_ops.py"))
dagster_ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dagster_ops)


def test_backfill_covers_every_monday_inclusive():
    keys = dagster_ops.weekly_partition_keys(date(2018, 1, 1), date(2024, 12, 30))
    assert len(keys) == 366
    assert (keys[0], keys[1], keys[-1]) == ("2018-01-01", "2018-01-08", "2024-12-30")


def test_backfill_bounds_must_be_mondays():
    with pytest.raises(ValueError, match="Monday"):
        dagster_ops.weekly_partition_keys(date(2018, 1, 2), date(2024, 12, 30))
