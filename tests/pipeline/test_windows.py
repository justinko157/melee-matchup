import pytest

from pipeline.ingest.windows import week_bounds


def test_week_bounds_cover_monday_to_sunday_utc():
    # 2025-01-06 00:00:00 UTC = 1736121600; next Monday = 1736726400
    assert week_bounds("2025-01-06") == (1736121600, 1736726399)


def test_adjacent_weeks_do_not_overlap():
    _, first_end = week_bounds("2025-01-06")
    second_start, _ = week_bounds("2025-01-13")
    assert second_start == first_end + 1


def test_rejects_non_monday():
    with pytest.raises(ValueError, match="not a Monday"):
        week_bounds("2025-01-07")


def test_rejects_malformed_key():
    with pytest.raises(ValueError):
        week_bounds("2025-W02")
