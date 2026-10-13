import pytest

from pipeline.spark_jobs.locks import exclusive_lock


def test_silver_lock_refuses_a_second_job(tmp_path):
    pytest.importorskip("fcntl")  # the jobs only run in Linux containers
    held = exclusive_lock(str(tmp_path / "silver"), "silver")
    with pytest.raises(SystemExit, match="another silver job is running") as info:
        exclusive_lock(str(tmp_path / "silver"), "silver")
    assert info.value.code != 0
    held.close()
    exclusive_lock(str(tmp_path / "silver"), "silver").close()
