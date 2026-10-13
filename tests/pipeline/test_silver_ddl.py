"""Silver DDL column lists must match the transform column lists. No SparkSession needed."""

import re

import pytest

pytest.importorskip("pyspark")

from pipeline.spark_jobs.silver import (  # noqa: E402
    MERGE_ORDER,
    WATERMARKS_DDL,
    silver_columns,
    silver_table_ddl,
)


def ddl_columns(ddl: str) -> tuple[str, ...]:
    body = re.search(r"\((.*)\)\s*USING", ddl, re.DOTALL).group(1)
    return tuple(column.split()[0] for column in body.split(","))


@pytest.mark.parametrize("entity", MERGE_ORDER)
def test_silver_ddl_matches_silver_columns(entity):
    ddl = silver_table_ddl(entity)
    assert ddl_columns(ddl) == silver_columns(entity)
    assert "'format-version'='2'" in ddl


def test_watermarks_ddl_columns():
    assert ddl_columns(WATERMARKS_DDL) == ("bronze_table", "snapshot_id", "updated_at")
