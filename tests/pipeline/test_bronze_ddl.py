"""DDL column lists must match the transform column lists. No SparkSession needed."""

import re

import pytest

pytest.importorskip("pyspark")

from pipeline.spark_jobs.bronze import (  # noqa: E402
    BRONZE_COLUMNS,
    REJECT_COLUMNS,
    REJECTS_DDL,
    bronze_table_ddl,
)


def ddl_columns(ddl: str) -> tuple[str, ...]:
    """Column names, in order, from a CREATE TABLE ... ( name TYPE, ... ) statement."""
    body = re.search(r"\((.*)\)\s*USING", ddl, re.DOTALL).group(1)
    return tuple(column.split()[0] for column in body.split(","))


def test_bronze_ddl_matches_bronze_columns():
    assert ddl_columns(bronze_table_ddl("bronze.startgg_sets")) == BRONZE_COLUMNS


def test_rejects_ddl_matches_reject_columns():
    assert ddl_columns(REJECTS_DDL) == REJECT_COLUMNS
