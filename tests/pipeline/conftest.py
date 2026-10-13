import pytest


@pytest.fixture(scope="session")
def spark():
    """Local SparkSession. Only Spark-marked tests use it; they need Java."""
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[1]")
        .appName("pipeline-tests")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture
def sql_settings(tmp_path):
    """A PyIceberg SQL catalog in a temp dir; stands in for Lakekeeper in unit tests."""
    pytest.importorskip("sqlalchemy")
    from pipeline.lakehouse import LakehouseSettings

    return LakehouseSettings(
        catalog_type="sql",
        uri=f"sqlite:///{(tmp_path / 'catalog.db').as_posix()}",
        warehouse=tmp_path.as_uri(),
    )


@pytest.fixture
def sql_catalog(sql_settings):
    from pipeline.lakehouse import load_lakehouse_catalog

    return load_lakehouse_catalog(sql_settings)
