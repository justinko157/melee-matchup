import pytest

ICEBERG_RUNTIME = "org.apache.iceberg:iceberg-spark-runtime-4.0_2.13:1.11.0"


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    """Local SparkSession with an Iceberg Hadoop catalog named `local` as the default.

    Only Spark-marked tests use it; they need Java.
    """
    from pyspark.sql import SparkSession

    warehouse = tmp_path_factory.mktemp("iceberg-warehouse")
    session = (
        SparkSession.builder.master("local[1]")
        .appName("pipeline-tests")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.jars.packages", ICEBERG_RUNTIME)
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
        .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.local.type", "hadoop")
        .config("spark.sql.catalog.local.warehouse", warehouse.as_uri())
        .config("spark.sql.defaultCatalog", "local")
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


@pytest.fixture
def mlflow_tracking(tmp_path):
    """A throwaway SQLite-backed MLflow tracking store and registry."""
    mlflow = pytest.importorskip("mlflow")
    from pipeline.ml.names import EXPERIMENT

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    mlflow.create_experiment(EXPERIMENT, artifact_location=(tmp_path / "artifacts").as_uri())
    return uri
