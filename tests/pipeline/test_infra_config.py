"""Keep infra config consistent with the Python code."""

import re
import tomllib
from pathlib import Path

from pipeline.dagster_defs.pools import ALL_POOLS
from pipeline.topics import ALL_TOPICS

COMPOSE = Path("infra/docker-compose.yml").read_text(encoding="utf-8")
SPARK_DEFAULTS = Path("infra/spark/spark-defaults.conf").read_text(encoding="utf-8")


def test_compose_creates_every_topic():
    for topic in ALL_TOPICS:
        assert topic in COMPOSE, f"{topic} missing from kafka-init"


def test_kafka_connector_matches_pyspark_version():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    (pin,) = [
        d
        for d in pyproject["project"]["optional-dependencies"]["pipeline"]
        if d.startswith("pyspark")
    ]
    pyspark_version = pin.split("==")[1]
    assert f"spark-sql-kafka-0-10_2.13:{pyspark_version}" in SPARK_DEFAULTS
    minor = ".".join(pyspark_version.split(".")[:2])
    assert f"iceberg-spark-runtime-{minor}_2.13" in SPARK_DEFAULTS


def test_compose_images_are_pinned():
    images = re.findall(r"^\s*image:\s*(\S+)", COMPOSE, flags=re.MULTILINE)
    assert images
    for image in images:
        if image in ("melee-pipeline:dev", "melee-app:dev"):
            continue
        assert ":" in image and not image.endswith((":latest", ":latest-main")), image


def test_lakekeeper_bootstrap_only_tolerates_already_bootstrapped():
    service = COMPOSE.split("lakekeeper-bootstrap:", 1)[1].split("\n  lakekeeper-warehouse:", 1)[0]
    assert "CatalogAlreadyBootstrapped" in service
    assert "exit 1" in service


def test_makefile_env_file_is_optional():
    makefile = Path("Makefile").read_text(encoding="utf-8")
    assert "$(if $(wildcard .env),--env-file .env)" in makefile


def test_pipeline_image_installs_the_ml_extra():
    dockerfile = Path("infra/pipeline.Dockerfile").read_text(encoding="utf-8")
    assert "--extra pipeline --extra ml --extra dev" in dockerfile


def _service(name: str) -> str:
    """The text of one top-level Compose service block (two-space indented key)."""
    start = COMPOSE.index(f"\n  {name}:\n")
    rest = COMPOSE[start + 1 :]
    following = re.search(r"\n  [a-z][a-z0-9-]*:\n", rest[1:])
    return rest if following is None else rest[: following.start() + 1]


def test_mlflow_uses_postgres_silo_and_one_worker():
    mlflow = _service("mlflow")
    assert "ghcr.io/mlflow/mlflow:v3.16.1-full" in mlflow
    assert "postgresql+psycopg2://" in mlflow and "@postgres:5432/mlflow" in mlflow
    assert "s3://mlflow" in mlflow and "--serve-artifacts" in mlflow
    assert '"1"' in mlflow.split("--workers", 1)[1].split("\n", 2)[1]
    assert "mlflow:5000" in mlflow  # allowed Host header for in-network clients


def test_mlflow_database_exists_on_new_and_existing_volumes():
    init_sql = Path("infra/postgres/init-databases.sql").read_text(encoding="utf-8")
    assert "CREATE DATABASE mlflow;" in init_sql
    assert "mlflow" in _service("postgres-databases")
    assert "local/mlflow" in _service("create-bucket")


def test_dagster_pools_service_sets_every_pool():
    pools = _service("dagster-pools")
    for pool in ALL_POOLS:
        assert f"concurrency set {pool} 1" in pools
    assert "spark_bronze" not in pools
    # gold and training moved into spark_lakehouse.
    assert ALL_POOLS == ("startgg_api", "spark_lakehouse")
    assert pools.count("concurrency set") == len(ALL_POOLS)


def test_app_service_reads_gold_and_mlflow():
    app = _service("app")
    assert "image: melee-app:dev" in app
    assert "LAKEKEEPER_URL: http://lakekeeper:8181/catalog" in app
    assert "MLFLOW_TRACKING_URI: http://mlflow:5000" in app
    assert 'MLFLOW_HTTP_REQUEST_MAX_RETRIES: "0"' in app
    assert 'MLFLOW_HTTP_REQUEST_TIMEOUT: "5"' in app
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    assert "data/" not in dockerfile
    assert "COPY pipeline/serving/" in dockerfile
