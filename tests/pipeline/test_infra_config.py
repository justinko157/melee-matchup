"""Keep infra config consistent with the Python code."""

import re
import tomllib
from pathlib import Path

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
        if image == "melee-pipeline:dev":
            continue
        assert ":" in image and not image.endswith((":latest", ":latest-main")), image
