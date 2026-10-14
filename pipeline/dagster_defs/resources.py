"""Dagster resources: start.gg, Kafka, Spark jobs, the lakehouse catalog and MLflow."""

import json
import os
import subprocess
from datetime import datetime

import dagster as dg

from pipeline.ingest.fixtures import FixtureClient
from pipeline.ingest.kafka_publisher import KafkaPublisher
from pipeline.ingest.startgg import StartGGClient
from pipeline.lakehouse import LakehouseSettings, load_lakehouse_catalog

SUMMARY_KEYS = ("tournaments", "sets", "rejects")
SILVER_SUMMARY_KEYS = ("tables", "dropped", "bronze_rows_read", "orphan_sets", "winner_not_player")


class StartGGResource(dg.ConfigurableResource):
    """start.gg API access. With fixture_path set, replays a recorded week instead."""

    token: str
    fixture_path: str | None = None
    min_attendees: int = 50

    def get_client(self):
        if self.fixture_path:
            return FixtureClient(self.fixture_path)
        return StartGGClient(self.token)


class KafkaResource(dg.ConfigurableResource):
    bootstrap_servers: str = "kafka:9092"

    def get_publisher(self) -> KafkaPublisher:
        return KafkaPublisher(self.bootstrap_servers)


def build_bronze_command(
    job_path: str, bootstrap_servers: str, checkpoint_root: str, since: datetime | None = None
) -> list[str]:
    command = [
        "spark-submit",
        job_path,
        "--bootstrap-servers",
        bootstrap_servers,
        "--checkpoint-root",
        checkpoint_root,
    ]
    if since is not None:
        command += ["--since", since.isoformat()]
    return command


def last_json_line(stdout: str, job: str) -> dict:
    """A Spark job's JSON summary: its last non-empty stdout line."""
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{job} printed no summary")
    return json.loads(lines[-1])


def parse_job_summary(stdout: str) -> dict[str, int]:
    """Read the bronze job's JSON summary from the last non-empty stdout line."""
    summary = last_json_line(stdout, "Bronze job")
    missing = set(SUMMARY_KEYS) - summary.keys()
    if missing:
        raise ValueError(f"Bronze job summary is missing {sorted(missing)}: {json.dumps(summary)}")
    return {key: int(summary[key]) for key in SUMMARY_KEYS}


def parse_silver_summary(stdout: str) -> dict:
    summary = last_json_line(stdout, "Silver job")
    missing = set(SILVER_SUMMARY_KEYS) - summary.keys()
    if missing:
        raise ValueError(f"Silver job summary is missing {sorted(missing)}: {json.dumps(summary)}")
    return summary


class SparkJobResource(dg.ConfigurableResource):
    job_path: str = "/app/pipeline/spark_jobs/bronze_job.py"
    silver_job_path: str = "/app/pipeline/spark_jobs/silver_job.py"
    maintenance_job_path: str = "/app/pipeline/spark_jobs/maintenance_job.py"
    bootstrap_servers: str = "kafka:9092"
    checkpoint_root: str = "/checkpoints/bronze"
    silver_lock_dir: str = "/checkpoints/silver"
    timeout_seconds: int = 3600  # bronze
    silver_timeout_seconds: int = 14400
    maintenance_timeout_seconds: int = 7200

    def _submit(self, job: str, command: list[str], timeout: int) -> str:
        """Run spark-submit and return stdout; fail the step on timeout or a non-zero exit."""
        # spark-submit never needs the start.gg token, so keep it out of its environment.
        env = {k: v for k, v in os.environ.items() if k != "STARTGG_API_TOKEN"}
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, check=False, env=env, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            raise dg.Failure(description=f"{job} Spark job timed out after {timeout}s") from None
        if result.returncode != 0:
            raise dg.Failure(
                description=f"{job} Spark job exited with code {result.returncode}",
                metadata={"stderr_tail": dg.MetadataValue.text(result.stderr[-5000:])},
            )
        return result.stdout

    def run_bronze_job(self, since: datetime | None = None) -> dict[str, int]:
        """Run the bronze job, counting rows committed after `since` (default: job start)."""
        command = build_bronze_command(
            self.job_path, self.bootstrap_servers, self.checkpoint_root, since
        )
        return parse_job_summary(self._submit("Bronze", command, self.timeout_seconds))

    def run_silver_job(self) -> dict:
        command = ["spark-submit", self.silver_job_path, "--lock-dir", self.silver_lock_dir]
        return parse_silver_summary(self._submit("Silver", command, self.silver_timeout_seconds))

    def run_maintenance_job(self) -> dict:
        command = ["spark-submit", self.maintenance_job_path]
        stdout = self._submit("Maintenance", command, self.maintenance_timeout_seconds)
        return last_json_line(stdout, "Maintenance job")


class LakehouseResource(dg.ConfigurableResource):
    """The Iceberg catalog through PyIceberg. The defaults match the Compose stack."""

    catalog_type: str = "rest"
    uri: str = "http://lakekeeper:8181/catalog"
    warehouse: str = "melee"
    s3_endpoint: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None

    def get_catalog(self):
        return load_lakehouse_catalog(
            LakehouseSettings(
                catalog_type=self.catalog_type,
                uri=self.uri,
                warehouse=self.warehouse,
                s3_endpoint=self.s3_endpoint,
                s3_access_key=self.s3_access_key,
                s3_secret_key=self.s3_secret_key,
            )
        )


def lakehouse_from_env() -> LakehouseResource:
    settings = LakehouseSettings.from_env()
    return LakehouseResource(
        catalog_type=settings.catalog_type,
        uri=settings.uri,
        warehouse=settings.warehouse,
        s3_endpoint=settings.s3_endpoint,
        s3_access_key=settings.s3_access_key,
        # EnvVar keeps the secret out of the config Dagster shows in the UI.
        s3_secret_key=dg.EnvVar("S3_SECRET_KEY") if settings.s3_secret_key else None,
    )


class MlflowResource(dg.ConfigurableResource):
    tracking_uri: str = "http://mlflow:5000"

    def activate(self):
        """Point MLflow's fluent API at this server and return a client for it."""
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.tracking_uri)
        return MlflowClient(tracking_uri=self.tracking_uri, registry_uri=self.tracking_uri)
