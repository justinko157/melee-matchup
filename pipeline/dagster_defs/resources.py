"""Dagster resources wrapping the start.gg client, Kafka and the Spark job."""

import json
import subprocess
from datetime import datetime

import dagster as dg

from pipeline.ingest.fixtures import FixtureClient
from pipeline.ingest.kafka_publisher import KafkaPublisher
from pipeline.ingest.startgg import StartGGClient

SUMMARY_KEYS = ("tournaments", "sets", "rejects")


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


def parse_job_summary(stdout: str) -> dict[str, int]:
    """Read the bronze job's JSON summary from the last non-empty stdout line."""
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Bronze job printed no summary")
    summary = json.loads(lines[-1])
    missing = set(SUMMARY_KEYS) - summary.keys()
    if missing:
        raise ValueError(f"Bronze job summary is missing {sorted(missing)}: {lines[-1]}")
    return {key: int(summary[key]) for key in SUMMARY_KEYS}


class SparkJobResource(dg.ConfigurableResource):
    job_path: str = "/app/pipeline/spark_jobs/bronze_job.py"
    bootstrap_servers: str = "kafka:9092"
    checkpoint_root: str = "/checkpoints/bronze"

    def run_bronze_job(self, since: datetime | None = None) -> dict[str, int]:
        """Run the bronze job, counting rows committed after `since` (default: job start)."""
        command = build_bronze_command(
            self.job_path, self.bootstrap_servers, self.checkpoint_root, since
        )
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise dg.Failure(
                description=f"Bronze Spark job exited with code {result.returncode}",
                metadata={"stderr_tail": dg.MetadataValue.text(result.stderr[-5000:])},
            )
        return parse_job_summary(result.stdout)
