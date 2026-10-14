"""Dagster definitions for the pipeline."""

import os

import dagster as dg

from pipeline.dagster_defs.assets import bronze_startgg, startgg_raw
from pipeline.dagster_defs.automation import (
    bronze_job,
    ingest_job,
    lakehouse_automation,
    land_bronze_after_ingest,
    refresh_lakehouse_job,
    refresh_recent_weeks,
)
from pipeline.dagster_defs.lakehouse_assets import gold_startgg, silver_startgg
from pipeline.dagster_defs.maintenance import lakehouse_maintenance, weekly_maintenance
from pipeline.dagster_defs.resources import (
    KafkaResource,
    MlflowResource,
    SparkJobResource,
    StartGGResource,
    lakehouse_from_env,
)
from pipeline.dagster_defs.training import daily_training, matchup_model, train_job

defs = dg.Definitions(
    assets=[startgg_raw, bronze_startgg, silver_startgg, gold_startgg, matchup_model],
    jobs=[ingest_job, bronze_job, refresh_lakehouse_job, train_job, lakehouse_maintenance],
    sensors=[land_bronze_after_ingest, lakehouse_automation],
    schedules=[refresh_recent_weeks, daily_training, weekly_maintenance],
    resources={
        # EnvVar keeps the token out of the config Dagster shows in the UI.
        "startgg": StartGGResource(
            token=dg.EnvVar("STARTGG_API_TOKEN"),
            fixture_path=os.getenv("STARTGG_FIXTURE_PATH") or None,
        ),
        "kafka": KafkaResource(),
        "spark_job": SparkJobResource(),
        "lakehouse": lakehouse_from_env(),
        "mlflow_tracking": MlflowResource(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        ),
    },
)
