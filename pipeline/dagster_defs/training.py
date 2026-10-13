"""Daily model training on gold, registered in MLflow with a gated champion alias."""

import dagster as dg
import pandas as pd

from pipeline.dagster_defs.lakehouse_assets import gold_startgg
from pipeline.dagster_defs.pools import SPARK_LAKEHOUSE
from pipeline.dagster_defs.resources import LakehouseResource, MlflowResource
from pipeline.gold.tables import SET_FEATURES
from pipeline.lakehouse import current_snapshot_id


class TrainingConfig(dg.Config):
    split_at: str = "2025-07-01T00:00:00+00:00"
    # Set for small datasets (the smoke test): split at this quantile of completed_at.
    split_quantile: float | None = None
    min_sets: int = 5


@dg.asset(
    deps=[gold_startgg],
    pool=SPARK_LAKEHOUSE,
    group_name="ml",
    kinds={"xgboost", "mlflow"},
    check_specs=[dg.AssetCheckSpec("model_beats_baseline", asset="matchup_model")],
)
def matchup_model(
    config: TrainingConfig, lakehouse: LakehouseResource, mlflow_tracking: MlflowResource
) -> dg.MaterializeResult:
    """XGBoost trained on gold.set_features, registered as a new MLflow model version."""
    from pipeline.ml.registry import train_and_register
    from pipeline.ml.train import load_features

    mlflow_tracking.activate()
    features, snapshot = load_features(lakehouse.get_catalog())
    outcome = train_and_register(
        features,
        snapshot,
        split_at=pd.Timestamp(config.split_at),
        split_quantile=config.split_quantile,
        min_sets=config.min_sets,
    )
    xgb, baseline = outcome.xgboost, outcome.baseline
    beats_baseline = bool(xgb.log_loss_val < baseline.log_loss_val)
    return dg.MaterializeResult(
        metadata={
            "model_version": outcome.version,
            "mlflow_run_id": outcome.run_id,
            "gold_snapshot_id": "" if snapshot is None else str(snapshot),
            "promotion_decision": outcome.decision.reason,
            "promoted": outcome.decision.promote,
            "xgboost_log_loss": xgb.log_loss_val,
            "baseline_log_loss": baseline.log_loss_val,
            "champion_log_loss": "none"
            if outcome.champion_log_loss is None
            else outcome.champion_log_loss,
            "champion_version_before": outcome.champion_version or "none",
            "xgboost_accuracy": xgb.accuracy,
            "xgboost_roc_auc": xgb.roc_auc,
        },
        check_results=[
            dg.AssetCheckResult(
                check_name="model_beats_baseline",
                passed=beats_baseline,
                metadata={
                    "xgboost_log_loss": xgb.log_loss_val,
                    "baseline_log_loss": baseline.log_loss_val,
                },
            )
        ],
    )


train_job = dg.define_asset_job("train_matchup_model", selection=[matchup_model])


def should_train(gold_snapshot: int | None, champion_snapshot: str | None) -> str | None:
    """A reason to skip today's training, or None to train."""
    if gold_snapshot is None:
        return "gold.set_features has no data yet"
    # A champion trained without a gold snapshot is tagged "": treat it like no tag.
    if champion_snapshot and champion_snapshot == str(gold_snapshot):
        return f"The champion was already trained on gold snapshot {gold_snapshot}"
    return None


@dg.schedule(
    job=train_job,
    cron_schedule="0 7 * * *",
    execution_timezone="America/Los_Angeles",
    default_status=dg.DefaultScheduleStatus.RUNNING,
)
def daily_training(
    context: dg.ScheduleEvaluationContext,
    lakehouse: LakehouseResource,
    mlflow_tracking: MlflowResource,
):
    """Train when gold has changed since the champion was trained."""
    from pipeline.ml.registry import champion_gold_snapshot

    catalog = lakehouse.get_catalog()
    gold = (
        current_snapshot_id(catalog, SET_FEATURES) if catalog.table_exists(SET_FEATURES) else None
    )
    skip = should_train(gold, champion_gold_snapshot(mlflow_tracking.activate()))
    if skip:
        return dg.SkipReason(skip)
    return dg.RunRequest(run_key=f"gold-{gold}")
