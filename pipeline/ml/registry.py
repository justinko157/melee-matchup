"""Train, log to MLflow, register a version, and decide whether it becomes champion."""

from dataclasses import dataclass

import mlflow
import mlflow.xgboost
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from sklearn.metrics import log_loss

from pipeline.ml.names import (
    CHAMPION_ALIAS,
    EXPERIMENT,
    GOLD_SNAPSHOT_TAG,
    MODEL_NAME,
    SHAP_ARTIFACT,
)
from pipeline.ml.promotion import PromotionDecision, decide_promotion
from pipeline.ml.train import (
    DEFAULT_MIN_SETS,
    ModelResult,
    shap_metadata,
    split_data,
    train_baseline,
    train_logistic,
    train_xgboost,
)
from pipeline.model_features import CORE_FEATURES, TARGET

SHAP_SAMPLE_ROWS = 1000


@dataclass
class TrainingOutcome:
    baseline: ModelResult
    logistic: ModelResult
    xgboost: ModelResult
    version: str
    run_id: str
    decision: PromotionDecision
    champion_version: str | None
    champion_log_loss: float | None


def champion_version(client: MlflowClient) -> ModelVersion | None:
    try:
        return client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    except MlflowException:  # no registered model or no alias yet
        return None


def champion_gold_snapshot(client: MlflowClient) -> str | None:
    """The gold snapshot id the current champion was trained on."""
    champion = champion_version(client)
    return None if champion is None else champion.tags.get(GOLD_SNAPSHOT_TAG)


def _log_result(prefix: str, result: ModelResult) -> None:
    mlflow.log_metrics(
        {
            f"{prefix}_accuracy": result.accuracy,
            f"{prefix}_log_loss": result.log_loss_val,
            f"{prefix}_brier": result.brier_score,
            f"{prefix}_roc_auc": result.roc_auc,
        }
    )


def train_and_register(
    features: pd.DataFrame,
    gold_snapshot_id: int | None,
    *,
    split_at: pd.Timestamp | None = None,
    split_quantile: float | None = None,
    min_sets: int = DEFAULT_MIN_SETS,
) -> TrainingOutcome:
    train, test = split_data(
        features, split_at=split_at, split_quantile=split_quantile, min_sets=min_sets
    )
    snapshot_tag = "" if gold_snapshot_id is None else str(gold_snapshot_id)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="train") as run:
        baseline = train_baseline(train, test)
        logistic = train_logistic(train, test)
        xgb = train_xgboost(train, test)
        mlflow.log_params(
            {
                "min_sets": min_sets,
                "split_at": "" if split_at is None else split_at.isoformat(),
                "split_quantile": "" if split_quantile is None else split_quantile,
                "train_size": len(train),
                "test_size": len(test),
                "features": ",".join(CORE_FEATURES),
            }
        )
        for prefix, result in (("baseline", baseline), ("logistic", logistic), ("xgboost", xgb)):
            _log_result(prefix, result)
        mlflow.set_tag(GOLD_SNAPSHOT_TAG, snapshot_tag)
        sample = test[CORE_FEATURES].head(SHAP_SAMPLE_ROWS)
        mlflow.log_dict(shap_metadata(xgb.model, sample), SHAP_ARTIFACT)
        info = mlflow.xgboost.log_model(xgb.model, name="model", registered_model_name=MODEL_NAME)

    client = MlflowClient()
    version = str(info.registered_model_version)
    client.set_model_version_tag(MODEL_NAME, version, GOLD_SNAPSHOT_TAG, snapshot_tag)

    champion = champion_version(client)
    champion_ll = None
    if champion is not None:
        model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
        champion_prob = model.predict_proba(test[CORE_FEATURES])[:, 1]
        champion_ll = float(log_loss(test[TARGET], champion_prob))

    decision = decide_promotion(xgb.log_loss_val, baseline.log_loss_val, champion_ll)
    client.set_model_version_tag(MODEL_NAME, version, "promotion", decision.reason)
    if decision.promote:
        client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version)

    return TrainingOutcome(
        baseline=baseline,
        logistic=logistic,
        xgboost=xgb,
        version=version,
        run_id=run.info.run_id,
        decision=decision,
        champion_version=None if champion is None else str(champion.version),
        champion_log_loss=champion_ll,
    )
