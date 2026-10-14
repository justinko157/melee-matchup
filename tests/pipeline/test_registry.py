import pytest

mlflow = pytest.importorskip("mlflow")

from mlflow import MlflowClient  # noqa: E402

from pipeline.ml.names import (  # noqa: E402
    CHAMPION_ALIAS,
    GOLD_SNAPSHOT_TAG,
    MODEL_NAME,
    SHAP_ARTIFACT,
)
from pipeline.ml.promotion import PromotionDecision  # noqa: E402
from pipeline.ml.registry import champion_gold_snapshot, train_and_register  # noqa: E402
from pipeline.model_features import CORE_FEATURES  # noqa: E402
from tests.pipeline.ml_data import synthetic_gold_features  # noqa: E402


def test_first_model_becomes_champion(mlflow_tracking):
    outcome = train_and_register(synthetic_gold_features(), 111, split_quantile=0.7)
    assert outcome.decision == PromotionDecision(True, "no_champion")
    assert outcome.champion_version is None
    champion = MlflowClient().get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    assert str(champion.version) == outcome.version


def test_version_carries_the_gold_snapshot_and_shap_artifact(mlflow_tracking):
    outcome = train_and_register(synthetic_gold_features(), 222, split_quantile=0.7)
    client = MlflowClient()
    assert client.get_model_version(MODEL_NAME, outcome.version).tags[GOLD_SNAPSHOT_TAG] == "222"
    run = client.get_run(outcome.run_id)
    assert run.data.tags[GOLD_SNAPSHOT_TAG] == "222"
    assert {"xgboost_log_loss", "baseline_log_loss", "logistic_roc_auc"} <= set(run.data.metrics)
    shap = mlflow.artifacts.load_dict(f"runs:/{outcome.run_id}/{SHAP_ARTIFACT}")
    assert shap["feature_names"] == CORE_FEATURES


def test_second_model_is_compared_with_the_champion(mlflow_tracking):
    first = train_and_register(synthetic_gold_features(), 1, split_quantile=0.7)
    second = train_and_register(synthetic_gold_features(), 2, split_quantile=0.7)
    assert second.champion_version == first.version
    assert second.champion_log_loss == pytest.approx(second.xgboost.log_loss_val)  # same data
    assert second.decision == PromotionDecision(True, "beats_or_ties_champion")
    assert champion_gold_snapshot(MlflowClient()) == "2"


def test_rejected_model_leaves_the_champion_alone(mlflow_tracking, monkeypatch):
    first = train_and_register(synthetic_gold_features(), 1, split_quantile=0.7)
    monkeypatch.setattr(
        "pipeline.ml.registry.decide_promotion",
        lambda *args: PromotionDecision(False, "worse_than_champion"),
    )
    second = train_and_register(synthetic_gold_features(), 2, split_quantile=0.7)
    client = MlflowClient()
    champion = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    assert str(champion.version) == first.version
    tags = client.get_model_version(MODEL_NAME, second.version).tags
    assert tags["promotion"] == "worse_than_champion"
    assert champion_gold_snapshot(client) == "1"
