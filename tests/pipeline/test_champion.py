import pytest

pytest.importorskip("mlflow")

from mlflow import MlflowClient  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
from mlflow.protos.databricks_pb2 import (  # noqa: E402
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
)

from pipeline.ml.registry import train_and_register  # noqa: E402
from pipeline.model_features import CORE_FEATURES  # noqa: E402
from pipeline.serving.champion import ChampionNotFound, load_champion  # noqa: E402
from tests.pipeline.ml_data import synthetic_gold_features  # noqa: E402


def test_load_champion_returns_the_aliased_model(mlflow_tracking):
    features = synthetic_gold_features()
    outcome = train_and_register(features, 7, split_quantile=0.7)
    champion = load_champion(mlflow_tracking)
    assert champion.version == outcome.version
    assert champion.shap_metadata["feature_names"] == CORE_FEATURES
    assert champion.model.predict_proba(features[CORE_FEATURES].head(3)).shape == (3, 2)


def test_missing_champion_raises_champion_not_found(mlflow_tracking):
    with pytest.raises(ChampionNotFound, match="No 'champion' version of melee-matchup"):
        load_champion(mlflow_tracking)


@pytest.mark.parametrize(
    "code", [RESOURCE_DOES_NOT_EXIST, INVALID_PARAMETER_VALUE], ids=["no-model", "no-alias"]
)
def test_alias_lookup_misses_raise_champion_not_found(monkeypatch, mlflow_tracking, code):
    def missing(self, name, alias):
        raise MlflowException("not found", error_code=code)

    monkeypatch.setattr(MlflowClient, "get_model_version_by_alias", missing)
    with pytest.raises(ChampionNotFound):
        load_champion(mlflow_tracking)


def test_other_mlflow_errors_are_not_reported_as_a_missing_champion(monkeypatch, mlflow_tracking):
    def broken(self, name, alias):
        raise MlflowException("API request failed", error_code=INTERNAL_ERROR)

    monkeypatch.setattr(MlflowClient, "get_model_version_by_alias", broken)
    with pytest.raises(MlflowException) as raised:
        load_champion(mlflow_tracking)
    assert not isinstance(raised.value, ChampionNotFound)


def test_unreachable_tracking_server_fails_fast_and_is_not_a_missing_champion(monkeypatch):
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    with pytest.raises(MlflowException) as raised:
        load_champion("http://127.0.0.1:9")
    assert not isinstance(raised.value, ChampionNotFound)
