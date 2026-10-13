import pytest

pytest.importorskip("mlflow")

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
