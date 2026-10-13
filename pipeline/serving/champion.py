"""Load the MLflow champion model and its SHAP metadata for the app."""

from dataclasses import dataclass

from pipeline.ml.names import CHAMPION_ALIAS, MODEL_NAME, SHAP_ARTIFACT
from pipeline.model_features import CORE_FEATURES


class ChampionNotFound(RuntimeError):
    """No model version holds the champion alias yet."""


@dataclass(frozen=True)
class Champion:
    model: object
    version: str
    shap_metadata: dict


def load_champion(tracking_uri: str) -> Champion:
    import mlflow
    import mlflow.xgboost
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    except MlflowException as exc:
        raise ChampionNotFound(
            f"No '{CHAMPION_ALIAS}' version of {MODEL_NAME} is registered yet"
        ) from exc
    model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
    metadata = mlflow.artifacts.load_dict(f"runs:/{version.run_id}/{SHAP_ARTIFACT}")
    if metadata.get("feature_names") != CORE_FEATURES:
        raise ValueError(
            f"Champion v{version.version} was trained on {metadata.get('feature_names')}, "
            f"not {CORE_FEATURES}"
        )
    return Champion(model=model, version=str(version.version), shap_metadata=metadata)
