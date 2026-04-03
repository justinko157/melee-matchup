"""Modeling pipeline for Melee set prediction.

Models:
1. Baseline: Logistic regression on seed differential only
2. Full LR: Logistic regression on all features
3. XGBoost: Gradient boosting on all features

Uses a temporal train/test split (no random shuffle) to avoid leakage.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

FEATURES_PATH = Path(__file__).parent.parent / "data" / "processed" / "features.parquet"

# Feature groups
SEED_FEATURES = ["seed_diff"]
CORE_FEATURES = [
    "elo_diff",
    "p1_expected",
    "p1_sets_played",
    "p2_sets_played",
    "recent_wr_diff",
    "p1_h2h_wr",
    "h2h_total",
    "seed_diff",
    "num_attendees",
]

TARGET = "p1_won"

# Temporal split: train on everything before this timestamp, test on the rest.
# Default: 2025-07-01 (last ~9 months as test set)
DEFAULT_SPLIT_TIMESTAMP = 1751328000  # 2025-07-01 UTC


@dataclass
class ModelResult:
    """Stores evaluation results for a single model."""

    name: str
    accuracy: float
    log_loss_val: float
    brier_score: float
    roc_auc: float
    y_true: np.ndarray
    y_prob: np.ndarray
    model: object
    feature_names: list[str] = field(default_factory=list)
    feature_importances: np.ndarray | None = None


def load_data(
    path: Path | str = FEATURES_PATH,
    min_sets: int = 5,
) -> pd.DataFrame:
    """Load feature data and apply minimum-activity filter.

    Args:
        path: Path to the parquet file.
        min_sets: Minimum sets played by BOTH players for a row to be included.
                  Filters out sets where Elo/form features are unreliable.
    """
    df = pd.read_parquet(path)
    before = len(df)
    df = df[(df["p1_sets_played"] >= min_sets) & (df["p2_sets_played"] >= min_sets)]
    logger.info(f"Loaded {before:,} rows, {len(df):,} after min_sets={min_sets} filter")
    return df


def temporal_split(
    df: pd.DataFrame,
    split_ts: int = DEFAULT_SPLIT_TIMESTAMP,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data temporally — no random shuffle."""
    train = df[df["completed_at"] < split_ts].copy()
    test = df[df["completed_at"] >= split_ts].copy()
    logger.info(f"Train: {len(train):,} sets, Test: {len(test):,} sets")
    return train, test


def evaluate(
    name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
) -> ModelResult:
    """Evaluate a model and return structured results."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_prob)
    bs = brier_score_loss(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    # Feature importances
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])

    logger.info(f"[{name}] Acc={acc:.4f}  LogLoss={ll:.4f}  Brier={bs:.4f}  AUC={auc:.4f}")

    return ModelResult(
        name=name,
        accuracy=acc,
        log_loss_val=ll,
        brier_score=bs,
        roc_auc=auc,
        y_true=y_test.values,
        y_prob=y_prob,
        model=model,
        feature_names=feature_names,
        feature_importances=importances,
    )


def train_baseline(
    train: pd.DataFrame, test: pd.DataFrame
) -> ModelResult:
    """Baseline: logistic regression on seed differential only."""
    features = SEED_FEATURES
    X_train, y_train = train[features], train[TARGET]
    X_test, y_test = test[features], test[TARGET]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return evaluate("Baseline (Seed Only)", model, X_test, y_test, features)


def train_logistic(
    train: pd.DataFrame, test: pd.DataFrame
) -> ModelResult:
    """Full logistic regression on all core features."""
    features = CORE_FEATURES
    X_train, y_train = train[features], train[TARGET]
    X_test, y_test = test[features], test[TARGET]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Wrap so evaluate() can call predict_proba on scaled data
    class ScaledModel:
        def __init__(self, scaler, model):
            self._scaler = scaler
            self._model = model
            self.coef_ = model.coef_

        def predict_proba(self, X):
            return self._model.predict_proba(self._scaler.transform(X))

    wrapped = ScaledModel(scaler, model)
    return evaluate("Logistic Regression (All Features)", wrapped, X_test, y_test, features)


def train_xgboost(
    train: pd.DataFrame, test: pd.DataFrame
) -> ModelResult:
    """XGBoost gradient boosting on all core features."""
    features = CORE_FEATURES
    X_train, y_train = train[features], train[TARGET]
    X_test, y_test = test[features], test[TARGET]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    return evaluate("XGBoost", model, X_test, y_test, features)


def run_all(
    data_path: Path | str = FEATURES_PATH,
    min_sets: int = 5,
    split_ts: int = DEFAULT_SPLIT_TIMESTAMP,
    log_to_mlflow: bool = True,
) -> list[ModelResult]:
    """Train and evaluate all models.

    Args:
        data_path: Path to the features parquet file.
        min_sets: Minimum sets for both players to be included.
        split_ts: Unix timestamp for temporal train/test split.
        log_to_mlflow: Whether to log to MLflow.

    Returns:
        List of ModelResult objects.
    """
    df = load_data(data_path, min_sets=min_sets)
    train, test = temporal_split(df, split_ts)

    results = []

    trainers = [
        ("baseline", train_baseline),
        ("logistic", train_logistic),
        ("xgboost", train_xgboost),
    ]

    for run_name, trainer_fn in trainers:
        if log_to_mlflow:
            mlflow.set_experiment("melee-matchup")
            with mlflow.start_run(run_name=run_name):
                result = trainer_fn(train, test)
                mlflow.log_params({
                    "model": result.name,
                    "min_sets": min_sets,
                    "split_ts": split_ts,
                    "train_size": len(train),
                    "test_size": len(test),
                    "features": result.feature_names,
                })
                mlflow.log_metrics({
                    "accuracy": result.accuracy,
                    "log_loss": result.log_loss_val,
                    "brier_score": result.brier_score,
                    "roc_auc": result.roc_auc,
                })
        else:
            result = trainer_fn(train, test)

        results.append(result)

    return results


def results_table(results: list[ModelResult]) -> pd.DataFrame:
    """Format results as a comparison table."""
    rows = []
    for r in results:
        rows.append({
            "Model": r.name,
            "Accuracy": f"{r.accuracy:.4f}",
            "Log Loss": f"{r.log_loss_val:.4f}",
            "Brier Score": f"{r.brier_score:.4f}",
            "ROC AUC": f"{r.roc_auc:.4f}",
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    results = run_all(log_to_mlflow=False)
    print("\n" + results_table(results).to_string(index=False))
