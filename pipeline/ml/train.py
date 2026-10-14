"""Model training on gold.set_features (moved from src/model.py).

Models:
1. Baseline: Logistic regression on seed differential only
2. Full LR: Logistic regression on all features
3. XGBoost: Gradient boosting on all features

Uses a temporal train/test split (no random shuffle) to avoid leakage.
Run directly to print the comparison table without logging to MLflow.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import shap
from pyiceberg.catalog import Catalog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from pipeline.gold.tables import SET_FEATURES
from pipeline.lakehouse import current_snapshot_id, load_lakehouse_catalog, read_table
from pipeline.model_features import CORE_FEATURES, SEED_FEATURES, TARGET

logger = logging.getLogger(__name__)

# Train on everything before this time, test on the rest (the last ~9 months).
DEFAULT_SPLIT_AT = pd.Timestamp("2025-07-01", tz="UTC")
DEFAULT_MIN_SETS = 5
# The latest slice of the training window, by time, that picks XGBoost's tree count.
VALIDATION_FRACTION = 0.1
XGB_MAX_TREES = 1000  # a ceiling: early stopping picks the count (334 on 2018-2026 gold)
XGB_EARLY_STOPPING_ROUNDS = 20
XGB_PARAMS = {
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "eval_metric": "logloss",
    "random_state": 42,
}


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


# What splitting, the min_sets filter and the models use (SEED_FEATURES and the
# sets-played columns are in CORE_FEATURES); reading only these keeps training memory down.
TRAINING_COLUMNS = tuple(
    dict.fromkeys(["set_id", "completed_at", *CORE_FEATURES, *SEED_FEATURES, TARGET])
)


def load_features(catalog: Catalog) -> tuple[pd.DataFrame, int | None]:
    """The training columns of gold.set_features in time order, plus the snapshot read."""
    snapshot = current_snapshot_id(catalog, SET_FEATURES)
    df = read_table(catalog, SET_FEATURES, TRAINING_COLUMNS, snapshot_id=snapshot)
    return df.sort_values(["completed_at", "set_id"]).reset_index(drop=True), snapshot


def filter_min_sets(df: pd.DataFrame, min_sets: int) -> pd.DataFrame:
    """Keep sets where both players have at least min_sets prior sets."""
    kept = df[(df["p1_sets_played"] >= min_sets) & (df["p2_sets_played"] >= min_sets)]
    logger.info(f"{len(df):,} rows, {len(kept):,} after min_sets={min_sets} filter")
    return kept


def temporal_split(df: pd.DataFrame, split_at: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data temporally — no random shuffle."""
    return df[df["completed_at"] < split_at].copy(), df[df["completed_at"] >= split_at].copy()


def quantile_split(df: pd.DataFrame, q: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal split at the q-th quantile of completed_at (for small datasets)."""
    return temporal_split(df, df["completed_at"].quantile(q))


def split_data(
    df: pd.DataFrame,
    *,
    split_at: pd.Timestamp | None = None,
    split_quantile: float | None = None,
    min_sets: int = DEFAULT_MIN_SETS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = filter_min_sets(df, min_sets)
    if split_quantile is not None:
        train, test = quantile_split(data, split_quantile)
    else:
        train, test = temporal_split(data, DEFAULT_SPLIT_AT if split_at is None else split_at)
    if train.empty or test.empty:
        side = "train" if train.empty else "test"
        raise ValueError(
            f"Empty {side} split: {len(train)} train / {len(test)} test rows after "
            f"min_sets={min_sets}. Choose a different split_at or split_quantile."
        )
    logger.info(f"Train: {len(train):,} sets, Test: {len(test):,} sets")
    return train, test


def validation_split(
    train: pd.DataFrame, fraction: float = VALIDATION_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the latest fraction of the training window (by time) for early stopping."""
    fit, valid = quantile_split(train, 1 - fraction)
    if fit.empty or valid.empty:
        raise ValueError(
            f"Empty validation split: {len(fit)} fit / {len(valid)} validation rows from "
            f"{len(train)} training rows."
        )
    return fit, valid


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


def train_baseline(train: pd.DataFrame, test: pd.DataFrame) -> ModelResult:
    """Baseline: logistic regression on seed differential only."""
    features = SEED_FEATURES
    model = LogisticRegression(max_iter=1000)
    model.fit(train[features], train[TARGET])
    return evaluate("Baseline (Seed Only)", model, test[features], test[TARGET], features)


class ScaledModel:
    """Logistic regression behind a StandardScaler, with a predict_proba interface."""

    def __init__(self, scaler, model):
        self._scaler = scaler
        self._model = model
        self.coef_ = model.coef_

    def predict_proba(self, X):
        return self._model.predict_proba(self._scaler.transform(X))


def train_logistic(train: pd.DataFrame, test: pd.DataFrame) -> ModelResult:
    """Full logistic regression on all core features."""
    features = CORE_FEATURES
    scaler = StandardScaler()
    model = LogisticRegression(max_iter=1000)
    model.fit(scaler.fit_transform(train[features]), train[TARGET])
    return evaluate(
        "Logistic Regression (All Features)",
        ScaledModel(scaler, model),
        test[features],
        test[TARGET],
        features,
    )


def train_xgboost(train: pd.DataFrame, test: pd.DataFrame) -> ModelResult:
    """XGBoost gradient boosting on all core features.

    The tree count is early-stopped on the latest slice of train, then the model is
    refit on all of train with that count. The test split is only scored.
    """
    features = CORE_FEATURES
    fit, valid = validation_split(train)
    probe = XGBClassifier(
        n_estimators=XGB_MAX_TREES, early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS, **XGB_PARAMS
    )
    probe.fit(
        fit[features], fit[TARGET], eval_set=[(valid[features], valid[TARGET])], verbose=False
    )
    model = XGBClassifier(n_estimators=probe.best_iteration + 1, **XGB_PARAMS)
    model.fit(train[features], train[TARGET], verbose=False)
    return evaluate("XGBoost", model, test[features], test[TARGET], features)


def shap_metadata(model, X_sample: pd.DataFrame) -> dict:
    """SHAP base value and mean absolute SHAP per feature, for the app's explanations."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    mean_abs = np.abs(shap_values.values).mean(axis=0).tolist()
    return {
        "base_value": float(explainer.expected_value),
        "feature_names": list(X_sample.columns),
        "mean_abs_shap": dict(zip(X_sample.columns, mean_abs, strict=True)),
    }


def walk_forward_cv(
    df: pd.DataFrame, min_sets: int = DEFAULT_MIN_SETS, n_folds: int = 4
) -> pd.DataFrame:
    """Walk-forward time-series cross-validation for XGBoost.

    Splits data into n_folds+1 temporal chunks. For each fold k (1..n_folds),
    trains on chunks 0..k-1 and tests on chunk k.
    """
    data = filter_min_sets(df, min_sets)
    quantiles = [i / (n_folds + 1) for i in range(1, n_folds + 1)]
    cuts = data["completed_at"].quantile(quantiles).tolist()
    cuts.append(data["completed_at"].max() + pd.Timedelta(seconds=1))

    fold_results = []
    for fold_idx, split_at in enumerate(cuts):
        train = data[data["completed_at"] < split_at]
        if fold_idx < len(cuts) - 1:
            in_fold = (data["completed_at"] >= split_at) & (
                data["completed_at"] < cuts[fold_idx + 1]
            )
            test = data[in_fold]
        else:
            test = data[data["completed_at"] >= split_at]
        if len(test) == 0:
            continue
        result = train_xgboost(train, test)
        fold_results.append(
            {
                "Fold": fold_idx + 1,
                "Train Size": len(train),
                "Test Size": len(test),
                "Accuracy": result.accuracy,
                "Log Loss": result.log_loss_val,
                "Brier Score": result.brier_score,
                "ROC AUC": result.roc_auc,
            }
        )

    results_df = pd.DataFrame(fold_results)
    mean_row = results_df[["Accuracy", "Log Loss", "Brier Score", "ROC AUC"]].mean()
    mean_row["Fold"] = "Mean"
    mean_row["Train Size"] = ""
    mean_row["Test Size"] = ""
    return pd.concat([results_df, pd.DataFrame([mean_row])], ignore_index=True)


def results_table(results: list[ModelResult]) -> pd.DataFrame:
    """Format results as a comparison table."""
    return pd.DataFrame(
        [
            {
                "Model": r.name,
                "Accuracy": f"{r.accuracy:.4f}",
                "Log Loss": f"{r.log_loss_val:.4f}",
                "Brier Score": f"{r.brier_score:.4f}",
                "ROC AUC": f"{r.roc_auc:.4f}",
            }
            for r in results
        ]
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    df, _ = load_features(load_lakehouse_catalog())
    train, test = split_data(df)
    results = [train_baseline(train, test), train_logistic(train, test), train_xgboost(train, test)]
    print("\n" + results_table(results).to_string(index=False))


if __name__ == "__main__":
    main()
