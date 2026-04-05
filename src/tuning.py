"""Optuna hyperparameter tuning for XGBoost with MLflow logging.

Searches the XGBoost hyperparameter space using Bayesian optimization
and logs every trial to MLflow for comparison.

Usage:
    python -m src.tuning                    # 50 trials (default)
    python -m src.tuning --n-trials 100     # more trials
"""

import argparse
import logging
from pathlib import Path

import mlflow
import optuna
from sklearn.metrics import log_loss, roc_auc_score
from xgboost import XGBClassifier

from .model import CORE_FEATURES, TARGET, load_data, temporal_split

logger = logging.getLogger(__name__)


def objective(trial: optuna.Trial, train, test) -> float:
    """Single Optuna trial — train XGBoost with sampled hyperparams."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    X_train, y_train = train[CORE_FEATURES], train[TARGET]
    X_test, y_test = test[CORE_FEATURES], test[TARGET]

    model = XGBClassifier(
        **params,
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    ll = log_loss(y_test, y_prob)

    # Log to MLflow
    mlflow.set_experiment("melee-matchup-tuning")
    with mlflow.start_run(run_name=f"trial-{trial.number}"):
        mlflow.log_params(params)
        mlflow.log_metrics({"roc_auc": auc, "log_loss": ll})

    return auc


def tune(
    data_path: Path | str | None = None,
    n_trials: int = 50,
    min_sets: int = 5,
) -> optuna.Study:
    """Run Optuna hyperparameter search.

    Args:
        data_path: Path to features parquet (uses default if None).
        n_trials: Number of Optuna trials to run.
        min_sets: Minimum sets played filter.

    Returns:
        The completed Optuna study.
    """
    kwargs = {"min_sets": min_sets}
    if data_path is not None:
        kwargs["path"] = data_path
    df = load_data(**kwargs)
    train, test = temporal_split(df)

    study = optuna.create_study(
        direction="maximize",
        study_name="xgboost-melee",
    )
    study.optimize(
        lambda trial: objective(trial, train, test),
        n_trials=n_trials,
    )

    best = study.best_trial
    logger.info(f"Best trial #{best.number}: AUC={best.value:.4f}")
    logger.info(f"Best params: {best.params}")

    return study


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Tune XGBoost hyperparams")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--min-sets", type=int, default=5)
    args = parser.parse_args()

    study = tune(n_trials=args.n_trials, min_sets=args.min_sets)

    print(f"\nBest AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
