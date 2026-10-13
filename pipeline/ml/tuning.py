"""Optuna hyperparameter tuning for XGBoost with MLflow logging (moved from src/tuning.py).

Usage (inside the stack): python -m pipeline.ml.tuning --n-trials 50
"""

import argparse
import logging

import mlflow
import optuna
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from xgboost import XGBClassifier

from pipeline.lakehouse import load_lakehouse_catalog
from pipeline.ml.train import DEFAULT_MIN_SETS, load_features, split_data
from pipeline.model_features import CORE_FEATURES, TARGET

logger = logging.getLogger(__name__)


def objective(trial: optuna.Trial, train, test, log_to_mlflow: bool = True) -> float:
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
    model = XGBClassifier(
        **params, eval_metric="logloss", early_stopping_rounds=20, random_state=42
    )
    model.fit(
        train[CORE_FEATURES],
        train[TARGET],
        eval_set=[(test[CORE_FEATURES], test[TARGET])],
        verbose=False,
    )
    y_prob = model.predict_proba(test[CORE_FEATURES])[:, 1]
    auc = roc_auc_score(test[TARGET], y_prob)
    if log_to_mlflow:
        mlflow.set_experiment("melee-matchup-tuning")
        with mlflow.start_run(run_name=f"trial-{trial.number}"):
            mlflow.log_params(params)
            mlflow.log_metrics({"roc_auc": auc, "log_loss": log_loss(test[TARGET], y_prob)})
    return auc


def tune(
    df: pd.DataFrame,
    n_trials: int = 50,
    min_sets: int = DEFAULT_MIN_SETS,
    log_to_mlflow: bool = True,
) -> optuna.Study:
    train, test = split_data(df, min_sets=min_sets)
    study = optuna.create_study(direction="maximize", study_name="xgboost-melee")
    study.optimize(lambda trial: objective(trial, train, test, log_to_mlflow), n_trials=n_trials)
    logger.info(f"Best trial #{study.best_trial.number}: AUC={study.best_value:.4f}")
    return study


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Tune XGBoost hyperparams")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--min-sets", type=int, default=DEFAULT_MIN_SETS)
    args = parser.parse_args()
    features, _ = load_features(load_lakehouse_catalog())
    study = tune(features, n_trials=args.n_trials, min_sets=args.min_sets)
    print(f"\nBest AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
