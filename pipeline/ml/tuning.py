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
from pipeline.ml.train import DEFAULT_MIN_SETS, load_features, split_data, validation_split
from pipeline.model_features import CORE_FEATURES, TARGET

logger = logging.getLogger(__name__)


def objective(trial: optuna.Trial, fit, valid, log_to_mlflow: bool = True) -> float:
    """Single Optuna trial: train XGBoost on fit with sampled hyperparams, score on valid."""
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
        fit[CORE_FEATURES],
        fit[TARGET],
        eval_set=[(valid[CORE_FEATURES], valid[TARGET])],
        verbose=False,
    )
    y_prob = model.predict_proba(valid[CORE_FEATURES])[:, 1]
    auc = roc_auc_score(valid[TARGET], y_prob)
    if log_to_mlflow:
        mlflow.set_experiment("melee-matchup-tuning")
        with mlflow.start_run(run_name=f"trial-{trial.number}"):
            mlflow.log_params(params)
            mlflow.log_metrics({"roc_auc": auc, "log_loss": log_loss(valid[TARGET], y_prob)})
    return auc


def tune(
    df: pd.DataFrame,
    n_trials: int = 50,
    min_sets: int = DEFAULT_MIN_SETS,
    log_to_mlflow: bool = True,
) -> optuna.Study:
    # Tune inside the training window; the test split stays unseen.
    train, _ = split_data(df, min_sets=min_sets)
    fit, valid = validation_split(train)
    study = optuna.create_study(direction="maximize", study_name="xgboost-melee")
    study.optimize(lambda trial: objective(trial, fit, valid, log_to_mlflow), n_trials=n_trials)
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
