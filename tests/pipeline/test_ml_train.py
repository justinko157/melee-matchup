import pandas as pd
import pytest

pytest.importorskip("xgboost")

from optuna.trial import FixedTrial  # noqa: E402

from pipeline.ml.train import (  # noqa: E402
    filter_min_sets,
    quantile_split,
    shap_metadata,
    split_data,
    temporal_split,
    train_baseline,
    train_logistic,
    train_xgboost,
)
from pipeline.ml.tuning import objective  # noqa: E402
from pipeline.model_features import CORE_FEATURES  # noqa: E402
from tests.pipeline.ml_data import synthetic_gold_features  # noqa: E402


def test_temporal_split_uses_a_timestamp():
    df = synthetic_gold_features(100)
    cut = df["completed_at"].iloc[60]
    train, test = temporal_split(df, cut)
    assert (len(train), len(test)) == (60, 40)
    assert train["completed_at"].max() < test["completed_at"].min()


def test_quantile_split_keeps_time_order():
    train, test = quantile_split(synthetic_gold_features(101), 0.8)
    assert len(train) == 80
    assert train["completed_at"].max() < test["completed_at"].min()


def test_filter_min_sets_requires_both_players():
    df = synthetic_gold_features(10)
    df.loc[0, "p1_sets_played"] = 2
    df.loc[1, "p2_sets_played"] = 2
    assert len(filter_min_sets(df, 5)) == 8


def test_models_report_metrics_and_xgboost_beats_the_seed_baseline():
    train, test = split_data(synthetic_gold_features(), split_quantile=0.7)
    baseline = train_baseline(train, test)
    logistic = train_logistic(train, test)
    xgb = train_xgboost(train, test)
    for result in (baseline, logistic, xgb):
        assert 0 < result.log_loss_val < 1
        assert 0 <= result.accuracy <= 1
    assert xgb.log_loss_val < baseline.log_loss_val


def test_shap_metadata_lists_the_core_features():
    train, test = split_data(synthetic_gold_features(), split_quantile=0.7)
    model = train_xgboost(train, test).model
    metadata = shap_metadata(model, test[CORE_FEATURES])
    assert metadata["feature_names"] == CORE_FEATURES
    assert set(metadata["mean_abs_shap"]) == set(CORE_FEATURES)
    assert isinstance(metadata["base_value"], float)


def test_empty_split_raises_a_clear_error():
    df = synthetic_gold_features(50)
    with pytest.raises(ValueError, match="Empty test split: 50 train / 0 test rows"):
        split_data(df, split_at=pd.Timestamp("2030-01-01", tz="UTC"))


def test_tuning_objective_returns_auc_without_mlflow():
    train, test = split_data(synthetic_gold_features(), split_quantile=0.7)
    params = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8,
              "colsample_bytree": 0.8, "min_child_weight": 1, "reg_alpha": 1e-3,
              "reg_lambda": 1e-3}  # fmt: skip
    auc = objective(FixedTrial(params), train, test, log_to_mlflow=False)
    assert 0.5 < auc <= 1.0
