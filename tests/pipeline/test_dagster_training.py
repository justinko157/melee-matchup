import dagster as dg
import pytest

pytest.importorskip("mlflow")

from pipeline.dagster_defs.resources import LakehouseResource, MlflowResource  # noqa: E402
from pipeline.dagster_defs.training import (  # noqa: E402
    daily_training,
    matchup_model,
    should_train,
)
from pipeline.gold.tables import SET_FEATURES, SET_FEATURES_SCHEMA  # noqa: E402
from pipeline.gold.writer import replace_table, to_arrow  # noqa: E402
from pipeline.lakehouse import current_snapshot_id  # noqa: E402
from tests.pipeline.ml_data import synthetic_gold_features  # noqa: E402


@pytest.mark.parametrize(
    ("gold", "champion", "expected"),
    [
        (None, None, "gold.set_features has no data yet"),
        (5, "5", "The champion was already trained on gold snapshot 5"),
        (6, "5", None),
        # train_and_register tags a missing snapshot as "": no usable champion snapshot.
        (6, "", None),
    ],
)
def test_should_train(gold, champion, expected):
    assert should_train(gold, champion) == expected


@pytest.fixture
def resources(sql_settings, sql_catalog, mlflow_tracking):
    replace_table(
        sql_catalog, SET_FEATURES, to_arrow(synthetic_gold_features(), SET_FEATURES_SCHEMA), {}
    )
    return {
        "lakehouse": LakehouseResource(
            catalog_type="sql", uri=sql_settings.uri, warehouse=sql_settings.warehouse
        ),
        "mlflow_tracking": MlflowResource(tracking_uri=mlflow_tracking),
    }


RUN_CONFIG = {"ops": {"matchup_model": {"config": {"split_quantile": 0.7}}}}


def test_matchup_model_registers_and_reports_promotion(resources, sql_catalog):
    result = dg.materialize([matchup_model], resources=resources, run_config=RUN_CONFIG)
    (materialization,) = result.asset_materializations_for_node("matchup_model")
    metadata = {key: value.value for key, value in materialization.metadata.items()}
    assert metadata["promotion_decision"] == "no_champion"
    assert metadata["promoted"] is True
    assert metadata["model_version"] == "1"
    assert metadata["gold_snapshot_id"] == str(current_snapshot_id(sql_catalog, SET_FEATURES))
    (check,) = result.get_asset_check_evaluations()
    assert check.check_name == "model_beats_baseline" and check.passed


def test_schedule_trains_once_per_gold_snapshot(resources, sql_catalog):
    gold = current_snapshot_id(sql_catalog, SET_FEATURES)
    request = daily_training(dg.build_schedule_context(resources=resources))
    assert isinstance(request, dg.RunRequest) and request.run_key == f"gold-{gold}"
    dg.materialize([matchup_model], resources=resources, run_config=RUN_CONFIG)
    skip = daily_training(dg.build_schedule_context(resources=resources))
    assert isinstance(skip, dg.SkipReason)


def test_training_schedule_runs_daily_at_7_pacific():
    assert daily_training.cron_schedule == "0 7 * * *"
    assert daily_training.execution_timezone == "America/Los_Angeles"
    assert daily_training.default_status == dg.DefaultScheduleStatus.RUNNING


def test_training_shares_the_lakehouse_pool():
    assert matchup_model.op.pool == "spark_lakehouse"
