from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from pipeline.gold.tables import (  # noqa: E402
    HEAD_TO_HEAD,
    HEAD_TO_HEAD_SCHEMA,
    PLAYER_RATINGS,
    PLAYER_RATINGS_SCHEMA,
)
from pipeline.gold.writer import replace_table, to_arrow  # noqa: E402

APP = Path(__file__).resolve().parents[2] / "app.py"
TS = pd.Timestamp("2025-01-06", tz="UTC")
UNAVAILABLE = "The data platform is unavailable"


@pytest.fixture(autouse=True)
def fresh_app(monkeypatch):
    """AppTest shares the process-wide resource cache; start and end every test empty."""
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def _seed_gold(monkeypatch, sql_settings, sql_catalog, sets_played):
    ratings = pd.DataFrame(
        [
            {"player_id": pid, "gamer_tag": tag, "elo": elo, "sets_played": sets_played,
             "recent_wr": 0.5, "wins": 3, "losses": 3, "last_played_at": TS}
            for pid, tag, elo in [(1, "A", 1600.0), (2, "B", 1700.0)]
        ]
    )  # fmt: skip
    h2h = pd.DataFrame(
        [{"player_a_id": 1, "player_b_id": 2, "a_wins": 1, "b_wins": 2, "total": 3,
          "last_played_at": TS}]
    )  # fmt: skip
    replace_table(sql_catalog, PLAYER_RATINGS, to_arrow(ratings, PLAYER_RATINGS_SCHEMA), {})
    replace_table(sql_catalog, HEAD_TO_HEAD, to_arrow(h2h, HEAD_TO_HEAD_SCHEMA), {})
    monkeypatch.setenv("LAKEHOUSE_CATALOG_TYPE", "sql")
    monkeypatch.setenv("LAKEKEEPER_URL", sql_settings.uri)
    monkeypatch.setenv("LAKEHOUSE_WAREHOUSE", sql_settings.warehouse)


def _single_error(app) -> str:
    assert not app.exception
    (error,) = app.error
    return error.value


def test_app_reports_an_unreachable_catalog(monkeypatch):
    monkeypatch.setenv("LAKEKEEPER_URL", "http://127.0.0.1:9/catalog")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    (error,) = app.error
    assert error.value.startswith("The data platform is unavailable: the lakehouse catalog")


def test_app_reports_a_missing_champion(monkeypatch, sql_settings, sql_catalog, mlflow_tracking):
    _seed_gold(monkeypatch, sql_settings, sql_catalog, sets_played=10)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", mlflow_tracking)
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    message = _single_error(app)
    assert message.startswith(f"{UNAVAILABLE}: the MLflow champion model")
    assert "ChampionNotFound" in message


def test_app_reports_an_unreachable_mlflow(monkeypatch, sql_settings, sql_catalog):
    _seed_gold(monkeypatch, sql_settings, sql_catalog, sets_played=10)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    message = _single_error(app)
    assert message.startswith(f"{UNAVAILABLE}: the MLflow champion model")
    assert "ChampionNotFound" not in message


def test_app_reports_gold_without_rated_players(monkeypatch, sql_settings, sql_catalog):
    _seed_gold(monkeypatch, sql_settings, sql_catalog, sets_played=2)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert _single_error(app).startswith(f"{UNAVAILABLE}: the gold tables have no players")
