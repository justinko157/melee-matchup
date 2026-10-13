from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[2] / "app.py"


def test_app_reports_an_unreachable_catalog(monkeypatch):
    monkeypatch.setenv("LAKEKEEPER_URL", "http://127.0.0.1:9/catalog")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    (error,) = app.error
    assert error.value.startswith("The data platform is unavailable: the lakehouse catalog")
